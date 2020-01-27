import os
import re
import subprocess
import sys
from collections import defaultdict
from logging import getLogger

import numpy as np

from derek.common.evaluation.metrics import f1_score
from derek.coref.data.chains_collection import get_collecting_strategy, collect_easy_first_mention, \
    collect_pron_vote_rank
from derek.coref.data.conll_serializer import CoNLLSerializer
from derek.data.readers import load
from derek.coref import CorefClassifier as Classifier, CorefTrainer as Trainer
from tools.common.helper import params_to_str, get_fold


def get_metrics(path):
    metric_re = 'METRIC (.+?):(.+?)====== TOTALS =======.+?Coreference:(.+?)\n'
    f1_re = 'F1: (.+?)%'
    recall_re = 'Recall: \(.+?\) (.+?)%'
    precision_re = 'Precision: \(.+?\) (.+?)%'
    results = {}

    with open(path) as f:
        text = f.read()
    for match in re.finditer(metric_re, text, flags=re.DOTALL):

        results[match.group(1) + '_micro'] = {
            'f1': float(re.findall(f1_re, match.group(3))[0]),
            'recall': float(re.findall(recall_re, match.group(3))[0]),
            'precision': float(re.findall(precision_re, match.group(3))[0]),
        }
        if match.group(1) != 'blanc':
            precision_macro = np.mean(list(map(lambda x: float(x), re.findall(precision_re, match.group(2)))))
            recall_macro = np.mean(list(map(lambda x: float(x), re.findall(recall_re, match.group(2)))))
            f1_macro = np.mean(list(map(lambda x: float(x), re.findall(f1_re, match.group(2)))))

            results[match.group(1) + '_macro_isp'] = {
                'f1': f1_score(precision_macro, recall_macro),
                'recall': recall_macro,
                'precision': precision_macro,
            }

            results[match.group(1) + '_macro'] = {
                'f1': f1_macro,
                'recall': recall_macro,
                'precision': precision_macro,
            }
    return results


def eval_model(gold_path, pred_path, result_path):
    command = "ext/scorer/scorer.pl all " + gold_path + ' ' + pred_path + ' > ' + result_path
    subprocess.check_output(command, shell=True).decode()
    metrics = get_metrics(result_path)

    return metrics


def get_strategy_rels(strategies, pairs_dict, docs_known_rels):
    ret = {}
    for strategy_name in strategies:
        if strategy_name == 'pron_vote_rank':
            rels = [collect_pron_vote_rank(pairs_dict[doc.name][-1], doc.relations) for doc in docs_known_rels]
        elif strategy_name != 'default':
            strategy = get_collecting_strategy(strategy_name)
            rels = [strategy(pairs_dict[doc.name][-1]) for doc in docs_known_rels]
        else:
            rels = [pairs_dict[doc.name][0] for doc in docs_known_rels]
        for i, doc in enumerate(docs_known_rels):
            try:
                rels[i] |= doc.relations
            except ValueError:
                pass
        ret[strategy_name] = rels
    return ret


def get_evaluating_hook(serializer, dev_docs, out_path, seed, fold, save_models, strategies, known_rels=None):
    best_result = defaultdict(dict)
    evaluation_result = defaultdict(dict)

    def evaluate(classifier, epoch):
        # ensure that we do not use gold labels for prediction
        docs_no_rels = [doc.without_relations() for doc in dev_docs]
        if known_rels is not None:
            docs_known_rels = [doc.with_relations(doc_rels) for doc, doc_rels in zip(docs_no_rels, known_rels)]
        else:
            docs_known_rels = docs_no_rels

        pairs_dict = classifier.predict_docs(docs_known_rels, include_probs=True, print_progress=True)
        gold_path = os.path.join(out_path, 'gold.conll')
        strategy_rels = get_strategy_rels(strategies, pairs_dict, docs_known_rels)
        print('Epoch', epoch, 'results:')
        for strategy_name in strategies:
            rels = strategy_rels[strategy_name]
            pred_path = os.path.join(out_path, '_'.join([str(epoch), strategy_name, 'annotated.conll']))
            result_path = os.path.join(out_path, '_'.join([str(epoch), strategy_name, 'result.txt']))

            with open(pred_path, 'w', encoding="utf-8") as f:
                docs_pred_rels = [doc.with_relations(doc_rels) for doc, doc_rels in zip(docs_no_rels, rels)]
                serializer.serialize_docs(docs_pred_rels, f)

            try:
                result = eval_model(gold_path, pred_path, result_path)
            except Exception:
                getLogger('logger').error("Model evaluation exception!")
                with open('diagnosis.log', 'w') as f:
                    for rel in rels:
                        f.write(str(rel) + '\n')
                return
            for metric, score in result.items():
                metric = metric + '_' + strategy_name
                print(metric, score)
                evaluation_result[metric].setdefault(str((seed, fold)), {})[epoch] = score
                if best_result[metric].setdefault('f1', 0) < score['f1']:
                    best_result[metric] = score
                    if save_models:
                        classifier.save(os.path.join(out_path, 'best_'+metric))
            print()
        with open(os.path.join(sys.argv[1], 'status'), 'w') as f:
            f.write(f'Epoch {epoch}')

        return False

    return evaluate, evaluation_result


def get_known_rels(classifier, docs):
    # ensure that we do not use gold labels for prediction
    docs_no_rels = [doc.without_relations() for doc in docs]

    pairs_dict = classifier.predict_docs(docs_no_rels, include_probs=True, print_progress=True)
    return [collect_easy_first_mention(pairs_dict[doc.name][-1]) for doc in docs]


class CorefTrainer:

    def train(self, props: dict, params: dict, working_dir: str):
        docs = load(params['data_path'])
        serializer = CoNLLSerializer()

        n_folds, fold_num, seed = params['n_folds'], props['fold_num'], props['seed']
        out_path = os.path.join(working_dir, str(fold_num) + '_fold', str(seed))

        param_str = params_to_str(props)
        print(param_str)

        os.makedirs(out_path, exist_ok=True)

        train_docs, dev_docs = get_fold(docs, n_folds, fold_num)
        print("Fold:", fold_num)

        with open(os.path.join(out_path, 'gold.conll'), 'w', encoding="utf-8") as f:
            serializer.serialize_docs(dev_docs, f)

        classifier_path = props.get('classifier_path')
        if classifier_path is not None and props.get('sampling_strategy', 'coref') in ['coref_pron',
                                                                                       'coref_pron_cluster',
                                                                                       'coref_pron_cluster_strict']:
            with Classifier(classifier_path) as clf:
                print("Applying known model")
                known_rels = get_known_rels(clf, dev_docs)
            strategies = ['pron_rank', 'pron_vote_rank']
        else:
            known_rels = None
            strategies = ['easy_first']

        hook, evaluation_result = get_evaluating_hook(serializer, dev_docs, out_path, seed, fold_num,
                                                      params.get('save_models', False), strategies, known_rels)

        with Trainer(props) as trainer:
            trainer.train(train_docs, hook)

        return dict(evaluation_result)
