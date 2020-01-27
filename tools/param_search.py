import json
import os
import shutil

from os.path import join
from typing import List

import argparse

from tools.common.helper import get_next_props, get_fold
from tools.common.read_conllu import read_conllu_file

from derek.data.readers import load
from derek.common.logger import init_logger
from derek import trainer_for, evaluator_for, transformer_from_props
from derek.data.model import Document

from derek.common.evaluation.results_controllers import ResultsStorage, get_best_model_picker, \
    history_improvement_controller
from derek.common.helper import FuncIterable


def dump_dict_as_json(dict_to_dump: dict, path: str):
    with open(path, 'w', encoding="utf-8") as f:
        f.write(json.dumps(dict_to_dump, indent=4, sort_keys=True))


def props_difference(base_props: dict, props: dict):
    ret = {}

    for key, val in props.items():
        base_val = base_props.get(key, None)

        if base_val is None:
            ret[key] = val
        elif base_val != val:
            if isinstance(val, dict) and isinstance(base_val, dict):
                ret[key] = props_difference(base_val, val)
            else:
                ret[key] = val

    return ret


def get_evaluating_hook(
        dev_docs: List[Document], train_docs: List[Document], evaluate, base_path: str, early_stopping_rounds: int):

    stats_path = join(base_path, "best_model_stats")
    os.makedirs(stats_path, exist_ok=True)
    model_path = join(base_path, "best_model")
    os.makedirs(model_path, exist_ok=True)

    picker, results_storage = get_best_model_picker(history_improvement_controller(early_stopping_rounds))

    def save_clf_and_dump_stats(classifier, stats_generator):
        classifier.save(model_path)

        if stats_generator is not None:
            for i, doc in enumerate(dev_docs):
                with open(join(stats_path, doc.name + '_stats.txt'), 'w', encoding='utf-8') as f:
                    f.write(stats_generator(i))

    def apply(classifier, epoch):
        print(f"Epoch {epoch}, dev results:")
        dev_main_score, dev_scores, stats_generator = evaluate(classifier, dev_docs, need_stats=True)
        print("Score={:.4f}".format(dev_main_score))
        print()

        stopped = picker(dev_main_score, dev_scores, lambda: save_clf_and_dump_stats(classifier, stats_generator))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, train results:")
            train_main_score, train_scores, _ = evaluate(classifier, train_docs, need_stats=False)
            print("Score={:.4f}".format(train_main_score))
            print()
        else:
            train_scores = None

        scores = {'dev': dev_scores}
        if train_scores is not None:
            scores['train'] = train_scores
        with open(join(base_path, 'results.txt'), 'a', encoding='utf-8') as f:
            f.write(json.dumps({f"epoch_{epoch}": scores}, indent=4, sort_keys=True))
            f.write('\n\n')

        return stopped

    return apply, results_storage


def create_argparser():
    parser = argparse.ArgumentParser(description='Hyperparameter optimizer')
    parser.add_argument('-task', type=str, dest='task_name', metavar='<task name>',
                        required=True, help='name of the task: one of {ner, net, rel_ext}')
    parser.add_argument('-props', type=str, dest='props', metavar='<props.json>',
                        required=True, help='props.json with base props')
    parser.add_argument('-lst', type=str, dest='lst', metavar='<lst.json>',
                        required=True, help='lst.json with props to optimize')
    parser.add_argument('-seeds', type=int, dest='seeds', metavar='<seeds>',
                        required=True, help='number of random seeds')
    parser.add_argument('-out', type=str, dest='out_dir', metavar='<output directory>',
                        required=True, help='directory where results are stored')
    parser.add_argument('-unlabeled', type=str, dest='unlabeled', metavar='<unlabeled path>',
                        required=False, default=None, help='path to CoNLL-U file with unlabeled data')

    strategies = parser.add_subparsers(dest='strategy')
    strategies.required = True

    holdout = strategies.add_parser("holdout")
    holdout.add_argument('-train', type=str, dest='train_dir', metavar='<train directory>',
                         required=True, help='directory with train docs.pkl')
    holdout.add_argument('-dev', type=str, dest='dev_dir', metavar='<dev directory>',
                         required=True, help='directory with dev docs.pkl')

    cross_validation = strategies.add_parser("cross_validation")
    cross_validation.add_argument('-traindev', type=str, dest='traindev_dir', metavar='<traindev directory>',
                                  required=True, help='directory with train/dev docs.pkl')
    cross_validation.add_argument('-folds', type=int, dest='folds', metavar='<folds>',
                                  required=True, help='number of folds')

    return parser


def get_props_to_evaluate(args):
    with open(args.props) as f:
        props_base = json.load(f)
    with open(args.lst) as f:
        props_list = json.load(f)
    return props_base, list(get_next_props(props_base, props_list))


def load_docs(args):
    class HoldoutDataset:
        def __init__(self, train_docs, dev_docs):
            self.__train_docs = train_docs
            self.__dev_docs = dev_docs

        def get_splits(self):
            yield (self.__train_docs, self.__dev_docs)

        @property
        def splits_number(self):
            return 1

        def transformed_by(self, transformer):
            return HoldoutDataset([transformer.transform(doc) for doc in self.__train_docs],
                                  [transformer.transform(doc) for doc in self.__dev_docs])

    class CVDataset:
        def __init__(self, docs):
            self.__docs = docs

        def get_splits(self):
            for i in range(args.folds):
                yield get_fold(self.__docs, args.folds, i)

        @property
        def splits_number(self):
            return args.folds

        def transformed_by(self, transformer):
            return CVDataset([transformer.transform(doc) for doc in self.__docs])

    if args.strategy == 'holdout':
        dataset = HoldoutDataset(load(args.train_dir), load(args.dev_dir))
    elif args.strategy == 'cross_validation':
        dataset = CVDataset(load(args.traindev_dir))
    else:
        raise Exception('Only holdout and cross_validation strategies are supported')

    return dataset, FuncIterable(lambda: read_conllu_file(args.unlabeled)) if args.unlabeled is not None else None


def seeds_cycle(task_name, seeds_num, props, props_idx, split_idx, dev_docs, train_docs, unlabeled_docs, path):
    seeds_results = ResultsStorage()

    for seed_idx in range(seeds_num):
        seed = 100 * (1 + seed_idx)
        cur_seed_path = join(path, f'seed_{seed_idx}')
        os.makedirs(cur_seed_path, exist_ok=True)

        hook, results = get_evaluating_hook(
            dev_docs, train_docs, evaluator_for(task_name), cur_seed_path,
            props.get("early_stopping_rounds", props["epoch"]))

        with trainer_for(task_name)({**props, "seed": seed}) as trainer:
            trainer.train(train_docs, unlabeled_docs, hook)

        best_main_score, best_scores = results.best_scores
        epoch_num = results.best_score_idx + 1  # epoch numbers begin with 1

        print(f"Best score for props #{props_idx}, split #{split_idx}, seed #{seed_idx}: "
              f"{best_main_score:.4f}, epoch: {epoch_num}")
        print()

        dump_dict_as_json({**best_scores, "epoch": epoch_num}, join(cur_seed_path, "best_results.json"))
        seeds_results.add_scores(best_main_score, best_scores)

    mean_main_score, mean_scores = seeds_results.mean_scores
    _, best_scores = seeds_results.best_scores

    print(f"Mean score for props #{props_idx}, split #{split_idx}: {mean_main_score:.4f}")
    print()

    dump_dict_as_json(mean_scores, join(path, "mean_results.json"))
    dump_dict_as_json({**best_scores, "seed_num": seeds_results.best_score_idx}, join(path, "best_results.json"))

    shutil.copytree(join(path, f'seed_{seeds_results.best_score_idx}'), join(path, 'best_seed'))

    return mean_main_score, mean_scores


def splits_cycle(task_name, seeds_num, props, props_idx, dataset, unlabeled_docs, path):
    splits_results = ResultsStorage()

    for split_idx, (train_docs, dev_docs) in enumerate(dataset.get_splits()):
        cur_split_path = join(path, f'split_{split_idx}')
        os.makedirs(cur_split_path, exist_ok=True)

        mean_main_score, mean_scores = seeds_cycle(
            task_name, seeds_num, props, props_idx, split_idx, dev_docs, train_docs, unlabeled_docs,
            cur_split_path)

        splits_results.add_scores(mean_main_score, mean_scores)

    mean_main_score, mean_scores = splits_results.mean_scores
    print(f"Overall mean split score for props #{props_idx}: {mean_main_score:.4f}")
    print()

    dump_dict_as_json(mean_scores, join(path, "mean_results.json"))

    return mean_main_score, mean_scores


def get_experiments_report(results: ResultsStorage, base_props: dict, props: List[dict], top_k=5):
    def diff_to_set_of_tuples(diff: dict):
        tuples_set = set()

        for key, val in diff.items():
            key_as_tuple = (key, )
            if not isinstance(val, dict):
                tuples_set.add(key_as_tuple)
            else:
                tuples_set.update(key_as_tuple + t for t in diff_to_set_of_tuples(val))

        return tuples_set

    def remove_from_base(base: dict, t: tuple):
        deleted = dict(base)

        if len(t) == 0:
            return deleted

        if t[0] in deleted:
            if len(t) > 1:
                new_inner_dict = remove_from_base(deleted[t[0]], t[1:])
                if new_inner_dict:
                    deleted[t[0]] = new_inner_dict
                else:
                    del deleted[t[0]]
            else:
                del deleted[t[0]]

        return deleted

    main_scores, scores_dicts = results.main_scores, results.scores_dicts
    top_k_indexes = [idx for idx, _ in sorted(enumerate(main_scores), key=lambda t: -t[1])[:top_k]]

    differences = [props_difference(base_props, props[idx]) for idx in top_k_indexes]
    differences_keys = [diff_to_set_of_tuples(d) for d in differences]
    intersecting_keys = set.intersection(*differences_keys)

    for k in intersecting_keys:
        base_props = remove_from_base(base_props, k)

    ret = {"base_props": base_props, "top_props": []}

    for i, props_idx in enumerate(top_k_indexes):
        props_d = {
            "props_idx": props_idx,
            "difference": differences[i],
            "scores": scores_dicts[props_idx]
        }
        ret["top_props"].append(props_d)

    return ret


def main():
    args = create_argparser().parse_args()
    base_props, props_to_evaluate = get_props_to_evaluate(args)
    dataset, unlabeled_docs = load_docs(args)
    props_picker, props_best_results = get_best_model_picker(history_improvement_controller(len(props_to_evaluate)))

    if not props_to_evaluate:
        print("No props found")
        return

    if os.path.exists(args.out_dir) and (not os.path.isdir(args.out_dir) or os.listdir(args.out_dir)):
        print("Output path should either not exists or be empty directory")
        return

    for props_idx, props in enumerate(props_to_evaluate):
        cur_props_path = join(args.out_dir, f'props_{props_idx}')
        os.makedirs(cur_props_path, exist_ok=True)
        dump_dict_as_json(props, join(cur_props_path, 'props.json'))

        with transformer_from_props(props.get("transformers", {})) as t:
            tr_dataset = dataset.transformed_by(t)
            tr_unlabeled_docs = [t.transform(doc) for doc in unlabeled_docs] if unlabeled_docs is not None else None

        mean_main_score, mean_scores = splits_cycle(
            args.task_name, args.seeds, props, props_idx, tr_dataset, tr_unlabeled_docs, cur_props_path)
        props_picker(mean_main_score, mean_scores, lambda: None)

    best_props_idx = props_best_results.best_score_idx
    best_main_score, best_scores = props_best_results.best_scores
    print(f"Overall experiment best score: {best_main_score:.4f}, props: #{best_props_idx}")

    best_props_path = join(args.out_dir, 'best_props')
    os.makedirs(best_props_path)

    for split_idx in range(dataset.splits_number):
        split_path = join(args.out_dir, f'props_{best_props_idx}', f'split_{split_idx}')
        split_best_seed = join(split_path, 'best_seed')

        shutil.copytree(split_best_seed, join(best_props_path, f'split_{split_idx}'))
        shutil.copy(join(split_path, 'mean_results.json'), join(best_props_path, f'split_{split_idx}'))

    dump_dict_as_json({**best_scores, "props_num": best_props_idx}, join(args.out_dir, "best_results.json"))
    dump_dict_as_json(
        get_experiments_report(props_best_results, base_props, props_to_evaluate),
        join(args.out_dir, "experiments_report.json"))


if __name__ == '__main__':
    init_logger('logger')
    main()
