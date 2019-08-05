import json
from argparse import ArgumentParser
from os import makedirs
from os.path import join

from tools.common.helper import get_fold
from derek.data.readers import load
from derek import evaluator_for, classifier_for, transformer_from_props


def create_argparser():
    parser = ArgumentParser(description="Model evaluator")
    parser.add_argument("task_name", type=str, metavar='<task_name>', help='task to evaluate')
    parser.add_argument("docs_path", type=str, metavar='<docs.pkl path>', help='path to docs to evaluate on')
    parser.add_argument("-transformers_props_path", required=False, type=str, metavar='<transformers.json>',
                        help='path to model transformers props')
    parser.add_argument("-stats_path", type=str, required=False, metavar='<stats path>',
                        help='optional path to save stats')

    strategies = parser.add_subparsers(dest='strategy')
    strategies.required = True

    holdout = strategies.add_parser("holdout")
    holdout.add_argument("model_path", type=str, metavar='<model_path>', help='path to model')

    cross_validation = strategies.add_parser("cross_validation")
    cross_validation.add_argument('splits_model_paths', nargs="+", metavar='<splits_model_paths>',
                                  help='model path for each split')

    return parser


def main():
    args = create_argparser().parse_args()
    docs = load(args.docs_path)
    evaluator = evaluator_for(args.task_name)

    if args.transformers_props_path is not None:
        with open(args.transformers_props_path, "r", encoding="utf-8") as f, \
                transformer_from_props(json.load(f)) as transformer:
            docs = [transformer.transform(doc) for doc in docs]

    if args.strategy == "holdout":
        folds_num = 1
        models = [args.model_path]
    else:
        folds_num = len(args.splits_model_paths)
        models = args.splits_model_paths

    main_scores = []
    for split_idx, model_path in enumerate(models):
        _, test_docs = get_fold(docs, folds_num, split_idx)

        with classifier_for(args.task_name)(model_path) as clf:
            main_score, scores, stats_generator = evaluator(clf, test_docs, args.stats_path is not None)
            main_scores.append(main_score)

            print("Split {}, Main score={:.4f}".format(split_idx, main_score))
            print(f"Scores: \n{json.dumps(scores, indent=4, sort_keys=True)}\n")

            if stats_generator is not None:
                stats_path = join(args.stats_path, f"split_{split_idx}")
                makedirs(stats_path, exist_ok=True)

                for doc_idx, doc in enumerate(test_docs):
                    with open(join(stats_path, doc.name + '_stats.txt'),
                              'w', encoding='utf-8') as f:

                        f.write(stats_generator(doc_idx))

    print("\nMean splits score={:.4f}".format(sum(main_scores) / len(main_scores)))


if __name__ == '__main__':
    main()
