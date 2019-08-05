import json
import os

from derek.data.readers import load, dump
from tools.common.argparsers import get_dataset_directories_argparser
from derek import preprocessor_for


def __create_argparser():
    parser, input_dirs_factory = get_dataset_directories_argparser('Dataset preprocessor')
    parser.add_argument('-task', type=str, dest='task_name', metavar='<task name>',
                        required=True, help='name of the task: one of {ner, net}')
    parser.add_argument('-props', type=str, dest='props', metavar='<props.json>',
                        required=True, help='props.json with parameters for preprocessor')
    return parser, input_dirs_factory


def main():
    parser, input_dirs_factory = __create_argparser()
    args = parser.parse_args()

    with open(args.props, "r", encoding="utf-8") as f:
        props = json.load(f)

    preprocessor = preprocessor_for(args.task_name, props)

    for collection_name, input_directory in input_dirs_factory(args):
        output_directory = args.output_directory
        if collection_name is not None:
            output_directory = os.path.join(output_directory, collection_name)

        docs = load(input_directory)
        processed_docs = list(map(preprocessor.process_doc, docs))
        dump(output_directory, processed_docs)


if __name__ == '__main__':
    main()
