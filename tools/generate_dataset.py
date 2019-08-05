from os.path import join
import json

from derek.coref.data.readers import RuCorDataReader
from derek.data.readers import BioNLPDataReader, ChemProtDataReader, BRATReader, FactRuEvalReader, CoNLLReader
from derek import transformer_from_props
from tools.common.argparsers import init_segmenter_argparser, get_dataset_directories_argparser


def build_argparser():
    def transformer_factory(args):
        if args.transformer_props is not None:
            with open(args.transformer_props, 'r', encoding='utf-8') as f:
                props = json.load(f)
        else:
            props = {}
        return transformer_from_props(props)

    segmentation_required = ["BioNLP", "ChemProt", "BRAT"]

    parser, input_dirs_factory = get_dataset_directories_argparser('Generate dataset')
    parser.add_argument('-transformer_props', type=str, dest='transformer_props', metavar='<transformers.json>',
                        required=False, help='path to transformer props')

    parsers = {}
    subparsers = parser.add_subparsers(dest='reader')
    subparsers.required = True
    parsers["BioNLP"] = subparsers.add_parser('BioNLP')

    parsers["ChemProt"] = subparsers.add_parser('ChemProt')
    parsers["ChemProt"].add_argument('-cpr', dest='use_cpr', action='store_true',
                                     help='use granular relation types instead of relation groups')
    parsers["ChemProt"].add_argument('--extra', dest='read_N', action='store_true',
                                     help='read extra annotations (for more details see dataset readme file)')
    parsers["BRAT"] = subparsers.add_parser("BRAT")
    parsers["BRAT"].add_argument('-collapse', dest='collapse', action='store_true',
                             help='collapse intersecting annotations')
    parsers["RuCor"] = subparsers.add_parser("RuCor")
    parsers["RuCor"].add_argument('-fix_entity_types', action='store_true',
                                  help='Compute entity types instead of reading')
    parsers["FactRuEval"] = subparsers.add_parser("FactRuEval")
    parsers["FactRuEval"].add_argument('-collapse', dest='collapse', action='store_true',
                             help='collapse intersecting annotations')
    parsers["FactRuEval"].add_argument('-nolocorg', dest='locorg', action='store_false',
                             help='use locorg class annotations or location instead')
    parsers["FactRuEval"].add_argument('-blacklist', dest='blacklist', nargs='+', type=str, default=set(),
                                       help='list of annotations classes to exclude')
    parsers["CoNLL"] = subparsers.add_parser("CoNLL")

    _, segmenter_factory = init_segmenter_argparser(parsers, segmentation_required)

    return parser, input_dirs_factory, segmenter_factory, transformer_factory


def main():
    parser, input_dirs_factory, segmenter_factory, transformer_factory = build_argparser()
    args = parser.parse_args()

    segmenter = segmenter_factory(args)

    if args.reader == 'BioNLP':
        print("Reading as BioNLP dataset")
        reader = BioNLPDataReader(segmenter)
    elif args.reader == 'ChemProt':
        print("Reading as ChemProt dataset")
        reader = ChemProtDataReader(segmenter, read_N=args.read_N, use_cpr=args.use_cpr)
    elif args.reader == 'BRAT':
        print("Reading as BRAT dataset")
        reader = BRATReader(segmenter, collapse_intersecting=args.collapse)
    elif args.reader == 'RuCor':
        reader = RuCorDataReader(fix_entity_types=args.fix_entity_types)
    elif args.reader == 'FactRuEval':
        reader = FactRuEvalReader(collapse_intersecting=args.collapse, locorg_allowed=args.locorg,
                                  blacklist=args.blacklist)
    elif args.reader == "CoNLL":
        reader = CoNLLReader()
    else:
        raise ValueError()

    with transformer_factory(args) as transformer:
        for collection_name, input_directory in input_dirs_factory(args):

            print(f"Reading {collection_name if collection_name is not None else ''} collection...")

            output_directory = args.output_directory
            if collection_name is not None:
                output_directory = join(output_directory, collection_name)
            docs = reader.read(input_directory)
            docs = [transformer.transform(doc) for doc in docs]
            reader.dump(output_directory, docs)


if __name__ == '__main__':
    main()
