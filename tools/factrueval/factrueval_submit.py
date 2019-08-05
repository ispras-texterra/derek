import argparse
import json
from os.path import join
from os import makedirs

from derek.data.model import Document
from derek.data.readers import RawTextReader, FactRuEvalReader
from derek import transformer_from_props
from derek.ner import ChainedNERClassifier
from tools.common.argparsers import init_segmenter_argparser

ENTITIES = {"Location": "loc", "Person": "per", "Org": "org", "LocOrg": "locorg"}

def write_ner_results(doc: Document, entities: list, path: str):
    makedirs(path, exist_ok=True)

    with open(join(path, doc.name+".task1"), "w", encoding="utf-8") as f:
        for entity in entities:
            start_pos = doc.token_features['char_spans'][entity.start_token][0]
            end_pos = doc.token_features['char_spans'][entity.end_token-1][1]
            length = end_pos - start_pos
            result_type = ENTITIES[entity.type]
            f.write(" ".join([result_type, str(start_pos), str(length)])+"\n")

def build_argparser():
    parser = argparse.ArgumentParser(description='FactRuEval submitter')
    parser.add_argument('-model_path', type=str, dest='model_path', metavar='<model path>',
                        required=True, help='trained model path')
    parser.add_argument('-test_path', type=str, dest='test_path', metavar='<test path>',
                        required=True, help='raw txt test set path')
    parser.add_argument('-out_path', type=str, dest='out_path', metavar='<out path>',
                        required=True, help='output directory for results')
    parser.add_argument('-transformers_props_path', type=str, dest='transformers_props_path',
                        metavar='<transformers props path>', required=False, help='transformers props path')
    _, segmenter_factory = init_segmenter_argparser({"main": parser}, ["main"])
    return parser, segmenter_factory

def path_walker(path):
    doc_names = FactRuEvalReader.get_document_names(path)
    return [doc+".txt" for doc in doc_names]

def main():
    parser, segmenter_factory = build_argparser()
    args = parser.parse_args()
    segmenter = segmenter_factory(args)

    model_path = args.model_path
    raw_text_reader = RawTextReader(segmenter=segmenter)
    docs = raw_text_reader.read(args.test_path, path_walker)
    out_path = args.out_path
    transformers_props_path = args.transformers_props_path

    if transformers_props_path is not None:
        with open(transformers_props_path, 'r', encoding='utf-8') as f, \
                transformer_from_props(json.load(f)) as transformer:
            docs = [transformer.transform(doc) for doc in docs]

    with ChainedNERClassifier(model_path) as classifier:
        for doc in docs:
            entities = classifier.predict_doc(doc)
            write_ner_results(doc, entities, out_path)


if __name__ == '__main__':
    main()
