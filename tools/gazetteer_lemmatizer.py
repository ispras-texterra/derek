import json

from argparse import ArgumentParser
from os.path import isfile

from derek import transformer_from_props
from derek.data.model import Document, Sentence, Paragraph


def build_argparser():
    parser = ArgumentParser(description="Lemmatize gazetteer file")
    parser.add_argument('-input', type=str, dest='input_path', metavar='<input_file>',
                        required=True, help='path to input gazetteer file to lemmatize')
    parser.add_argument('-transformer_props', type=str, dest='transformer_props', metavar='<transformers.json>',
                        required=True, help='path to transformer props')
    parser.add_argument('-output', type=str, dest='output_path', metavar='<output_file>',
                        required=True, help='path to save lemmatized gazetteer')
    parser.add_argument('-lower', dest='lower', action="store_true",
                        required=False, help='lowercase after lemmatization')
    return parser


def _get_lemma(token, transformer, to_lowercase):
    doc = Document("", [token], [Sentence(0, 1)], [Paragraph(0, 1)])
    featured_doc = transformer.transform(doc)
    lemma = featured_doc.token_features['lemmas'][0]
    return lemma.lower() if to_lowercase else lemma


def lemmatize(input_path, output_path, transformers_props_path, to_lowercase):
    with open(transformers_props_path, 'r', encoding='utf-8') as f, \
            transformer_from_props(json.load(f)) as transformer, \
            open(input_path, 'r', encoding='utf-8') as readfile, \
            open(output_path, 'w', encoding='utf-8', newline='\n') as outfile:
        for line in readfile:
            lemma = _get_lemma(line.strip(), transformer, to_lowercase)
            outfile.write(lemma + "\n")


def main():
    parser = build_argparser()
    args = parser.parse_args()
    input_path = args.input_path
    out_path = args.output_path
    to_lowercase = args.lower
    transformers_props_path = args.transformer_props
    if isfile(input_path):
        lemmatize(input_path, out_path, transformers_props_path, to_lowercase)
    else:
        raise FileNotFoundError()


if __name__ == '__main__':
    main()
