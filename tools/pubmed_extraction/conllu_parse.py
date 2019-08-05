import os
import argparse
from itertools import repeat
import multiprocessing
import re

from derek.data.model import Sentence, Document, Paragraph
from derek.data.babylondigger_processor import BabylonDiggerProcessor
from tools.doc_converters.digger_converter import convert_from_derek_to_digger

from babylondigger.evaluation.conllu_io import CoNLLUWriter
from babylondigger.wrappers.udpipe1_2 import UDPipeTaggerParser

class _ReadSegmenter:
    lines_regexp = re.compile("^.*$", re.MULTILINE)
    tokens_regexp = re.compile("[^\s]+")

    def segment(self, text):
        tokens = []
        sentences = []
        raw_tokens = []

        token_sent_start = 0
        for line_match in self.lines_regexp.finditer(text):
            raw_sent_start, raw_sent_end = line_match.span()
            sent_text = line_match.group()

            token_matches = list(self.tokens_regexp.finditer(sent_text))
            if not token_matches:
                continue

            sentences.append(Sentence(token_sent_start, token_sent_start + len(token_matches)))
            token_sent_start += len(token_matches)

            for token_match in token_matches:
                token_start, token_end = token_match.span()
                token_text = token_match.group()

                raw_tokens.append((raw_sent_start + token_start, raw_sent_start + token_end))
                tokens.append(token_text)

        return tokens, sentences, raw_tokens


def _conllu_text(text) -> str:
    tokens, sentences, _ = segmentor.segment(text)
    token_features = processor.get_token_features(tokens, sentences)
    # assume doc as paragraph
    paragraphs = [Paragraph(0, len(sentences))]

    doc = Document("", tokens, sentences, paragraphs, token_features=token_features)
    return writer.write_to_str(convert_from_derek_to_digger(doc))


def _conllu_directory(args):
    input_directory = args[0]
    output_directory = args[1]
    directory = args[2]

    output_path = os.path.join(output_directory, directory)
    os.makedirs(output_path)

    input_path = os.path.join(input_directory, directory)
    files = os.listdir(input_path)
    input_txts = [file for file in files if os.path.isfile(os.path.join(input_path, file)) and file.endswith(".txt")]

    for txt_file in input_txts:
        with open(os.path.join(input_path, txt_file), "r", encoding='utf-8') as f:
            text = f.read()

        with open(os.path.join(output_path, txt_file[:-len(".txt")] + ".conllu"), "w", encoding='utf-8') as f:
            f.write(_conllu_text(text))


def _init_processor(model_path):
    global segmentor
    global processor
    global writer
    segmentor = _ReadSegmenter()
    processor = BabylonDiggerProcessor.from_processor(UDPipeTaggerParser.load_udpipe(model_path))
    writer = CoNLLUWriter()


def main():
    parser = argparse.ArgumentParser(description='PMD and PMC plain text segmentor')

    parser.add_argument('-i', type=str, dest='input_directory', metavar='<input directory>',
                        required=True, help='directory with tokenized pubmed articles directories, sentence per line')

    parser.add_argument('-o', type=str, dest='output_directory', metavar='<output_directory>',
                        required=True, help='output directory')

    parser.add_argument('-u', type=str, dest='ud_path', metavar='<ud model path>',
                        required=True, help='UDPipe model path')

    args = parser.parse_args()

    directories = [name for name in os.listdir(args.input_directory)
                   if os.path.isdir(os.path.join(args.input_directory, name))]

    print("{} directories to be processed".format(len(directories)))
    processed = 0

    with multiprocessing.Pool(initializer=_init_processor, initargs=(args.ud_path,)) as p:
        pool_args = zip(
            repeat(args.input_directory), repeat(args.output_directory), directories)

        for _ in p.imap_unordered(_conllu_directory, pool_args):
            processed += 1
            if processed % 100 == 0:
                print("{} directories from {} processed".format(processed, len(directories)))


if __name__ == "__main__":
    main()
