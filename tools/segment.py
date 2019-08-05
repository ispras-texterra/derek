import os
import argparse
from itertools import repeat
import multiprocessing
from tools.common.argparsers import init_segmenter_argparser


def _init_segmenter(factory):
    global segmenter
    segmenter = factory()


def _segment_file(args):
    input_directory = args[0]
    output_directory = args[1]
    sent_on_each_line = args[2]
    file_path = args[3]

    input_file_path = os.path.join(input_directory, file_path)
    output_file_path = os.path.join(output_directory, file_path)
    output_file_directory = os.path.dirname(output_file_path)

    if output_directory != output_file_directory:
        os.makedirs(output_file_directory, exist_ok=True)

    with open(input_file_path, "r", encoding='utf-8') as f:
        text = f.read()

    with open(output_file_path, "w", encoding='utf-8') as f:
        f.write(_segment_raw_text(text, sent_on_each_line))


def _segment_raw_text(text: str, sent_on_each_line: bool) -> str:
    paragraphs_segmented = []

    for paragraph in text.split('\n'):
        if not paragraph:
            tokens, sentences = [], []
        else:
            tokens, sentences, _ = segmenter.segment(paragraph)

        if sent_on_each_line:
            paragraphs_segmented.append('\n'.join(' '.join(tokens[s.start_token:s.end_token]) for s in sentences))
        else:
            paragraphs_segmented.append(' '.join(tokens))

    return '\n'.join(paragraphs_segmented)


def main():
    parser = argparse.ArgumentParser(description='Plain text segmentor')

    parser.add_argument('-i', type=str, dest='input_directory', metavar='<input directory>',
                        required=True, help='directory with txt files')

    parser.add_argument('-o', type=str, dest='output_directory', metavar='<output_directory>',
                        required=True, help='output directory')

    parser.add_argument('-s', dest='sent', action='store_true', help='write single sentence on each line')

    _, segmenter_factory = init_segmenter_argparser({"main": parser}, ["main"])
    args = parser.parse_args()

    input_files = [
                os.path.join(parent, name)
                for (parent, subdirs, files) in os.walk(args.input_directory)
                for name in files + subdirs if name.endswith(".txt") and os.path.isfile(os.path.join(parent, name))
    ]

    input_files = [os.path.relpath(f, args.input_directory) for f in input_files]

    print(f"{len(input_files)} files to be processed")
    processed = 0

    with multiprocessing.Pool(initializer=_init_segmenter, initargs=(lambda: segmenter_factory(args), )) as p:
        pool_args = zip(repeat(args.input_directory), repeat(args.output_directory), repeat(args.sent), input_files)

        for _ in p.imap_unordered(_segment_file, pool_args, chunksize=1000):
            processed += 1
            if processed % 1000 == 0:
                print(f"{processed} files from {len(input_files)} processed")


if __name__ == "__main__":
    main()
