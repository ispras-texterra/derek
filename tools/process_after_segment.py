import os
import argparse
from itertools import repeat
import multiprocessing

from derek.data.processing_helper import StandardTokenProcessor


def _init_processor(factory):
    global processor
    processor = factory()


def _process_file(args):
    input_directory = args[0]
    output_directory = args[1]
    file_path = args[2]

    input_file_path = os.path.join(input_directory, file_path)
    output_file_path = os.path.join(output_directory, file_path)
    output_file_directory = os.path.dirname(output_file_path)

    if output_directory != output_file_directory:
        os.makedirs(output_file_directory, exist_ok=True)

    with open(input_file_path, "r", encoding='utf-8') as f:
        text = f.read()

    with open(output_file_path, "w", encoding='utf-8') as f:
        f.write(_process_raw_text(text))


def _process_raw_text(text: str) -> str:
    paragraphs_processed = []

    for paragraph in text.split('\n'):
        paragraph_tokens = [processor(token) for token in paragraph.split(" ")]
        paragraphs_processed.append(' '.join(paragraph_tokens))

    return '\n'.join(paragraphs_processed)


def main():
    parser = argparse.ArgumentParser(description='Segmented text preprocessor')

    parser.add_argument('-i', type=str, dest='input_directory', metavar='<input directory>',
                        required=True, help='directory with txt files')

    parser.add_argument('-o', type=str, dest='output_directory', metavar='<output_directory>',
                        required=True, help='output directory')

    parser.add_argument('-zeros', action='store_true', required=False, help='replace digits with zeros')
    parser.add_argument('-quotes', action='store_true', required=False, help='replace different quotes with \"')
    parser.add_argument('-lower', action='store_true', required=False, help='lowercase text')

    args = parser.parse_args()

    input_files = [
                os.path.join(parent, name)
                for (parent, subdirs, files) in os.walk(args.input_directory)
                for name in files + subdirs if name.endswith(".txt") and os.path.isfile(os.path.join(parent, name))
    ]

    input_files = [os.path.relpath(f, args.input_directory) for f in input_files]

    processor_factory = lambda: StandardTokenProcessor(args.lower, args.zeros, args.quotes)

    print(f"{len(input_files)} files to be processed")
    processed = 0

    with multiprocessing.Pool(initializer=_init_processor, initargs=(processor_factory,)) as p:
        pool_args = zip(repeat(args.input_directory), repeat(args.output_directory), input_files)

        for _ in p.imap_unordered(_process_file, pool_args, chunksize=1000):
            processed += 1
            if processed % 1000 == 0:
                print(f"{processed} files from {len(input_files)} processed")


if __name__ == "__main__":
    main()
