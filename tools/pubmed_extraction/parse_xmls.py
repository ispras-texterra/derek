import os
import argparse
from itertools import repeat
import multiprocessing
from bs4 import BeautifulSoup as BS

PROHIBITED_TITLES = {
    "supplemental digital content", "electronic supplementary material", "web resource url",
    "electronic supplementary materials"
}

PROHIBITED_TITLE_WORDS = {
    "registration", "registrations", "registry", "clinicaltrials.gov", "trial", "trials"
}

PROHIBITED_TAGS = {
    "inline-formula"
}

# make sure that titles and words are lowercased
PROHIBITED_TITLES = set(title.lower() for title in PROHIBITED_TITLES)
PROHIBITED_TITLE_WORDS = set(word.lower() for word in PROHIBITED_TITLE_WORDS)


def _is_bad_title(title):
    title = title.lower()
    return title in PROHIBITED_TITLES or any(word in title for word in PROHIBITED_TITLE_WORDS)


def _get_ps_from_sections(sections):
    ps = []

    for sec in sections:
        if sec.title is not None and sec.title.string is not None and _is_bad_title(sec.title.string):
            continue

        ps.extend(sec.find_all("p"))

    return ps


def _get_plain_text_from_ps(ps):
    if not ps:
        return None
    p_texts = []

    for p in ps:
        text = ""
        for tag in p:
            if tag.name not in PROHIBITED_TAGS and tag.string is not None:
                tag_text = tag.string.strip('\n')
                if tag_text:
                    text += tag_text

        # obtrusive invisible char
        text = text.replace("\xad", "")
        p_texts.append(text)

    return '\n\n'.join(p_texts)


def get_abstract(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        xml = BS(f, "xml")

    abstract = xml.find("abstract", {"abstract-type": None})

    if abstract is None:
        return None

    sections = abstract.find_all("sec")

    if sections:
        ps = _get_ps_from_sections(sections)
        # add <p> children from <abstract>
        ps += abstract.find_all("p", recursive=False)
    else:
        ps = abstract.find_all("p")

    return _get_plain_text_from_ps(ps)


def extract_abstracts_from_path(directory_path):
    files = os.listdir(directory_path)
    xmls = [file for file in files if os.path.isfile(os.path.join(directory_path, file)) and file.endswith(".nxml")]
    abstracts = {}

    for xml in xmls:
        abstract = get_abstract(os.path.join(directory_path, xml))
        if abstract:
            abstracts[xml] = abstract

    return abstracts


def extract(args):
    input_directory = args[0]
    output_directory = args[1]
    directory = args[2]

    directory_abstracts = extract_abstracts_from_path(os.path.join(input_directory, directory))

    if directory_abstracts:
        abstracts_directory = os.path.join(output_directory, directory)
        os.makedirs(abstracts_directory)
        for xml, abstract in directory_abstracts.items():
            with open(os.path.join(abstracts_directory, xml[:-len(".nxml")] + ".txt"), "w", encoding="utf-8") as f:
                f.write(abstract)


def main():
    parser = argparse.ArgumentParser(description='PMD and PMC xmls abstracts plain text extractor')

    parser.add_argument('-i', type=str, dest='input_directory', metavar='<input directory>',
                        required=True, help='directory with pubmed articles directories')

    parser.add_argument('-o', type=str, dest='output_directory', metavar='<output_directory>',
                        required=True, help='output directory')
    args = parser.parse_args()

    directories = [name for name in os.listdir(args.input_directory)
                   if os.path.isdir(os.path.join(args.input_directory, name))]

    print("{} directories to be processed".format(len(directories)))
    processed = 0

    with multiprocessing.Pool() as p:
        pool_args = zip(repeat(args.input_directory), repeat(args.output_directory), directories)

        for _ in p.imap_unordered(extract, pool_args):
            processed += 1
            if processed % 100 == 0:
                print("{} directories from {} processed".format(processed, len(directories)))


if __name__ == "__main__":
    main()
