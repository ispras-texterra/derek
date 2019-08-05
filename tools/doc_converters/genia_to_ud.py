import os
import sys

import StanfordDependencies
from bs4 import BeautifulSoup as bs, NavigableString

# These are control chars in PTB bracket format
replacements = {
    '(': '-LRB-',
    ')': '-RRB-',
    '[': '-LSB-',
    ']': '-RSB-',
    '{': '-LCB-',
    '}': '-RCB-'
}

rev_replacements = {value: key for key,value in replacements.items()}


def replace(text, map):
    for key,value in map.items():
        text = text.replace(key, value)
    return text


def convert_genia(genia_path, conllu, sd):
    articles = sorted(os.path.join(genia_path, name)
                      for name in os.listdir(genia_path)
                      if name.endswith('.xml'))

    for article in articles:
        convert_article(article, conllu, sd)


def convert_article(article, conllu, sd):
    with open(article, mode='r', encoding='utf-8') as f:
        tree = bs(f, 'xml')

    pmid = tree.PMID.text
    print(f'# newdoc id = {pmid}', file=conllu)

    sents_nodes = tree.find_all('sentence')
    for sent_node in sents_nodes:
        convert_sent(pmid, sent_node, conllu, sd)


def convert_sent(pmid, sent_node, conllu, sd):
    space_after = []
    ptb_sent = convert_xml_to_ptb(sent_node.cons, space_after)
    # add space between sentences
    space_after[-1] = True
    try:
        ud_sent = sd.convert_tree(ptb_sent)
        tokens_conllu = []
        for i, (token, token_space) in enumerate(zip(ud_sent, space_after)):
            token_conllu = token.as_conll()
            if not token_space:
                token_conllu = token_conllu[:-1] + "SpaceAfter=No"

            tokens_conllu.append(token_conllu)

        print(replace('\n'.join(tokens_conllu), rev_replacements), file=conllu)
        print(file=conllu)
    except:
        print(f'PMID: {pmid}, sentence ID: {sent_node.attrs["id"]}\
              could not be converted and therefore is skipped.')


def convert_xml_to_ptb(cons, space_after):
    result = f'({cons.attrs["cat"]}'
    for node in cons:
        if node.name == 'cons' and 'null' not in node.attrs:
            result += f' {convert_xml_to_ptb(node, space_after)}'
        elif node.name == 'tok':
            # Some tokens have suspicious spaces in them which break CoreNLP tool
            result += f' ({node.attrs["cat"]} {replace(node.text, replacements).replace(" ", "")})'
            space_after.append(False)
        elif isinstance(node, NavigableString):
            space_after[-1] = True

    result += ')'

    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: path_to_genia path_to_conllu")
        return

    genia_path = sys.argv[1]
    conllu_path = sys.argv[2]

    sd = StanfordDependencies.get_instance()

    with open(conllu_path, mode='w', encoding='utf-8') as conllu:
        convert_genia(genia_path, conllu, sd)


if __name__ == '__main__':
    main()
