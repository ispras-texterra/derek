import re
from os import listdir, walk
from os.path import join, isfile, basename, splitext
from itertools import chain
from typing import List, Tuple, Set, Dict
from collections import defaultdict
from warnings import warn

from derek.data.model import Paragraph, Relation, Document, Entity, Sentence
from derek.data.helper import adjust_sentences, align_raw_entities, collapse_intersecting_entities
from derek.common.io import save_with_pickle, load_with_pickle
from derek.ner.feature_extraction.labelling_strategies import BIO2LabellingStrategy


class DocumentPathReader:
    def __init__(self, path, load_function):
        self.path = path
        self.load_function = load_function

    def __iter__(self):
        file_names = sorted(name for name in listdir(self.path) if isfile(join(self.path, name)))

        return chain.from_iterable(map(lambda name: self.load_function(self.path, name), file_names))


class RawTextReader:
    def __init__(self, segmenter):
        self.segmenter = segmenter

        self.dump = dump
        self.load = load

    def read(self, path: str, path_walker=None) -> List[Document]:
        document_names = _read_from_directory(path) if path_walker is None else path_walker(path)
        documents = []
        for name in document_names:
            doc = self.read_file(path, name)
            documents.append(doc)
        return documents

    def read_files(self, path: str, doc_name: str) -> List[Document]:
        with open(join(path, doc_name), "r", encoding='utf-8') as f:
            text = f.read()

        docs = []

        for i, doc_text in enumerate(text.split("\n\n")):
            docs.append(self._get_doc_from_raw_text(doc_text, doc_name + '_' + str(i)))

        return docs

    def read_file(self, path: str, doc_name: str) -> Document:
        with open(join(path, doc_name), "r", encoding='utf-8') as f:
            text = f.read()

        return self._get_doc_from_raw_text(text, doc_name)

    def _get_doc_from_raw_text(self, raw_text, doc_name) -> Document:
        tokens, sentences, raw_tokens = self.segmenter.segment(raw_text)

        # here we assume all text to be one paragraph
        paragraphs = [Paragraph(0, len(sentences))]

        return Document(splitext(doc_name)[0], tokens, sentences, paragraphs,  token_features={"char_spans": raw_tokens})

    @staticmethod
    def dump_docs(path: str, docs: List[Document], name: str):
        save_with_pickle(docs, path, name)

    @staticmethod
    def load_file(path: str, name: str) -> List[Document]:
        return load_with_pickle(path, name)

    @staticmethod
    def load_files(data_path: str) -> List[Document]:
        file_names = sorted(name for name in listdir(data_path) if isfile(join(data_path, name)))
        docs = []

        for file_name in file_names:
            docs += RawTextReader.load_file(data_path, file_name)

        return docs


class ChemProtDataReader:
    def __init__(self, segmenter, *, read_N=False, use_cpr=True, ignore_gene_type=True):
        self.segmenter = segmenter
        self.read_N = read_N
        self.use_cpr = use_cpr
        self.ignore_gene_type = ignore_gene_type

        self.dump = dump
        self.load = load

    def read(self, path: str) -> List[Document]:
        file_name_base = basename(path)
        abstracts = self._read_abstracts(join(path, f"{file_name_base}_abstracts.tsv"))
        entities = self._read_entities(join(path, f"{file_name_base}_entities.tsv"))
        relations = self._read_relations(join(path, f"{file_name_base}_relations.tsv"))

        docs = []
        for pmid in abstracts:
            doc = self._build_document(pmid, abstracts[pmid], entities[pmid], relations[pmid])
            docs.append(doc)
        return docs

    def _build_document(self, pmid, abstract, raw_entities, raw_relations) -> Document:
        tokens, sentences, raw_tokens = self.segmenter.segment(abstract['text'])
        raw_paragraphs = abstract['paragraphs']
        sentences, paragraphs, entities, relations = _merge(
            raw_tokens, sentences, raw_paragraphs, raw_entities, raw_relations)

        return Document(pmid, tokens, sentences, paragraphs, entities, relations)

    def _read_abstracts(self, path):
        abstracts = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                pmid, title, text = line.strip().split('\t')
                joined = "\t".join((title, text))
                abstracts[pmid] = {
                    'text': joined,
                    'paragraphs': ((0, len(title)), (len(title) + 1, len(joined)))
                }
        return abstracts

    def _read_entities(self, path):
        entities = defaultdict(list)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                pmid, id_, type_, start, end, _ = line.strip().split('\t')
                if self.ignore_gene_type and type_ in {'GENE-N', 'GENE-Y'}:
                    type_ = 'GENE'
                entities[pmid].append({'id': id_, 'type': type_, 'start': int(start), 'end': int(end)})
        return entities

    def _read_relations(self, path):
        relations = defaultdict(list)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                pmid, group, ev_type, rel_type, first, second = line.strip().split('\t')

                if 'N' in ev_type and not self.read_N:
                    continue

                relations[pmid].append({
                    'type': rel_type if self.use_cpr else group,
                    'first': first.split(':')[1],
                    'second': second.split(':')[1]
                })
        return relations


def _read_brat_annotations(format_str: str):
    """
        Reader for BRAT annotations file, currently support only binary relations and events
    """
    entities, relations = [], []
    equiv_entities_groups = []

    for line in format_str.splitlines():
        if not line:
            continue
        id_, information = line.split("\t")[:2]
        id_type = id_[0]

        if id_type == "T":
            # entity
            space_index = information.index(" ")
            ent_type = information[:space_index]
            spans_info = information[space_index+1:]
            ent_spans = tuple()
            for span_line in spans_info.split(";"):
                ent_spans += (tuple(int(a) for a in span_line.split()),)

            entities.append({"id": id_, "type": ent_type, "spans": ent_spans})
        elif id_type in ("R", "E"):
            # relation or event
            rel_type, _, arg1, _, arg2 = re.split(r"[ :]", information)
            relations.append({"id": id_, "type": rel_type, "first": arg1, "second": arg2})
        elif id_type == "*":
            equiv_entities_groups.append(information.split()[1:])

    if not equiv_entities_groups:
        return entities, relations

    expanded_relations = list(relations)
    for rel in relations:
        for group in equiv_entities_groups:
            for argument in ('first', 'second'):
                if rel[argument] in group:
                    for e in group:
                        if rel[argument] == e:
                            continue
                        expanded_relations.append({**rel, argument: e})

    return entities, expanded_relations


def _expand_spans(raw_entities_with_spans: list):
    raw_entities = []

    for ent in raw_entities_with_spans:
        spans = ent["spans"]
        start, end = min(s[0] for s in spans), max(s[1] for s in spans)
        raw_entities.append({"id": ent["id"], "type": ent["type"], "start": start, "end": end})

    return raw_entities


class BRATReader:
    def __init__(self, segmenter, collapse_intersecting=False):
        self.segmenter = segmenter
        self.collapse_intersecting = collapse_intersecting

        self.dump = dump
        self.load = load

    def read(self, path: str) -> List[Document]:
        all_files = listdir(path)
        file_names = sorted(set(splitext(f)[0] for f in all_files))
        docs = []

        for f in file_names:
            with open(join(path, f"{f}.txt"), "r", encoding="utf-8") as g:
                raw_text = g.read()

            if f"{f}.ann" not in all_files:
                warn(f"Skipping {f}.txt, no {f}.ann file found")
                continue

            with open(join(path, f"{f}.ann"), "r", encoding="utf-8") as g:
                annotations = g.read()

            tokens, sentences, raw_tokens = self.segmenter.segment(raw_text)
            raw_entities, raw_relations = _read_brat_annotations(annotations)
            raw_entities = _expand_spans(raw_entities)
            sentences, _, entities, relations = _merge(raw_tokens, sentences, [], raw_entities, raw_relations)

            if self.collapse_intersecting:
                entities, relations = collapse_intersecting_entities(entities, relations)

            # here we assume all doc to be in one paragraph
            doc = Document(f, tokens, sentences, [Paragraph(0, len(sentences))], entities, relations)
            docs.append(doc)

        return docs


class BioNLPDataReader:

    SYMMETRIC_RELATION_TYPES = {"Is_Linked_To", "Has_Sequence_Identical_To", "Is_Functionally_Equivalent_To"}

    def __init__(self, segmenter):
        self.segmenter = segmenter

        self.dump = dump
        self.load = load

    def read(self, path: str) -> List[Document]:
        files = sorted(listdir(path))
        docs = []
        for file in files:
            split = file.split(".")
            if split[-1] != "txt":
                continue
            doc = self._read_document(path, split[0])
            docs.append(doc)

        return docs

    def _read_document(self, path: str, name: str):
        with open(join(path, name + ".txt"), encoding="utf-8") as f:
            raw_text = f.read()

        tokens, sentences, raw_tokens = self.segmenter.segment(raw_text)
        raw_entities, raw_paragraphs, raw_relations = BioNLPDataReader._read_annotations(path, name)
        sentences, paragraphs, entities, relations = _merge(
            raw_tokens, sentences, raw_paragraphs, raw_entities, raw_relations,
            symmetric_types=BioNLPDataReader.SYMMETRIC_RELATION_TYPES)

        return Document(name, tokens, sentences, paragraphs, entities, relations)

    @staticmethod
    def _read_annotations(path: str, name: str):
        with open(join(path, name + ".a1"), encoding="utf-8") as f:
            annotations = f.read()

        raw_entities, _ = _read_brat_annotations(annotations)
        raw_entities = _expand_spans(raw_entities)

        filt_raw_entities = []
        raw_paragraphs = []
        # in BB3 corpus could be paragraph entities
        for ent in raw_entities:
            if ent["type"] in ("Paragraph", "Title"):
                raw_paragraphs.append((ent['start'], ent['end']))
            else:
                filt_raw_entities.append(ent)

        if isfile(join(path, name + ".a2")):
            with open(join(path, name + ".a2"), encoding="utf-8") as f:
                annotations = f.read()
            _, raw_relations = _read_brat_annotations(annotations)
        else:
            raw_relations = []

        return filt_raw_entities, raw_paragraphs, raw_relations


def _merge(raw_tokens: list, sentences: list, raw_paragraphs: list, raw_entities: list, raw_relations: list, *,
           symmetric_types: set = None) -> Tuple[List[Sentence], List[Paragraph], List[Entity], Set[Relation]]:
    """
    :param raw_tokens: list of tuples: (start, end, text)
    :param sentences: list of Sentence objects
    :param raw_paragraphs: list of tuples: (start, end)
    :param raw_entities: list of dicts: {'id', 'type', 'start', 'end'}
    :param raw_relations: list of dicts: {'type', 'first', 'second'}
    """
    paragraphs = []

    cur_par_idx = 0
    par_start = 0

    entities = sorted(align_raw_entities(raw_entities, raw_tokens))
    entities_dict = {ent.id: ent for ent in entities}
    sentences = adjust_sentences(sentences, entities)

    for i, sentence in enumerate(sentences):
        for token in raw_tokens[sentence.start_token: sentence.end_token]:
            if par_start != i + 1 and (_end_of_text(sentences, raw_tokens, sentence, token, i)
                                       or _end_of_paragraph(raw_paragraphs, cur_par_idx, token)):
                paragraphs.append(Paragraph(par_start, i + 1))
                par_start = i + 1
                cur_par_idx += 1

    return sentences, paragraphs, entities, _get_relations(raw_relations, entities_dict, symmetric_types)


def _get_relations(raw_relations: list, entities_dict: dict, symmetric_types: set):
    relations = set()
    for rel in raw_relations:
        e1 = entities_dict[rel['first']]
        e2 = entities_dict[rel['second']]
        rel_type = rel['type']

        relations.add(Relation(e1, e2, rel_type))

        if symmetric_types and rel_type in symmetric_types:
            relations.add(Relation(e2, e1, rel_type))

    return relations


def _end_of_text(sentences, raw_tokens, sentence, token, i):
    return i == len(sentences) - 1 and token == raw_tokens[sentence.end_token - 1]


def _end_of_paragraph(raw_paragraphs, cur_par_idx, token):
    return raw_paragraphs and raw_paragraphs[cur_par_idx][1] <= token[1]


def _read_from_directory(path):
    document_names = []
    for root, dirs, files in walk(path):
        for filename in files:
            document_names.append(filename)
    return  document_names


def dump(path: str, docs: list):
    save_with_pickle(docs, path, "docs.pkl")


def load(path: str) -> List[Document]:
    return load_with_pickle(path, "docs.pkl")


class FactRuEvalReader:
    def __init__(self, collapse_intersecting=True, locorg_allowed=True, blacklist=set()):
        self.collapse_intersecting = collapse_intersecting
        self.blacklist = set(blacklist).union({"Project", "Facility"})
        if locorg_allowed:
            self.convert_locorg = lambda x: x
        else:
            self.convert_locorg = lambda x: "Location" if x == "LocOrg" else x

        self.dump = dump
        self.load = load

    def read(self, path: str) -> List[Document]:
        document_names = self.get_document_names(path)
        documents = []
        for name in document_names:
            doc = self._read_document(path, name)
            documents.append(doc)
        return documents

    @staticmethod
    def get_document_names(path: str):
        document_names = []
        for root, dirs, files in walk(path):
            for filename in files:
                if filename.endswith('.tokens'):
                    document_names.append(splitext(filename)[0])
        return document_names

    def _read_document(self, directory, name):
        path = join(directory, name)
        tokens, fre_id2token_id, sentences, char_spans = self._get_tokens_and_sentences(path)
        paragraphs = [Paragraph(0, len(sentences))]
        entities = self._get_entities(path, fre_id2token_id)
        return Document(name, tokens, sentences, paragraphs, entities, token_features={"char_spans": char_spans})

    @staticmethod
    def _read_file(name: str):
        with open(name, "r", encoding="utf-8") as f:
            for line in f:
                yield line.split("#")[0].split()

    def _get_tokens_and_sentences(self, filename: str) -> Tuple[List[str], Dict[str, int], List[Sentence], List[list]]:
        tokens = []
        fre_id2token_id = {}
        sentences = []
        char_spans = []
        token_counter = 0
        start_sentence = 0

        for line in self._read_file(filename + ".tokens"):
            if len(line) > 1:
                tokens.append(line[-1])
                fre_id2token_id[line[0]] = token_counter
                start = int(line[1])
                end = start + int(line[2])
                char_spans.append((start, end))
                token_counter += 1
            else:
                assert start_sentence != token_counter
                sentences.append(Sentence(start_sentence, token_counter))
                start_sentence = token_counter

        assert start_sentence == token_counter, f"No extra line in the end of {filename}, last sentence is missing"

        return tokens, fre_id2token_id, sentences, char_spans

    def _get_entities(self, name: str, fre_id2token_id: dict):
        span_dict = self._get_spans(name, fre_id2token_id)
        objects = self._get_objects(name)
        entities = self._create_entities_from(span_dict, objects)
        if self.collapse_intersecting:
            entities, _ = collapse_intersecting_entities(sorted(entities, key=lambda x: x.start_token), set())
        return entities

    def _get_spans(self, filename: str, fre_id2token_id: dict):
        span_dict = {}
        for line in self._read_file(filename + ".spans"):
            span_id = line[0]
            token_start = fre_id2token_id[line[4]]
            token_end = token_start + int(line[5])
            span_dict[span_id] = {"start": token_start, "end": token_end}
        return span_dict

    def _get_objects(self, filename: str):
        objects = []
        for line in self._read_file(filename + ".objects"):
            if line[1] not in self.blacklist:
                objects.append({"id": line[0], "type": line[1], "spans": line[2:]})
        return objects

    def _create_entities_from(self, span2position: dict, fre_objects: list) -> List[Entity]:
        entities = []
        for fre_object in fre_objects:
            tokens = sorted([span2position[span] for span in fre_object["spans"]], key=lambda x: x['start'])
            start = tokens[0]["start"]
            end = tokens[-1]["end"]
            ent_type = self.convert_locorg(fre_object["type"])
            entities.append(Entity(fre_object["id"], start, end, ent_type))
        return entities


class CoNLLReader:
    def __init__(self):
        self.dump = dump
        self.load = load
        self._decode_strategy = BIO2LabellingStrategy()

    def read(self, path: str) -> List[Document]:
        docs = []

        with open(path, "r", encoding="utf-8") as f:
            doc_raw_tokens = []

            for line in f:
                if line.startswith("-DOCSTART-"):
                    if doc_raw_tokens:
                        docs.append(self._create_doc(doc_raw_tokens, len(docs)))
                        doc_raw_tokens = []
                    continue
                doc_raw_tokens.append(line.split())

            if doc_raw_tokens:
                docs.append(self._create_doc(doc_raw_tokens, len(docs)))

        return docs

    def _create_doc(self, doc_raw_tokens: List[List[str]], doc_idx) -> Document:
        tokens, sentences, entities, pos_tags = [], [], [], []

        sent_tokens, sent_pos_tags, sent_entities_labels = [], [], []
        sent_start = 0
        for raw_token in doc_raw_tokens:
            if not raw_token:
                if sent_tokens:
                    tokens.extend(sent_tokens)
                    pos_tags.extend(sent_pos_tags)
                    sentences.append(Sentence(sent_start, sent_start + len(sent_tokens)))
                    sent_start += len(sent_tokens)
                    entities.extend(self._decode_strategy.decode_labels(sentences[-1], sent_entities_labels))
                    sent_tokens, sent_pos_tags, sent_entities_labels = [], [], []
                continue

            token, pos_tag, _, ent_label = raw_token
            sent_tokens.append(token)
            sent_entities_labels.append(ent_label)
            sent_pos_tags.append(pos_tag)

        if sent_tokens:
            tokens.extend(sent_tokens)
            pos_tags.extend(sent_pos_tags)
            sentences.append(Sentence(sent_start, sent_start + len(sent_tokens)))
            entities.extend(self._decode_strategy.decode_labels(sentences[-1], sent_entities_labels))

        return Document(
            str(doc_idx), tokens, sentences, [Paragraph(0, len(sentences))], entities, token_features={"pos": pos_tags})
