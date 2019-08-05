from genericpath import isfile
from os.path import join
from typing import List, IO, Dict

from derek.coref.data.model import CoreferenceChain
from derek.data.helper import adjust_sentences, find_span_head_token
from derek.data.model import Document, Sentence, Paragraph, Entity, Relation
from derek.data.readers import dump, load


class RuCorDataReader:

    def __init__(self, fix_entity_types=True):
        self.fix_entity_types = fix_entity_types

        self.dump = dump
        self.load = load

    def read(self, path: str) -> List[Document]:
        """
        :param path: path to dir with Tokens.txt and Groups.txt
        :return: list of documents and list of relations
        """
        tokens_path = join(path, 'Tokens.txt')
        groups_path = join(path, 'Groups.txt')
        speech_path = join(path, 'TokensSpeech.txt') if isfile(join(path, 'TokensSpeech.txt')) else None
        with open(tokens_path, encoding="utf-8") as token_file, open(groups_path, encoding="utf-8") as group_file:
            raw_docs = self._read_raw_docs(token_file)
            groups = self._read_groups(group_file)

        if speech_path is not None:
            with open(speech_path, encoding="utf-8") as speech_file:
                raw_docs = self._add_speech(speech_file, raw_docs)

        docs_dict = self._get_docs(raw_docs, groups)
        # (doc_id, list of chains)
        chains_dict = {doc_id: self._get_chains(doc, groups) for doc_id, doc in docs_dict.items()}
        # (doc_id, list of relation set)
        rels_dict = {doc_id: list(map(lambda x: x.to_relations_set(), chains.values()))
                     for doc_id, chains in chains_dict.items()}
        # (doc_id, set of relations)
        rels_dict = {doc_id: set(sum(relations, [])) for doc_id, relations in rels_dict.items()}
        docs = []
        for doc_id in sorted(rels_dict):
            docs.append(docs_dict[doc_id].with_relations(rels_dict[doc_id]))
        if self.fix_entity_types:
            docs = self._fix_entity_types(docs)
        return docs

    @staticmethod
    def _read_raw_docs(token_file: IO) -> Dict[str, List[dict]]:
        raw_docs = {}
        for line in token_file.readlines()[1:]:
            doc_id, shift, _, token, lemma, gram = line.split()
            raw_docs.setdefault(doc_id, []).append({
                'shift': int(shift),
                'token': token,
                'lemma': lemma,
                'gram': gram
            })
        return raw_docs

    @staticmethod
    def _read_groups(group_file: IO) -> Dict[str, list]:
        groups = {}
        for line in group_file.readlines()[1:]:
            doc_id, _, group_id, chain_id, link, _, _, _, tk_shifts, attributes = line.split('\t')[:10]
            if not attributes:
                # Fixing dataset bugs
                attributes = 'str:noun'

            groups.setdefault(doc_id, []).append({
                'group_id': group_id,
                'chain_id': chain_id,
                'link': link,
                'tk_shifts': list(map(lambda x: int(x), tk_shifts.split(','))),
                'attributes': {attr.split(':')[0]: attr.split(':')[1] for attr in attributes.split('|')},
            })

        return groups

    def _get_docs(self, raw_docs: Dict[str, List[dict]], groups: Dict[str, list]) -> Dict[str, Document]:
        docs = {}
        for doc_id, raw_tokens in raw_docs.items():
            tokens = []
            token_features = {}
            sentences = []
            sent_start = 0
            shift2idx = {}

            for i, raw_token in enumerate(raw_tokens):
                tokens.append(raw_token['token'])
                token_features.setdefault('lemma', []).append(raw_token['lemma'])
                token_features.setdefault('gram', []).append(raw_token['gram'])
                if "speech" in raw_token:
                    token_features.setdefault("speech", []).append(raw_token['speech'])
                    token_features.setdefault("said", []).append(raw_token['said'])
                    token_features.setdefault("author_comment", []).append(raw_token['author_comment'])
                    token_features.setdefault("speech_verb", []).append(raw_token['speech_verb'])
                shift2idx[raw_token['shift']] = i

                if raw_token['gram'] == 'SENT':
                    sentences.append(Sentence(sent_start, i + 1))
                    sent_start = i + 1
            if sentences[-1].end_token != len(tokens):
                sentences.append(Sentence(sent_start, len(tokens)))
            entities = self._get_entities(groups, shift2idx, doc_id)
            sentences = adjust_sentences(sentences, entities)

            doc = Document(doc_id, tokens, sentences, [Paragraph(0, len(sentences))], entities,
                           token_features=token_features)
            docs[doc_id] = doc
            
        return docs

    @staticmethod
    def _get_entities(groups: Dict[str, list], shift2idx: Dict[str, int], doc_id: str) -> List[Entity]:
        entities = []
        if doc_id not in groups:
            return entities
        for mention in groups[doc_id]:
            name = mention['group_id']
            start = shift2idx[mention['tk_shifts'][0]]
            end = shift2idx[mention['tk_shifts'][-1]] + 1
            e_type = mention['attributes']['str']
            if not e_type:
                # Fixing dataset bugs
                e_type = 'noun'
            if e_type in {'refl', 'poss',  'rel', 'dem', 'pron'}:
                e_type = 'pron'
            if e_type in {'noun', 'appo'}:
                e_type = 'noun'

            entities.append(Entity(name, start, end, e_type))
        return entities

    @staticmethod
    def _get_chains(doc: Document, groups: Dict[str, list]) -> Dict[str, CoreferenceChain]:
        chains = {}

        if doc.name not in groups:
            return chains

        entities = doc.entities
        entities_dict = {entity.id: entity for entity in entities}

        for mention in groups[doc.name]:
            chains.setdefault(mention['chain_id'], []).append(entities_dict[mention['group_id']])
        chains = {key: CoreferenceChain(val) for key, val in chains.items()}
        return chains

    def _add_speech(self, speech_file: IO, raw_docs: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
        new_docs = {}

        current_doc_id = None
        token_idx = 0
        for line in speech_file.readlines()[1:]:
            doc_id, shift, _, _, speech, said, author_comment, speech_verb = line.split()
            if doc_id != current_doc_id:
                token_idx = 0
                current_doc_id = doc_id
            for raw_token in raw_docs[current_doc_id][token_idx:]:
                if raw_token['shift'] == int(shift):
                    new_docs.setdefault(current_doc_id, []).append({**raw_token,
                                                                    "speech": speech,
                                                                    "said": said,
                                                                    "author_comment": author_comment,
                                                                    "speech_verb": speech_verb
                                                                    })
                    token_idx += 1
                    break
        return new_docs

    @staticmethod
    def _fix_entity_types(docs):
        ret = []
        for doc in docs:
            new_entities = []
            entity_mapping = {}
            new_rels = []
            for entity in doc.entities:
                head = find_span_head_token(doc, entity)
                if doc.token_features['pos'][head] == 'PRON':
                    e_type = 'pron'
                else:
                    e_type = 'noun'
                new_entity = entity.with_type(e_type)
                entity_mapping[entity] = new_entity
                new_entities.append(new_entity)
            for rel in doc.relations:
                new_rels.append(Relation(entity_mapping[rel.first_entity], entity_mapping[rel.second_entity], rel.type))

            ret.append(doc.without_relations().without_entities().with_entities(new_entities).with_relations(new_rels))
        return ret
