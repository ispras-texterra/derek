from collections import defaultdict, Sequence
from typing import Tuple, List, Dict, Any, Iterable
from warnings import warn


class Document:
    def __init__(self, name: str, tokens: List[str], sentences: Iterable['Sentence'], paragraphs: List['Paragraph'],
                 entities: Iterable['Entity'] = None, relations: Iterable['Relation'] = None,
                 token_features: Dict[str, List[Any]] = None, extras: Dict[str, Any] = None):
        """
        :param token_features: all dict values are lists aligned to tokens
        All lists are ordered by position in text from left to right
        """
        self.name = name
        self.tokens = tokens
        self.sentences = sentences if isinstance(sentences, SortedSpansSet) else SortedSpansSet(sentences)
        self.paragraphs = paragraphs

        self.__entities = None if entities is None else \
            (entities if isinstance(entities, SortedSpansSet) else SortedSpansSet(entities))
        self.__relations = frozenset(relations) if relations is not None else None
        if self.__entities is None and self.__relations is not None:
            raise ValueError("Could not create document with relations but without entities")

        self.token_features = token_features if token_features is not None else {}
        self.extras = extras if extras is not None else {}

    @property
    def entities(self):
        if self.__entities is None:
            raise ValueError("Given document doesn't provide entities")
        return self.__entities

    def without_entities(self):
        if self.__relations is not None:
            raise ValueError("Could not remove entities from document with relations")
        return Document(self.name, self.tokens, self.sentences, self.paragraphs,
                        token_features=self.token_features, extras=self.extras)

    def with_entities(self, entities):
        if self.__entities is not None:
            raise ValueError("Given document already provide entities")
        return Document(self.name, self.tokens, self.sentences, self.paragraphs, entities,
                        token_features=self.token_features, extras=self.extras)

    @property
    def relations(self):
        if self.__relations is None:
            raise ValueError("Given document doesn't provide relations")
        return self.__relations

    def without_relations(self):
        return Document(self.name, self.tokens, self.sentences, self.paragraphs, self.__entities,
                        token_features=self.token_features, extras=self.extras)

    def with_relations(self, relations):
        if self.__relations is not None:
            raise ValueError("Given document already provide relations")
        return Document(self.name, self.tokens, self.sentences, self.paragraphs, self.__entities, relations,
                        token_features=self.token_features, extras=self.extras)

    def with_additional_token_features(self, token_features):
        common_keys = self.token_features.keys() & token_features.keys()
        if common_keys:
            warn(f'Provided features has the following existing keys: {common_keys}')

        return Document(self.name, self.tokens, self.sentences, self.paragraphs, self.__entities, self.__relations,
                        {**self.token_features, **token_features}, extras=self.extras)

    def with_additional_extras(self, extras):
        common_keys = self.extras.keys() & extras.keys()
        if common_keys:
            warn(f'Provided extras has the following existing keys: {common_keys}')

        return Document(self.name, self.tokens, self.sentences, self.paragraphs, self.__entities, self.__relations,
                        self.token_features, {**self.extras, **extras})

    def __eq__(self, other):
        if isinstance(other, Document):
            return self.name == other.name and self.tokens == other.tokens and \
                   self.sentences == other.sentences and self.paragraphs == other.paragraphs and \
                   self.__entities == other.__entities and self.__relations == other.__relations and \
                   self.token_features == other.token_features and self.extras == other.extras
        return False

    def __repr__(self):
        return f"Tokens: {self.tokens}\nSentences:{self.sentences}\nParagraphs:{self.paragraphs}\n" + \
               f"Entities:{self.__entities}\nRelations:{self.__relations}\n" + \
               f"Token_features:{self.token_features}\nExtras:{self.extras}"

    def get_token_sent_idx(self, token_idx):
        return self.sentences.indexed_at_token(token_idx)[0][0]

    def get_entity_sent_idx(self, entity):
        ent_start_sent_idx = self.get_token_sent_idx(entity.start_token)
        ent_end_sent_idx = self.get_token_sent_idx(entity.end_token - 1)

        if ent_start_sent_idx != ent_end_sent_idx:
            warn("Entity is not contained in one sentence, return first sentence", stacklevel=2)
        return ent_start_sent_idx


class TokenSpan:
    """
        If inheritor class has additional fields it must implement as_tuple returning fields as constructor arguments,
        and own __lt__ method for expected comparison
    """
    def __init__(self, start_token_idx: int, end_token_idx: int):
        """
            TokenSpan on [start_token_idx, end_token_idx)
        """
        if start_token_idx < 0 or end_token_idx < 0:
            raise Exception("start_token_idx and end_token_idx must be >= 0")

        if start_token_idx >= end_token_idx:
            raise Exception("start_token_idx must be < end_token_idx")

        self.__start_idx = start_token_idx
        self.__end_idx = end_token_idx

    @property
    def start_token(self) -> int:
        return self.__start_idx

    @property
    def end_token(self) -> int:
        return self.__end_idx

    def as_tuple(self) -> Tuple[int, int]:
        return self.__start_idx, self.__end_idx

    def __repr__(self):
        return repr(self.as_tuple())

    def __hash__(self):
        return hash(self.as_tuple())

    def __eq__(self, other):
        return type(self) is type(other) and self.as_tuple() == other.as_tuple()

    def __lt__(self, other):
        if type(self) is not type(other):
            raise Exception(f"{type(self)} expected")
        return self.as_tuple() < other.as_tuple()

    def __len__(self):
        return self.__end_idx - self.__start_idx

    def contains(self, obj: 'TokenSpan') -> bool:
        if not isinstance(obj, TokenSpan):
            raise Exception("provided object is not TokenSpan instance")

        return self.start_token <= obj.start_token and self.end_token >= obj.end_token

    def intersects(self, obj: 'TokenSpan') -> bool:
        return not self.token_distance_to(obj)

    def token_distance_to(self, obj: 'TokenSpan') -> int:
        """
            Returns absolute token distance to provided TokenSpan
            If objects intersect -> 0 is returned, otherwise dist + 1 is returned
        """
        if not isinstance(obj, TokenSpan):
            raise Exception("provided object is not TokenSpan instance")

        return max(0, max(self.start_token, obj.start_token) - min(self.end_token, obj.end_token) + 1)

    def coincides(self, obj: 'TokenSpan') -> bool:
        if not isinstance(obj, TokenSpan):
            raise Exception("provided object is not TokenSpan instance")
        
        return self.start_token == obj.start_token and self.end_token == obj.end_token


class Sentence(TokenSpan):
    def __init__(self, start_token: int, end_token: int):
        super().__init__(start_token, end_token)

    def relocated(self, start_token: int, end_token: int) -> 'Sentence':
        return Sentence(start_token, end_token)


class Paragraph:
    def __init__(self, start_sentence: int, end_sentence: int):
        """
        :param start_sentence: index of starting sentence
        :param end_sentence: index of ending sentence
        """
        self.end_sentence = end_sentence
        self.start_sentence = start_sentence

    def __repr__(self):
        return "Start: " + str(self.start_sentence) + " End: " + str(self.end_sentence)

    def __eq__(self, other):
        if isinstance(other, Paragraph):
            return self.start_sentence == other.start_sentence and self.end_sentence == other.end_sentence

        return False

    def __hash__(self):
        return hash((self.start_sentence, self.end_sentence))


class Entity(TokenSpan):
    def __init__(self, id_: str, start_token: int, end_token: int, ent_type: str):
        """
        :param id_: entity id in corpus
        :param start_token: index of starting token
        :param end_token: index of ending token (not included in entity)
        :param ent_type: type of entity
        """
        super().__init__(start_token, end_token)
        self.__id = id_
        self.__type = ent_type

    @property
    def id(self) -> str:
        return self.__id

    @property
    def type(self) -> str:
        return self.__type

    def as_tuple(self) -> Tuple[str, int, int, str]:
        return self.id, self.start_token, self.end_token, self.type

    def __lt__(self, other):
        if type(other) is not Entity:
            raise Exception(f"{Entity} expected")

        return (self.start_token, self.end_token, self.id, self.type) < \
               (other.start_token, other.end_token, other.id, other.type)

    def relocated(self, start_token: int, end_token: int) -> 'Entity':
        return Entity(self.id, start_token, end_token, self.type)

    def with_type(self, new_type: str) -> 'Entity':
        return Entity(self.id, self.start_token, self.end_token, new_type)


class Relation:
    def __init__(self, first_entity: Entity, second_entity: Entity, rel_type):
        """
        :param first_entity: first Entity object in relation
        :param second_entity: second Entity object in relation
        :param rel_type: relation type
        """
        self.type = rel_type
        self.second_entity = second_entity
        self.first_entity = first_entity

    def __repr__(self):
        return "First: " + repr(self.first_entity) + " Second: " + repr(self.second_entity)

    def __eq__(self, other):
        if isinstance(other, Relation):
            return self.type == other.type and self.first_entity == other.first_entity \
                   and self.second_entity == other.second_entity
        return False

    def __hash__(self):
        return hash((self.type, self.second_entity, self.first_entity))

    @property
    def entities_types(self) -> Tuple[str, str]:
        return self.first_entity.type, self.second_entity.type


class SortedSpansSet(Sequence):
    def __init__(self, spans_iter: Iterable[TokenSpan]):
        spans_set = set(spans_iter)
        spans = sorted(spans_set)

        self.__spans = spans
        self.__token2spans = SortedSpansSet.__build_token2spans(spans)
        self.__spans_set = spans_set

    def __len__(self):
        return len(self.__spans)

    def __getitem__(self, span_idx):
        return self.__spans[span_idx]

    def indexed_at_token(self, token_idx):
        """
        :return: list of (span idx, span) pairs for spans which contain given token idx
        """
        return [(span_idx, self.__getitem__(span_idx)) for span_idx in self.__token2spans[token_idx]]

    def at_token(self, token_idx):
        """
        :return: list of spans which contain given token idx
        """
        return [span for (_, span) in self.indexed_at_token(token_idx)]

    def indexed_contained_in(self, container: TokenSpan):
        """
        :return: list of (span idx, span) pairs for spans which are contained in given span
        """
        ret = []
        seen_spans = set()
        for i in range(container.start_token, container.end_token):
            for span_idx in self.__token2spans[i]:
                if span_idx in seen_spans:
                    continue
                span = self.__getitem__(span_idx)
                if container.contains(span):
                    ret.append((span_idx, span))
                seen_spans.add(span_idx)
        return ret

    def contained_in(self, container: TokenSpan):
        """
        :return: list of spans which are contained in given span
        """
        return [span for (_, span) in self.indexed_contained_in(container)]

    def __repr__(self):
        return str(self.__spans)

    def __contains__(self, span):
        return span in self.__spans_set

    def __eq__(self, other):
        if isinstance(other, SortedSpansSet):
            return self.__spans_set == other.__spans_set
        return False

    @staticmethod
    def __build_token2spans(spans):
        ret = defaultdict(list)
        for span_idx, span in enumerate(spans):
            for token_idx in range(span.start_token, span.end_token):
                ret[token_idx].append(span_idx)
        return ret
