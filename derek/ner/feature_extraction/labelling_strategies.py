from typing import Set, List, Tuple
from abc import abstractmethod, ABCMeta

from derek.data.model import Sentence, Entity


class __AbstractLabellingStrategy(metaclass=ABCMeta):
    def __init__(self, ent_labels: Set[str], outside_ent_label: str = "O"):
        self.ent_labels = ent_labels
        self.outside_ent_label = outside_ent_label

    def get_possible_categories(self, ent_types: Set[str]) -> Set[str]:
        return {f"{label}-{e_type}" for label in self.ent_labels for e_type in ent_types}.union(
            (self.outside_ent_label,))

    @abstractmethod
    def encode_labels(self, sent: Sentence, sent_entities: List[Entity]) -> List[str]:
        raise NotImplementedError("Calling an abstract method of __AbstractLabellingStrategy")

    @abstractmethod
    def decode_labels(self, sent: Sentence, sent_labels: List[str]) -> List[Entity]:
        raise NotImplementedError("Calling an abstract method of __AbstractLabellingStrategy")


class __BIOFactoryLabellingStrategy(__AbstractLabellingStrategy, metaclass=ABCMeta):
    def __init__(self, start_symbols: Set[str] = None, end_symbols: Set[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = _BIODecoder(start_symbols, end_symbols)

    # TODO This class should be able to work on arbitrary span
    # TODO This class should be extracted from NER and decoupled from Entity
    def encode_labels(self, sent: Sentence, sent_entities: List[Entity]) -> List[str]:
        labels = [self.outside_ent_label] * len(sent)

        for e in sent_entities:
            assert labels[e.start_token - sent.start_token: e.end_token - sent.start_token] == ["O"] * len(e), \
                "Intersecting Entities!"

            labels[e.start_token - sent.start_token: e.end_token - sent.start_token] = \
                (f"{l}-{e.type}" for l in self._encode_entity(e, sent_entities))

        return labels

    def decode_labels(self, sent: Sentence, sent_labels: List[str]) -> List[Entity]:
        return [Entity("generated", start + sent.start_token, end + sent.start_token, t)
                for start, end, t in self.decoder.decode(sent_labels)]

    @abstractmethod
    def _encode_entity(self, ent: Entity, other_entities: List[Entity]) -> List[str]:
        pass


class _BIODecoder:
    def __init__(self, start_symbols: Set[str] = None, end_symbols: Set[str] = None):
        self.start_symbols = start_symbols if start_symbols is not None else set()
        self.end_symbols = end_symbols if end_symbols is not None else set()

    @staticmethod
    def _is_object(label):
        return label != 'O'

    @staticmethod
    def _get_object_type(label):
        return label.split('-')[-1]

    @staticmethod
    def _get_object_encoding(label):
        return label.split('-')[0]

    def decode(self, labels: List[str]) -> List[Tuple[int, int, str]]:
        object_start = None
        object_type = None
        result = []

        for i, label in enumerate(labels):
            if object_type is None:
                if self._is_object(label) and self._get_object_encoding(label) not in self.end_symbols:
                    # at object start
                    object_start = i
                    object_type = self._get_object_type(label)
                elif self._is_object(label) and self._get_object_encoding(label) in self.end_symbols:
                    result.append((i, i + 1, self._get_object_type(label)))
                else:
                    # waiting for object start
                    pass
            else:
                if not self._is_object(label):
                    # at object end
                    result.append((object_start, i, object_type))
                    object_start = None
                    object_type = None
                elif self._get_object_type(label) != object_type or \
                        self._get_object_encoding(label) in self.start_symbols:
                    # at both object end and start
                    result.append((object_start, i, object_type))
                    object_start = i
                    object_type = self._get_object_type(label)
                elif self._get_object_encoding(label) in self.end_symbols:
                    result.append((object_start, i + 1, object_type))
                    object_start = None
                    object_type = None
                else:
                    # waiting for object end
                    pass

        if object_type is not None:
            result.append((object_start, len(labels), object_type))

        return result


class IOLabellingStrategy(__BIOFactoryLabellingStrategy):
    def __init__(self):
        super().__init__(set(), set(), {"I"})

    def _encode_entity(self, ent: Entity, other_entities: List[Entity]):
        return ["I"] * len(ent)


class BIO2LabellingStrategy(__BIOFactoryLabellingStrategy):
    def __init__(self):
        super().__init__({"B"}, set(), {"B", "I"})

    def _encode_entity(self, ent: Entity, other_entities: List[Entity]):
        # TODO Could be optimized through Document.entities.at_token
        first_symbol = "B" if any(e.end_token == ent.start_token and e.type == ent.type for e in other_entities) \
            else "I"
        return [first_symbol] + ["I"] * (len(ent) - 1)


class BIOLabellingStrategy(__BIOFactoryLabellingStrategy):
    def __init__(self):
        super().__init__({"B"}, set(), {"B", "I"})

    def _encode_entity(self, ent: Entity, other_entities: List[Entity]):
        return ["B"] + ["I"] * (len(ent) - 1)


class BILOULabellingStrategy(__BIOFactoryLabellingStrategy):
    def __init__(self):
        super().__init__({"B", "U"}, {"U", "L"}, {"B", "I", "L", "U"})

    def _encode_entity(self, ent: Entity, other_entities: List[Entity]):
        if len(ent) == 1:
            return ["U"]

        return ["B"] + ["I"] * (len(ent) - 2) + ["L"]


_LABELLING_STRATEGIES = {
    "IO": IOLabellingStrategy,
    "BIO": BIOLabellingStrategy,
    "BIO2": BIO2LabellingStrategy,
    "BILOU": BILOULabellingStrategy
}


def get_labelling_strategy(name: str):
    strategy = _LABELLING_STRATEGIES.get(name, None)
    if strategy is None:
        raise KeyError(f"Uknown {name} NER strategy")

    return strategy()
