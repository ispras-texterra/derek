from typing import Iterable, Generator, Tuple, Any
from derek.data.model import Document, Sentence


class DefaultSamplingStrategy:
    @staticmethod
    def generate(doc: Document, sent: Sentence) -> Generator[Tuple[int, int, Any], None, None]:
        for t1_idx in range(len(sent)):
            t1_parent_idx = t1_idx + doc.token_features["dt_head_distances"][sent.start_token + t1_idx]
            t1_arc_label = doc.token_features["dt_labels"][sent.start_token + t1_idx]

            for t2_idx in range(t1_idx + 1, len(sent)):
                t2_parent_idx = t2_idx + doc.token_features["dt_head_distances"][sent.start_token + t2_idx]
                t2_arc_label = doc.token_features["dt_labels"][sent.start_token + t2_idx]

                if t1_parent_idx == t2_idx:
                    arc_type = ("right", t1_arc_label)
                elif t1_idx == t2_parent_idx:
                    arc_type = ("left", t2_arc_label)
                else:
                    arc_type = None

                yield t1_idx, t2_idx, arc_type

    @staticmethod
    def get_possible_arc_types(docs: Iterable[Document]) -> set:
        label_types = set()

        for doc in docs:
            for dt_dist, dt_label in zip(doc.token_features["dt_head_distances"], doc.token_features["dt_labels"]):
                if dt_dist != 0:
                    label_types.add(dt_label)

        arc_types = {None}
        for label in label_types:
            arc_types.add(("left", label))
            arc_types.add(("right", label))

        return arc_types


class PosFilteringSamplingStrategy:
    def __init__(self, filter_pos: set):
        self.filter_pos = filter_pos

    def generate(self, doc: Document, sent: Sentence) -> Generator[Tuple[int, int, Any], None, None]:
        head_distances = doc.token_features["dt_head_distances"][sent.start_token: sent.end_token]
        pos = doc.token_features["pos"][sent.start_token: sent.end_token]
        for token_idx in range(len(sent)):
            token_children = [idx for idx in range(len(sent))
                              if idx != token_idx and idx + head_distances[idx] == token_idx]

            for child in token_children:
                child_pos = pos[child]

                if child_pos in self.filter_pos:
                    continue

                negative = None
                negative_dist = None

                for idx in range(len(sent)):
                    if idx != token_idx and idx != child and pos[idx] == child_pos:
                        dist = abs(child - idx)
                        if negative is None or negative_dist > dist:
                            negative = idx
                            negative_dist = dist

                if negative is not None:
                    yield (token_idx, child, True)
                    yield (token_idx, negative, False)

    @staticmethod
    def get_possible_arc_types(docs: Iterable[Document]) -> set:
        return {True, False}
