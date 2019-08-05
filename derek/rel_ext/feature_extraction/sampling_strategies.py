from typing import List, Tuple
from derek.data.model import Document, Entity


class DefaultCandidateMaker:
    @staticmethod
    def apply(pairs, rels):
        rel_dict = {(rel.first_entity, rel.second_entity): rel.type for rel in rels} if rels is not None else {}
        return [(*pair, rel_dict.get(pair, None)) for pair in pairs]


class DefaultPairExtractionStrategy:
    def __init__(self, candidate_filter=None):
        self.candidate_filter = candidate_filter

    def apply(self, doc: Document, *, include_labels=False) -> List[Tuple[Entity, Entity]]:
        pairs = []
        for e1 in doc.entities:
            for e2 in doc.entities:
                if self.candidate_filter is not None and not self.candidate_filter.apply(doc, e1, e2):
                    continue
                pairs.append((e1, e2))

        return pairs


class DefaultCandidateExtractionStrategy:
    def __init__(self, pair_extractor, candidate_maker=DefaultCandidateMaker()):
        self.pair_extractor = pair_extractor
        self.candidate_maker = candidate_maker

    def apply(self, doc: Document, *, include_labels=False):
        pairs = self.pair_extractor.apply(doc, include_labels=include_labels)
        return self.candidate_maker.apply(pairs, doc.relations if include_labels else None)


class DifferentEntitiesCandidateFilter:
    @staticmethod
    def apply(doc, e1, e2):
        return e1 != e2


class InSameSentenceCandidateFilter:
    @staticmethod
    def apply(doc, e1, e2):
        return doc.get_entity_sent_idx(e1) == doc.get_entity_sent_idx(e2)


class MaxTokenDistanceCandidateFilter:
    def __init__(self, max_token_distance):
        self.max_token_distance = max_token_distance

    def apply(self, doc, e1, e2):
        return e1.token_distance_to(e2) <= self.max_token_distance


class RelArgTypesCandidateFilter:
    def __init__(self, rel_arg_types):
        self.rel_arg_types = rel_arg_types

    def apply(self, doc, e1, e2):
        return (e1.type, e2.type) in self.rel_arg_types


class IntersectingCandidateFilter:
    @staticmethod
    def apply(doc, e1, e2):
        return not e1.intersects(e2)


class AndFilter:
    def __init__(self, filters):
        self.filters = filters

    def apply(self, doc, e1, e2):
        return all(f.apply(doc, e1, e2) for f in self.filters)
