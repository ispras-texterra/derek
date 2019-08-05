from typing import List, Tuple

from derek.coref.data.chains_collection import collect_chains, chains2rels, get_closest_entity
from derek.coref.data.model import CoreferenceChain
from derek.data.helper import get_entity_distance_between_entities
from derek.data.model import Document, Entity
from derek.rel_ext.feature_extraction.sampling_strategies import DefaultCandidateMaker


class OneGroupSamplingStrategy:
    def __init__(self, pair_extraction_strategy, candidate_maker, pair_filter=None):
        self.pair_extraction_strategy = pair_extraction_strategy
        self.candidate_maker = candidate_maker
        self.pair_filter = pair_filter

    def apply(self, doc: Document, use_filter=False, *, include_labels=False)\
            -> List[Tuple[int, int, List[Tuple[Entity, Entity, str]]]]:

        pairs = self.pair_extraction_strategy.apply(doc, include_labels=include_labels)
        if self.pair_filter is not None and use_filter:
            pairs = self.pair_filter.apply(pairs, doc)
        candidates = self.candidate_maker.apply(pairs, doc.relations if include_labels else None)
        return candidates


class CorefPairExtractionStrategy:
    def __init__(self, max_candidate_distance, preprocessing_strategy=None):
        self.max_candidate_distance = max_candidate_distance
        self.preprocessing_strategy = preprocessing_strategy

    def apply(self, doc: Document, *, include_labels=False) -> List[Tuple[Entity, Entity]]:
        if self.preprocessing_strategy is not None:
            doc = self.preprocessing_strategy.apply(doc)

        pairs = []
        if not include_labels:
            for i, entity in enumerate(doc.entities):
                start_idx = max([0, i - self.max_candidate_distance])
                for preceding_entity in doc.entities[start_idx: i]:
                    pairs.append((preceding_entity, entity))
            return pairs

        for rel in _convert_to_rel_chains(doc.relations, doc.entities):
            e1, e2 = rel.first_entity, rel.second_entity

            pairs.append((e1, e2))

            for entity in doc.entities:
                if entity != e1 and entity != e2 and \
                        e1.start_token <= entity.start_token and entity.end_token <= e2.end_token:
                    pairs.append((entity, e2))
        return pairs


def _convert_to_rel_chains(rels, entities):
    chains = collect_chains(rels, entities)
    return chains2rels(chains)


class AbstractPronPairExtractionStrategy:
    def __init__(self, max_candidate_distance):
        self.max_candidate_distance = max_candidate_distance

    def apply(self, doc, *, include_labels=False):
        pairs = []
        noun_chains = get_noun_chains(doc)

        for entity in filter(lambda x: x.type != 'noun', doc.entities):
            for chain in noun_chains:
                pairs += self._get_pairs(doc, entity, chain)
        return pairs

    def _get_pairs(self, doc, entity, chain):
        raise Exception("Abstract class")


class PronPairExtractionStrategy(AbstractPronPairExtractionStrategy):
    def __init__(self, max_candidate_distance):
        super().__init__(max_candidate_distance)

    def _get_pairs(self, doc, entity, chain):
        pairs = []
        for e in chain.entities:
            if e.start_token <= entity.start_token and \
                    get_entity_distance_between_entities(doc, e, entity) <= self.max_candidate_distance:
                pairs.append(tuple(sorted([e, entity])))
        return pairs


class ClusterPronPairExtractionStrategy(AbstractPronPairExtractionStrategy):
    def __init__(self, max_candidate_distance, strict=False):
        super().__init__(max_candidate_distance)
        self.strict = strict

    def _get_pairs(self, doc, entity, chain):
        pairs = []
        closest_entity = get_closest_entity(chain, entity, left_only=True)
        if (closest_entity is None or
            get_entity_distance_between_entities(doc, closest_entity, entity) > self.max_candidate_distance) \
                and not self.strict:
            closest_entity = get_closest_entity(chain, entity, left_only=False)
        if closest_entity is not None \
                and get_entity_distance_between_entities(doc, closest_entity, entity) <= self.max_candidate_distance:
            pairs.append(tuple(sorted([closest_entity, entity])))
        return pairs


class CorefCandidateMaker:
    def __init__(self, label):
        self.label = label

    def apply(self, pairs, rels):
        if rels is None:
            return DefaultCandidateMaker().apply(pairs, rels)
        chains = collect_chains(rels)
        ret = []
        for e1, e2 in pairs:
            if any(map(lambda x: e1 in x.entities and e2 in x.entities, chains)):
                ret.append((e1, e2, self.label))
            else:
                ret.append((e1, e2, None))
        return ret


class NounPreprocessingStrategy:
    def apply(self, doc):
        noun_entities = list(filter(lambda x: x.type == 'noun', doc.entities))
        noun_chains = get_noun_chains(doc)

        doc = doc.without_relations().without_entities().with_entities(noun_entities)

        if noun_chains is not None:
            noun_rels = chains2rels(noun_chains)
            doc = doc.with_relations(noun_rels)

        return doc


def get_noun_chains(doc):
    try:
        rels = doc.relations
    except ValueError:
        return None

    chains = collect_chains(rels)
    chains = [get_noun_chain(chain) for chain in chains]
    return [chain for chain in chains if chain.entities]


def get_noun_chain(chain: CoreferenceChain):
    noun_entities = list(filter(lambda x: x.type == 'noun', chain.entities))
    return CoreferenceChain(noun_entities)
