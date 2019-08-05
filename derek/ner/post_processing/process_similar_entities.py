from collections import Counter
from typing import Callable, List

from derek.coref.data.chains_collection import collect_chains
from derek.coref.data.model import CoreferenceChain
from derek.data.model import Entity, Relation, Document
from nltk.metrics.distance import edit_distance


def levenshtein_lemma_compare_tokens(doc: Document, idx1: int, idx2: int, decision_number=1):
    return edit_distance(doc.token_features["lemmas"][idx1], doc.token_features["lemmas"][idx2]) <= decision_number


def strict_compare_tokens(doc: Document, idx1: int, idx2: int):
    return doc.tokens[idx1] == doc.tokens[idx2]


class __CombineComparators:
    def __init__(self, comparators):
        self.__comparators = comparators

    def __call__(self, doc: Document, idx1: int, idx2: int):
        return any(comp(doc, idx1, idx2) for comp in self.__comparators)


def compare_entities_by_tokens(
        doc: Document, ent1: Entity, ent2: Entity,
        tokens_comparator: Callable[[Document, int, int], bool] =
        __CombineComparators([strict_compare_tokens, levenshtein_lemma_compare_tokens])) -> bool:
    """
        :return True if smaller entity consist of tokens matching with different tokens of larger entity
    """

    if len(ent2) > len(ent1):
        ent1, ent2 = ent2, ent1

    matched_tokens = set()

    for idx2 in range(ent2.start_token, ent2.end_token):
        for idx1 in range(ent1.start_token, ent1.end_token):
            if idx1 in matched_tokens:
                continue

            if tokens_comparator(doc, idx1, idx2):
                matched_tokens.add(idx1)
                break
        else:
            # didn't matched idx2 token
            break
    else:
        # matched all ent2 tokens
        return True

    return False


def chain_similar_entities(
        doc: Document, entities: List[Entity],
        entity_comparator: Callable[[Document, Entity, Entity], bool] = compare_entities_by_tokens) \
            -> List[CoreferenceChain]:

    relations = set()

    for i, e1 in enumerate(entities):
        for e2 in entities[:i]:
            if entity_comparator(doc, e1, e2):
                relations.add(Relation(e1, e2, "match"))

    return collect_chains(relations, entities)


def unify_types_of_similar_entities(
        doc: Document, entities: List[Entity],
        entity_comparator: Callable[[Document, Entity, Entity], bool] = compare_entities_by_tokens) -> List[Entity]:

    chains = chain_similar_entities(doc, entities, entity_comparator)
    ret = []

    for chain in chains:
        (most_common_type, count), = Counter(e.type for e in chain.entities).most_common(1)

        if count > len(chain.entities) // 2:
            proposed_type = most_common_type
        else:
            proposed_type = chain.entities[0].type

        ret.extend(e.with_type(proposed_type) for e in chain.entities)

    return sorted(ret)
