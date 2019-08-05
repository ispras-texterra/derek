from typing import List, Set, Dict, Optional

import numpy as np

from derek.coref.data.model import CoreferenceChain
from derek.data.model import Relation, Entity


def collect_chains(rels: Set[Relation], entities: Optional[List[Entity]]=None) -> List[CoreferenceChain]:
    if entities is not None:
        group_dict = {entity: {entity} for entity in entities}
    else:
        group_dict = {}

    for rel in rels:
        first_group = group_dict.setdefault(rel.first_entity, {rel.first_entity})
        second_group = group_dict.setdefault(rel.second_entity, {rel.second_entity})
        if not first_group and not second_group or first_group != second_group:
            new_group = first_group | second_group | {rel.first_entity, rel.second_entity}
            for entity in new_group:
                group_dict[entity] = new_group

    return _collect_chains_from_dict(group_dict)


def collect_easy_first_mention_chains(pairs: Dict[tuple, dict]):
    """
    :param pairs: list of tuples (scores, entity1, entity2)
    :return: list of relations
    """
    clusters = {}  # idx -> cluster
    group_dict = {}  # entity -> cluster
    unlinked = set()

    for (e1, e2), scores in sorted(pairs.items(), key=lambda x: -max(x[-1].values())):
        e1_cluster = group_dict.get(e1, {e1})
        e2_cluster = group_dict.get(e2, {e2})
        if e1_cluster == e2_cluster:
            continue

        e1_cluster_idx = _get_cluster_idx(clusters, e1_cluster)
        e2_cluster_idx = _get_cluster_idx(clusters, e2_cluster)

        if (e1_cluster_idx, e2_cluster_idx) in unlinked or (e2_cluster_idx, e1_cluster_idx) in unlinked:
            continue

        if max(scores, key=scores.get) is None:
            unlinked.add((e1_cluster_idx, e2_cluster_idx))
        else:
            additional_unlinked = set()
            for pair in unlinked:
                if e2_cluster_idx == pair[0]:
                    additional_unlinked.add((e1_cluster_idx, pair[1]))
                elif e2_cluster_idx == pair[1]:
                    additional_unlinked.add((e1_cluster_idx, pair[0]))
            unlinked |= additional_unlinked

            e1_cluster |= e2_cluster
            clusters[e1_cluster_idx] = e1_cluster
            clusters[e2_cluster_idx] = e1_cluster
            for entity in e1_cluster:
                group_dict[entity] = e1_cluster
    return _collect_chains_from_dict(group_dict)


def collect_easy_first_mention(pairs: Dict[tuple, dict]):
    """
    :param pairs: list of tuples (scores, entity1, entity2)
    :return: set of relations
    """
    chains = collect_easy_first_mention_chains(pairs)
    return set(chains2rels(chains))


def collect_pron_vote_rank(pairs: Dict[tuple, dict], known_rels):
    """
    This method collects relations from pairs with given class confidence, using knowledge about known relations.
    Known relations are used to get info about clusters. Mention is connected to cluster with score chosen as mean of
    all pair scores.
    :param pairs: scores of mention pairs
    :param known_rels: known relations
    :return: relations selected from pairs
    """
    entities = sum(map(lambda x: [x[0], x[1]], pairs), [])
    nouns = set(filter(lambda x: x.type != 'pron', entities))
    chains = collect_chains(known_rels, list(nouns))

    entities = set(filter(lambda x: x.type == 'pron', entities))
    rels = set()
    for entity in entities:
        best_score = 0
        best_candidate = None
        for chain in chains:
            if not chain.entities:
                continue
            chain_scores = []
            candidate = get_closest_entity(chain, entity, False)
            for e in chain.entities:
                score = None
                if (e, entity) in pairs:
                    score = pairs[(e, entity)]["COREF"]
                if (entity, e) in pairs:
                    score = pairs[(entity, e)]["COREF"]
                if score is not None:
                    chain_scores.append(score)
            chain_score = np.mean(chain_scores) if chain_scores else 0
            if best_candidate is None or best_score < chain_score:
                best_candidate = candidate
                best_score = chain_score
        if best_candidate is not None:
            rels.add(Relation(best_candidate, entity, "COREF"))

    return rels


def collect_pron_rank(pairs: Dict[tuple, dict]):
    entities = sum(map(lambda x: [x[0], x[1]], pairs), [])
    entities = set(filter(lambda x: x.type == 'pron', entities))
    relations = _get_rank_rels(entities, pairs)
    return relations


def collect_rank(pairs: Dict[tuple, dict]):
    """
    :param pairs: list of tuples (scores, entity1, entity2)
    :return: set of relations
    """
    entities = set(sum(map(lambda x: [x[0], x[1]], pairs), []))
    relations = _get_rank_rels(entities, pairs)
    return relations


def _get_rank_rels(entities, pairs):
    relations = set()
    for entity in entities:
        best_score = 0
        best_candidate = None
        best_label = None
        for (e1, e2), scores in pairs.items():
            if entity != e2:
                continue
            max_scores = max(scores.values())
            label = max(scores, key=scores.get)
            if max_scores > best_score and label is not None:
                best_score = max_scores
                best_candidate = e1
                best_label = label
        if best_candidate is not None:
            relations.add(Relation(entity, best_candidate, best_label))
    return relations


def get_collecting_strategy(name):
    if name == 'easy_first':
        return collect_easy_first_mention
    if name == 'rank':
        return collect_rank
    if name == 'pron_rank':
        return collect_pron_rank


def chains2rels(chains: List[CoreferenceChain]):
    return sum(list(map(lambda x: x.to_relations_chain(), chains)), [])


def _get_cluster_idx(clusters: dict, cluster: set):
    for key, val in clusters.items():
        if val == cluster:
            return key
    idx = len(clusters)
    clusters[idx] = cluster
    return idx


def _collect_chains_from_dict(group_dict):
    groups = []
    for group in group_dict.values():
        if group not in groups:
            groups.append(group)

    chains = map(lambda x: CoreferenceChain(x), groups)
    return sorted(chains, key=lambda x: x.entities[0].start_token)  # Sort all groups by start token of first entity


def get_closest_entity(chain: CoreferenceChain, entity: Entity, left_only):
    min_distance = None
    closest_entity = None
    for e in chain.entities:
        if left_only and e.start_token > entity.start_token:
            continue
        distance = e.token_distance_to(entity)
        if min_distance is None or min_distance > distance:
            min_distance = distance
            closest_entity = e
    return closest_entity
