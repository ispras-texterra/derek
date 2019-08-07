from collections import defaultdict
from logging import getLogger
from typing import Iterable

from derek.common.feature_extraction.converters import create_categorical_converter,\
    create_unsigned_integers_converter, create_unsigned_log_integers_converter
from derek.common.feature_extraction.factory_helper import get_categorical_meta_converters, collect_entities_types
from derek.common.feature_extraction.token_position_feature_extractor import generate_token_position_feature_extractor
from derek.data.model import Document
from derek.rel_ext.feature_extraction.feature_extractor import NegativeSamplesFilteringFeatureExtractor
from derek.rel_ext.feature_extraction.sampling_strategies import DefaultPairExtractionStrategy,\
    DefaultCandidateExtractionStrategy, DifferentEntitiesCandidateFilter, InSameSentenceCandidateFilter,\
    MaxTokenDistanceCandidateFilter, RelArgTypesCandidateFilter, IntersectingCandidateFilter, AndFilter
from derek.rel_ext.feature_extraction.feature_meta import AttentionFeaturesMeta, Metas
from derek.rel_ext.feature_extraction.spans_feature_extractor import SpansCommonFeatureExtractor

logger = getLogger('logger')


def generate_feature_extractor(
        docs: Iterable[Document], props: dict, shared_feature_extractor: SpansCommonFeatureExtractor):

    entities_types = collect_entities_types(docs)

    pair_filters = [
        DifferentEntitiesCandidateFilter(),
        InSameSentenceCandidateFilter(),
        MaxTokenDistanceCandidateFilter(props['max_candidate_distance'])
    ]

    if props.get("filter_intersecting", False):
        pair_filters.append(IntersectingCandidateFilter())

    rels = [_filter_doc_rels(doc, AndFilter(pair_filters)) for doc in docs]
    rel_arg_types = create_rel_arg_types(rels)
    pair_filters.append(RelArgTypesCandidateFilter(rel_arg_types))
    candidate_extractor = DefaultCandidateExtractionStrategy(DefaultPairExtractionStrategy(AndFilter(pair_filters)))

    rel_dict = create_rel_dict(rels)
    valid_ent_rel_types = collect_valid_rel_types(rels)

    entities_encoder_meta, entities_encoder_converters = get_categorical_meta_converters(
        _get_entities_encoder_features(props, entities_types))

    token_position_fe, tp_meta = generate_token_position_feature_extractor(props)
    attention_token_meta = tp_meta.namespaced('e1') + tp_meta.namespaced('e2')

    attention_rel_meta, attention_converters = get_categorical_meta_converters(
        _get_relation_level_features(props, rel_arg_types, entities_types, "attention"))

    attention_meta = AttentionFeaturesMeta(attention_token_meta, attention_rel_meta)

    classifier_meta, classifier_converters = get_categorical_meta_converters(
        _get_relation_level_features(props, rel_arg_types, entities_types))

    feature_extractor = NegativeSamplesFilteringFeatureExtractor(
        shared_feature_extractor, rel_dict, entities_encoder_converters,
        token_position_fe, attention_converters, classifier_converters, candidate_extractor, valid_ent_rel_types,
        negative_ratio=props.get("negative_samples_ratio", float("inf"))
    )

    return feature_extractor, Metas(entities_encoder_meta, attention_meta, classifier_meta)


def _filter_doc_rels(doc, pair_filter):
    return set(r for r in doc.relations if pair_filter.apply(doc, r.first_entity, r.second_entity))


def _get_relation_level_features(props, rel_arg_types, entities_types, name_postfix="classifier"):
    dual_features = {}
    single_features = {}

    rel_args_feature_name = "rel_args_in_{}".format(name_postfix)
    rel_args_size = props.get("rel_args_in_{}_size".format(name_postfix), -1)

    if rel_args_size >= 0:
        single_features[rel_args_feature_name] = {
            "converter": create_categorical_converter(rel_arg_types)
        }

        if rel_args_size != 0:
            single_features[rel_args_feature_name]["embedding_size"] = rel_args_size

    entities_types_feature_name = "entities_types_in_{}".format(name_postfix)
    entities_types_size = props.get(entities_types_feature_name + "_size", -1)

    if entities_types_size >= 0:
        dual_features[entities_types_feature_name] = {
            "converter": create_categorical_converter(entities_types)
        }

        if entities_types_size > 0:
            dual_features[entities_types_feature_name]["embedding_size"] = entities_types_size

    token_distance_feature_name = "entities_token_distance_in_{}".format(name_postfix)
    token_distance_size = props.get(token_distance_feature_name + "_size", -1)

    if token_distance_size >= 0:
        single_features[token_distance_feature_name] = {
            "converter": create_unsigned_integers_converter(props["max_token_entities_distance"])
        }

        if token_distance_size != 0:
            single_features[token_distance_feature_name]["embedding_size"] = token_distance_size

    token_log_distance_feature_name = "entities_token_log_distance_in_{}".format(name_postfix)
    token_log_distance_size = props.get(token_log_distance_feature_name + "_size", -1)

    if token_log_distance_size >= 0:
        single_features[token_log_distance_feature_name] = {
            "converter": create_unsigned_log_integers_converter(props["max_token_entities_distance"])
        }

        if token_log_distance_size != 0:
            single_features[token_log_distance_feature_name]["embedding_size"] = token_log_distance_size

    sent_distance_feature_name = "entities_sent_distance_in_{}".format(name_postfix)
    sent_distance_size = props.get(sent_distance_feature_name + "_size", -1)

    if sent_distance_size >= 0:
        single_features[sent_distance_feature_name] = {
            "converter": create_unsigned_integers_converter(props["max_sent_entities_distance"])
        }

        if sent_distance_size != 0:
            single_features[sent_distance_feature_name]["embedding_size"] = sent_distance_size

    rel_direction_feature_name = "rel_dir_in_{}".format(name_postfix)
    rel_direction_size = props.get(rel_direction_feature_name + "_size", -1)

    if rel_direction_size >= 0:
        categories = {"e1_e2", "e2_e1", "e1_in_e2", "e2_in_e1"}
        single_features[rel_direction_feature_name] = {
            "converter": create_categorical_converter(categories)
        }

        if rel_direction_size != 0:
            single_features[rel_direction_feature_name]["embedding_size"] = rel_direction_size

    return duplicate_features_config(dual_features, single_features)


def _get_entities_encoder_features(props, entities_types):
    entities_types = entities_types.union({None})
    features = {}

    if props.get("entities_types_emb_size", -1) >= 0:
        features['entities_types'] = {
            'converter': create_categorical_converter(entities_types)
        }
        if props["entities_types_emb_size"] != 0:
            features["entities_types"]['embedding_size'] = props["entities_types_emb_size"]

    if props.get("entities_depth_emb_size", -1) >= 0:
        features['entities_depths'] = {
            'converter': create_unsigned_integers_converter(props["max_entities_depth"])
        }
        if props["entities_depth_emb_size"] != 0:
            features["entities_depths"]['embedding_size'] = props["entities_depth_emb_size"]

    return features


def duplicate_features_config(dual_features_config, single_features_config):
    config = {}

    for key, value in dual_features_config.items():
        for i in range(2):
            config[key + "_{}".format(i)] = value

    config.update(single_features_config)

    return config


def create_rel_dict(rels: list):
    rel_types = {None}

    for doc_rels in rels:
        rel_types.update([x.type for x in doc_rels])

    return create_categorical_converter(rel_types, zero_padding=False)


def create_rel_arg_types(rels: list):
    result = set()
    for doc_rels in rels:
        for rel in doc_rels:
            result.add(rel.entities_types)
    return result


def collect_valid_rel_types(rels: list):
    ret = defaultdict(set)

    for doc_rels in rels:
        for rel in doc_rels:
            ret[rel.entities_types].add(rel.type)

    logger.info('Valid relation argument types:' + str(dict(ret)))
    return ret
