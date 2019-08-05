from logging import getLogger
from typing import Iterable

from derek.common.feature_extraction.converters import create_categorical_converter, create_unsigned_integers_converter
from derek.common.feature_extraction.factory_helper import get_categorical_meta_converters
from derek.common.feature_extraction.token_position_feature_extractor import generate_token_position_feature_extractor
from derek.rel_ext.feature_extraction.factory import duplicate_features_config
from derek.data.model import Document
from derek.rel_ext.feature_extraction.spans_feature_extractor import SpansCommonFeatureExtractor
from derek.rel_ext.syntactic.parser.feature_extraction.feature_extractor import ParserFeatureExtractor
from derek.rel_ext.feature_extraction.feature_meta import AttentionFeaturesMeta, Metas
from derek.rel_ext.syntactic.parser.feature_extraction.sampling_strategies import DefaultSamplingStrategy, \
    PosFilteringSamplingStrategy
from derek.common.feature_extraction.features_meta import get_empty_basic_meta

logger = getLogger('logger')


def generate_feature_extractor(
        docs: Iterable[Document], props: dict, shared_feature_extractor: SpansCommonFeatureExtractor):

    strategy = _get_sampling_strategy(props)
    arc_converter = create_categorical_converter(strategy.get_possible_arc_types(docs), zero_padding=False)

    token_position_fe, tp_meta = generate_token_position_feature_extractor(props)
    attention_token_meta = tp_meta.namespaced('head') + tp_meta.namespaced('dep')

    attention_arc_meta, attention_converters = get_categorical_meta_converters(
        _get_arc_level_features(props, "attention"))

    attention_meta = AttentionFeaturesMeta(attention_token_meta, attention_arc_meta)

    classifier_meta, classifier_converters = get_categorical_meta_converters(_get_arc_level_features(props))

    feature_extractor = ParserFeatureExtractor(
        shared_feature_extractor, arc_converter, token_position_fe,
        attention_converters, classifier_converters, strategy)

    return feature_extractor, Metas(get_empty_basic_meta(), attention_meta, classifier_meta)


def _get_sampling_strategy(props):
    props_strategy = props.get("sampling_strategy", "default")

    if props_strategy == "default":
        return DefaultSamplingStrategy()
    elif props_strategy == "pos_filtering":
        return PosFilteringSamplingStrategy(set(props.get("pos_filter_types", ["PUNCT"])))


def _get_arc_level_features(props, name_postfix="classifier"):
    dual_features = {}
    single_features = {}

    token_distance_feature_name = "arc_token_distance_in_{}".format(name_postfix)
    token_distance_size = props.get(token_distance_feature_name + "_size", -1)

    if token_distance_size >= 0:
        single_features[token_distance_feature_name] = {
            "converter": create_unsigned_integers_converter(props["max_arc_token_distance"])
        }

        if token_distance_size != 0:
            single_features[token_distance_feature_name]["embedding_size"] = token_distance_size

    return duplicate_features_config(dual_features, single_features)
