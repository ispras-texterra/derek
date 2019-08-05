from typing import Iterable

from derek.common.feature_extraction.converters import \
    create_categorical_converter, create_unsigned_integers_converter
from derek.common.feature_extraction.factory import init_categorical_features
from derek.common.feature_extraction.factory_helper import get_converters_from_features_config, \
    collect_entities_types
from derek.common.feature_extraction.features_meta import BasicFeaturesMeta
from derek.coref.feature_extraction.entity_feature_extractor import create_entity_feature_extractor
from derek.coref.feature_extraction.feature_extractor import CorefFeatureExtractor
from derek.common.feature_extraction.helper import create_feature
from derek.coref.feature_extraction.heuristics import CLASSIFIERS, ComplexFilter
from derek.coref.feature_extraction.sampling_strategies import CorefPairExtractionStrategy, NounPreprocessingStrategy, \
    PronPairExtractionStrategy, CorefCandidateMaker, OneGroupSamplingStrategy, ClusterPronPairExtractionStrategy
from derek.data.model import Document
from derek.rel_ext.feature_extraction.factory import create_rel_dict, duplicate_features_config
from derek.rel_ext.feature_extraction.feature_meta import Metas, get_empty_attention_meta


def create_coref_feature_extractor(docs: Iterable[Document], props: dict, shared_feature_extractor):
    rel_dict = create_rel_dict([doc.relations for doc in docs])
    sampling_strategy = _get_coref_sampling_startegy(props)

    classifier_config = _get_classifier_features(props, docs)
    classifier_features_meta = BasicFeaturesMeta(*init_categorical_features(classifier_config))
    classifier_features_converters = get_converters_from_features_config(classifier_config)

    entity_fe, encoder_features_meta = create_entity_feature_extractor(docs, props, shared_feature_extractor)

    coref_feature_extrator = CorefFeatureExtractor(entity_fe, sampling_strategy, rel_dict,
                                                   classifier_features_converters, props.get('agreement_types', []))

    return coref_feature_extrator, Metas(encoder_features_meta, get_empty_attention_meta(), classifier_features_meta)


def _get_coref_sampling_startegy(props):
    if props.get("sampling_strategy", "coref") == "coref_noun":
        pair_extraction = CorefPairExtractionStrategy(props["max_entity_distance"], NounPreprocessingStrategy())
    elif props.get("sampling_strategy", "coref") == 'coref_pron_cluster':
        pair_extraction = ClusterPronPairExtractionStrategy(props["max_entity_distance"])
    elif props.get("sampling_strategy", "coref") == 'coref_pron_cluster_strict':
        pair_extraction = ClusterPronPairExtractionStrategy(props["max_entity_distance"], True)
    elif props.get("sampling_strategy", "coref") == 'coref_pron':
        pair_extraction = PronPairExtractionStrategy(props["max_entity_distance"])
    elif props.get("sampling_strategy", "coref") == 'coref':
        pair_extraction = CorefPairExtractionStrategy(props["max_entity_distance"])
    else:
        raise Exception("Unknown strategy!")

    classifiers = []
    for name in props.get('classifiers', []):
        classifiers.append(CLASSIFIERS[name])

    return OneGroupSamplingStrategy(pair_extraction, CorefCandidateMaker('COREF'),
                                    pair_filter=ComplexFilter(classifiers))


def _get_classifier_features(props, docs):
    classifier_features = {}
    dual_features = {}

    classifier_agreement_size = props.get('classifier_agreement_size', -1)
    if classifier_agreement_size >= 0:
        agreement_types = props.get('agreement_types', [])
        for agreement_type in agreement_types:
            converter = create_categorical_converter({"agreement", "disagreement", "unknown"})
            classifier_features.update(
                create_feature(agreement_type + "_agreement", props, converter, 'classifier_agreement'))

    classifier_features.update(
        create_feature('mention_distance', props, 
                       create_unsigned_integers_converter(props["max_mention_distance"])))

    classifier_features.update(
        create_feature('mention_interrelation', props, 
                       create_categorical_converter({"CONTAINS", "CONTAINED", "INTERSECTS", "SEPARATED"})))

    classifier_features.update(
        create_feature('classifier_entity_distance', props, 
                       create_unsigned_integers_converter(props["max_entity_distance"])))

    classifier_features.update(
        create_feature('entities_token_distance_in_classifier', props,
                       create_unsigned_integers_converter(props["max_token_entities_distance"])))
    
    classifier_features.update(
        create_feature('entities_sent_distance_in_classifier', props,
                       create_unsigned_integers_converter(props["max_sent_entities_distance"])))

    dual_features.update(
        create_feature('entities_types_in_classifier', props,
                       create_categorical_converter(collect_entities_types(docs))))

    dual_features.update(
        create_feature('head_ne_types', props,
                       create_categorical_converter(collect_entities_types(docs, extras=True).union('O'), has_oov=True)))

    classifier_features.update(_get_binary_features(props))

    return duplicate_features_config(dual_features, classifier_features)


def _get_binary_features(props):
    classifier_features = {}
    binary_feature_names = ['exact_str_match', 'head_str_match', 'partial_str_match',
                            'ordered_exact_str_match', 'ordered_partial_str_match']

    for feature_name in binary_feature_names:
        feature_size = props.get(feature_name + '_size', -1)
        if feature_size >= 0:
            classifier_features[feature_name] = _create_binary_feature(feature_size)
    return classifier_features


def _create_binary_feature(size):
    feature = {
        "converter": create_categorical_converter({True, False})
    }
    if size != 0:
        feature["embedding_size"] = size
    return feature
