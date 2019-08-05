from typing import Iterable

from derek.common.feature_extraction.continuous_converters import WordEmbeddingConverter, ExtrasIdentityConverter
from derek.common.feature_extraction.converters import create_categorical_converter
from derek.common.feature_extraction.factory_helper import collect_entities_types, collect_feature_labels, \
    init_categorical_features, get_converters_from_features_config

from derek.common.feature_extraction.features_meta import BasicFeaturesMeta
from derek.common.feature_extraction.helper import create_feature
from derek.data.helper import find_span_head_token
from derek.data.model import Document, Entity


def ne_type_for_token(doc, token_idx):
    entities = doc.extras['ne'].at_token(token_idx)
    return 'O' if not entities else entities[0].type


def create_entity_feature_extractor(docs: Iterable[Document], props: dict, shared_feature_extractor):
    continuous_converters = _init_continuous_converters(props, docs)
    vectorized_features = _get_vectorized_features(continuous_converters)

    encoder_config = _get_encoder_features(props, docs)
    encoder_embeded_features, encoder_one_hot_features = init_categorical_features(encoder_config)
    encoder_features_meta = BasicFeaturesMeta(encoder_embeded_features, encoder_one_hot_features, vectorized_features)
    encoder_features_convetrets = get_converters_from_features_config(encoder_config)

    return EntityFeatureExtractor(shared_feature_extractor, encoder_features_convetrets, continuous_converters,
                                  props.get('morph_feats_list', []),
                                  props.get('speech_types', []),
                                  props.get('identity_features', [])), encoder_features_meta


class EntityFeatureExtractor:

    def __init__(self, shared_fe, entity_converters, continuous_converters, feats_list, speech_types,
                 identity_head_features):
        self.shared_fe = shared_fe
        self.entity_converters = entity_converters
        self.continuous_converters = continuous_converters
        self.feats_list = feats_list
        self.speech_types = speech_types
        self.identity_head_features = identity_head_features

    def extract_features(self, doc: Document, entity: Entity) -> dict:
        features = self.shared_fe.extract_features_from_doc(doc, entity.start_token, entity.end_token)
        if 'encoder_entity_types' in self.entity_converters:
            features['encoder_entity_types'] = self.entity_converters['encoder_entity_types'][entity.type]

        head = find_span_head_token(doc, entity)
        for feat_name in self.feats_list:
            name = "encoder_" + feat_name
            if name in self.entity_converters:
                features[name] = self.entity_converters[name][doc.token_features[feat_name][head]]

        for speech_type in self.speech_types:
            name = "encoder_" + speech_type
            if name in self.entity_converters:
                has_type = any(list(map(lambda x: x != 'O',
                                        doc.token_features[speech_type][entity.start_token: entity.end_token])))
                features[name] = self.entity_converters[name][has_type]

        if 'head_we_in_encoder' in self.continuous_converters:
            features['head_we_in_encoder'] = self.continuous_converters['head_we_in_encoder'][doc.tokens[head].lower()]
        for name in self.identity_head_features:
            if name in self.continuous_converters:
                features[name] = self.continuous_converters[name][doc.token_features[name][head]]

        if 'encoder_entity_ne' in self.entity_converters:
            features['encoder_entity_ne'] = self.entity_converters['encoder_entity_ne'][ne_type_for_token(doc, head)]

        features['entity_seq_len'] = features['seq_len']
        del features['seq_len']

        return features

    def get_padding_value_and_rank(self, name):
        if name == 'entity_seq_len':
            return 0, 0
        if name in self.entity_converters:
            return self.entity_converters[name]['$PADDING$'], 0
        if name in self.continuous_converters:
            # padding must have the same type as values
            return 0.0, 1
        if name == "seq_len":
            return None

        return self.shared_fe.get_padding_value_and_rank(name)


def _init_continuous_converters(props, docs):
    converters = {}
    head_we_in_encoder_model = props.get('head_we_in_encoder_model', None)
    if head_we_in_encoder_model is not None:
        converters['head_we_in_encoder'] = WordEmbeddingConverter(head_we_in_encoder_model)
    for name in props.get('identity_features', []):
        converters[name] = ExtrasIdentityConverter(docs, name)
    return converters


def _get_encoder_features(props, docs):
    encoder_features = {}

    encoder_features.update(
        create_feature('encoder_entity_types', props,
                       create_categorical_converter(collect_entities_types(docs), zero_padding=True)))

    encoder_features.update(
        create_feature('encoder_entity_ne', props,
                       create_categorical_converter(collect_entities_types(docs, extras=True).union('O'),
                                                    zero_padding=True, has_oov=True)))

    speech_types = props.get('speech_types', [])
    speech_size = props.get('speech_size', -1)
    if speech_size >= 0:
        for speech_type in speech_types:
            encoder_features.update(create_feature('encoder_' + speech_type, props,
                                                   create_categorical_converter({True, False}, zero_padding=True),
                                                   'speech'))

    feats_types = props.get('morph_feats_list', [])
    feats_size = props.get('morph_feats_size', -1)
    if feats_size >= 0:
        for feat_name in feats_types:
            encoder_features.update(create_feature('encoder_' + feat_name, props,
                                                   create_categorical_converter(collect_feature_labels(docs, feat_name),
                                                                                zero_padding=True), 'morph_feats'))
    return encoder_features


def _get_vectorized_features(converters):
    ret = []

    for name, converter in converters.items():
        ret.append({'name': name, "size": len(converter)})
    return ret
