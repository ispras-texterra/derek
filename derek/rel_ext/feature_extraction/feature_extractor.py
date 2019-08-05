from itertools import chain
from typing import Iterable

from derek.common.helper import FuncIterable, namespaced, from_namespace
from derek.common.io import load_with_pickle, save_with_pickle
from derek.data.helper import get_sentence_distance_between_entities
from derek.data.model import Document, Entity
from derek.common.feature_extraction.helper import encode_sequence


class RelExtFeatureExtractor:
    def __init__(self, shared_feature_extractor, rel_converter, entities_encoder_features_converters,
                 token_position_fe, attention_features_converters, classifier_features_converters, candidate_extractor,
                 valid_ent_rel_types):

        self.shared_feature_extractor = shared_feature_extractor
        self.rel_converter = rel_converter
        self.rel_reversed_converter = rel_converter.get_reversed_converter()
        self.entities_encoder_features_converters = entities_encoder_features_converters
        self.token_position_fe = token_position_fe
        self.attention_features_converters = attention_features_converters
        self.classifier_features_converters = classifier_features_converters
        self.candidate_extractor = candidate_extractor
        self.valid_ent_rel_types = valid_ent_rel_types

    def extract_features_from_docs(self, docs) -> Iterable:
        return FuncIterable(lambda: chain.from_iterable(
            map(lambda doc: self.extract_features_from_doc(doc, include_labels=True)[0], docs)))

    def extract_features_from_doc(self, doc: Document, *, include_labels=False):
        samples = []
        entity_pairs = []

        for e1, e2, rel_type in self.candidate_extractor.apply(doc, include_labels=include_labels):
            sample = self._extract_features(doc, e1, e2, rel_type, include_labels=include_labels)
            samples.append(sample)
            entity_pairs.append((e1, e2))

        return samples, entity_pairs

    def _extract_features(self, doc: Document, e1: Entity, e2: Entity, rel_type: str, *, include_labels=False):
        e1_sent_idx = doc.get_entity_sent_idx(e1)
        e2_sent_idx = doc.get_entity_sent_idx(e2)

        start_token = doc.sentences[min(e1_sent_idx, e2_sent_idx)].start_token
        end_token = doc.sentences[max(e1_sent_idx, e2_sent_idx)].end_token

        e1_wrt_span = (e1.start_token, e1.end_token, e1_sent_idx)
        e2_wrt_span = (e2.start_token, e2.end_token, e2_sent_idx)

        features = {
            **self.shared_feature_extractor.extract_features_from_doc(
                doc, start_token, end_token, e1_wrt_span, e2_wrt_span),
            **self._get_entities_encoder_features(doc, start_token, end_token),
            **self._get_attention_features(doc, e1, e2, start_token, end_token),
            **self._get_classifier_features(doc, e1, e2)
        }

        if include_labels:
            features["labels"] = self.rel_converter[rel_type]

        rel_mask = [0] * len(self.rel_converter)
        rel_mask[self.rel_converter[None]] = 1
        for key in self.valid_ent_rel_types[e1.type, e2.type]:
            rel_mask[self.rel_converter[key]] = 1
        features["labels_mask"] = rel_mask

        features["indices"] = [
            [e1.start_token - start_token, e1.end_token - start_token],
            [e2.start_token - start_token, e2.end_token - start_token]
        ]

        return features

    def _get_entities_encoder_features(self, doc, start_token: int, end_token: int) -> dict:
        features = {}

        # currently we can have here only entitites types and depths, use common pattern
        for feature, feature_converter in self.entities_encoder_features_converters.items():
            features[feature] = encode_sequence(
                doc.token_features[feature][start_token: end_token], feature_converter)

        return features

    def _get_attention_features(
            self, doc: Document, e1: Entity, e2: Entity, start_token: int, end_token: int) -> dict:

        attention_features = {}
        for namespace, ent in zip(['e1', 'e2'], [e1, e2]):
            wrt_span = (ent.start_token, ent.end_token, doc.get_entity_sent_idx(ent))
            position_features = self.token_position_fe.extract_features_from_doc(doc, start_token, end_token, wrt_span)
            attention_features.update(namespaced(position_features, namespace))

        attention_features.update(
            self._get_relation_features(doc, e1, e2, self.attention_features_converters, "attention"))

        return attention_features

    def _get_classifier_features(self, doc: Document, e1: Entity, e2: Entity) -> dict:
        return self._get_relation_features(doc, e1, e2, self.classifier_features_converters, "classifier")

    @staticmethod
    def _get_relation_features(doc, e1: Entity, e2: Entity, converters, name_postfix):
        features = {}

        feature_name = "rel_args_in_{}".format(name_postfix)
        converter = converters.get(feature_name, None)
        if converter is not None:
            features[feature_name] = converter[(e1.type, e2.type)]

        feature_name = "entities_token_distance_in_{}".format(name_postfix)
        converter = converters.get(feature_name, None)
        if converter is not None:
            features[feature_name] = converter[e1.token_distance_to(e2)]

        feature_name = "entities_token_log_distance_in_{}".format(name_postfix)
        converter = converters.get(feature_name, None)
        if converter is not None:
            features[feature_name] = converter[e1.token_distance_to(e2)]

        feature_name = "entities_sent_distance_in_{}".format(name_postfix)
        converter = converters.get(feature_name, None)
        if converter is not None:
            features[feature_name] = converter[get_sentence_distance_between_entities(doc, e1, e2)]

        feature_name = "rel_dir_in_{}".format(name_postfix)
        converter = converters.get(feature_name, None)
        if converter is not None:
            features[feature_name] = converter[RelExtFeatureExtractor._get_direction_feature(e1, e2)]

        for ent_num, ent in enumerate((e1, e2)):
            feature_name = "entities_types_in_{}_{}".format(name_postfix, ent_num)
            converter = converters.get(feature_name, None)
            if converter is not None:
                features[feature_name] = converter[ent.type]

        return features

    def get_labels_size(self):
        return len(self.rel_converter)

    def get_type(self, val: int):
        return self.rel_reversed_converter[val]

    def get_padding_value_and_rank(self, name):
        if name == "labels":
            return 0, 0

        if name == "labels_mask":
            return 0, 1

        if name == "indices":
            return 0, 2

        if name in self.classifier_features_converters:
            return self.classifier_features_converters[name]["$PADDING$"], 0

        if name in self.attention_features_converters:
            return self.attention_features_converters[name]['$PADDING$'], 0

        if name in self.entities_encoder_features_converters:
            return self.entities_encoder_features_converters[name]['$PADDING$'], 1

        for namespace in ["e1", "e2"]:
            value = self.token_position_fe.get_padding_value_and_rank(from_namespace(name, namespace))
            if value is not None:
                return value

        return self.shared_feature_extractor.get_padding_value_and_rank(name)

    @staticmethod
    def _get_direction_feature(e1: Entity, e2: Entity):
        if e1.contains(e2):
            return "e2_in_e1"
        if e2.contains(e1):
            return "e1_in_e2"
        if e1.start_token < e2.start_token:
            return "e1_e2"

        return "e2_e1"

    def save(self, out_path):
        save_with_pickle(self, out_path, "feature_extractor.pkl")

    @staticmethod
    def load(path):
        return load_with_pickle(path, "feature_extractor.pkl")
