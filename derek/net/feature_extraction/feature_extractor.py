from collections import defaultdict
from itertools import chain, starmap
from logging import getLogger
from typing import Iterable, Union, List, Tuple, Dict, Iterator, Any
from warnings import warn

from derek.common.feature_extraction.converters import create_categorical_converter
from derek.common.feature_extraction.factory import generate_token_feature_extractor
from derek.common.feature_extraction.factory_helper import get_categorical_meta_converters
from derek.common.feature_extraction.features_meta import get_empty_basic_meta
from derek.common.feature_extraction.token_position_feature_extractor import generate_token_position_feature_extractor
from derek.common.helper import FuncIterable
from derek.common.io import save_with_pickle, load_with_pickle
from derek.data.model import Document, Entity
from derek.rel_ext.feature_extraction.feature_meta import AttentionFeaturesMeta, Metas

logger = getLogger('logger')


def generate_feature_extractor(docs: Iterable[Document], props: dict, char_padding_size: int = 0):
    token_feature_extractor, token_features_meta = generate_token_feature_extractor(docs, props, char_padding_size)
    types_mapping = _get_ne_type_to_label_mapping(docs, set(props.get("restricted_ne_types", set())))
    valid_ne_types = types_mapping.keys()
    valid_ent_types = set(chain.from_iterable(types_mapping.values()))
    types_converter = create_categorical_converter(valid_ent_types, zero_padding=False)

    token_position_fe, tp_meta = generate_token_position_feature_extractor(props)

    attention_ent_meta, attention_converters = get_categorical_meta_converters(
        _get_features(props, valid_ne_types, "attention"))

    attention_meta = AttentionFeaturesMeta(tp_meta, attention_ent_meta)

    classifier_meta, classifier_converters = get_categorical_meta_converters(_get_features(props, valid_ne_types))

    feature_extractor = NETFeatureExtractor(
        token_feature_extractor, token_position_fe, attention_converters, classifier_converters,
        types_converter, types_mapping)
    # no encoder meta in this task
    metas = Metas(get_empty_basic_meta(), attention_meta, classifier_meta)

    return feature_extractor, metas, token_features_meta


def _get_ne_type_to_label_mapping(docs: Iterable[Document], restricted_ne_types: set):
    valid_types = defaultdict(set)

    for doc in docs:
        for ne in doc.extras["ne"]:
            if ne.type in restricted_ne_types:
                continue
            label = _map_ne(doc, ne)
            valid_types[ne.type].add(label)

    need_training = False
    for key, labels in valid_types.items():
        if len(labels) == 1:
            label = next(iter(labels))
            if label is None:
                warn(f"{key} NEs are not mapped to any entity")
            else:
                warn(f"{key} NEs are always mapped to {label} entity")
        else:
            need_training = True

    if not need_training:
        raise Exception("Each train NE type mapped to single entity type")

    logger.info(f"Valid entity types: {dict(valid_types)}")

    return valid_types


def _map_ne(doc: Document, ne: Entity):
    ents_at_ne = doc.entities.contained_in(ne)

    for ent in ents_at_ne:
        if ne.coincides(ent):
            return ent.type

    return None


def _get_features(props, ne_types, name_postfix="classifier"):
    features = {}

    ne_type_feature_name = "ne_type_in_{}".format(name_postfix)
    ne_type_size = props.get(ne_type_feature_name + "_size", -1)

    if ne_type_size >= 0:
        features[ne_type_feature_name] = {
            "converter": create_categorical_converter(ne_types)
        }

        if ne_type_size > 0:
            features[ne_type_feature_name]["embedding_size"] = ne_type_size

    return features


class NETFeatureExtractor:
    def __init__(
            self, token_feature_extractor, token_position_fe,
            attention_features_converters, classifier_features_converters,
            label_converter, types_mapping):

        self.token_feature_extractor = token_feature_extractor
        self.token_position_fe = token_position_fe
        self.attention_features_converters = attention_features_converters
        self.classifier_features_converters = classifier_features_converters

        self.label_converter = label_converter
        self.reversed_label_converter = label_converter.get_reversed_converter()
        self.types_mapping = types_mapping

    def extract_features_from_docs(self, docs) -> Iterable:
        def extract_samples(doc):
            ent_samples = self.extract_features_from_doc(doc, include_labels=True)
            return [sample for ent, sample in ent_samples if isinstance(sample, dict)]

        return FuncIterable(lambda: chain.from_iterable(map(extract_samples, docs)))

    def extract_features_from_doc(self, doc: Document, *, include_labels=False) \
            -> List[Tuple[Entity, Union[dict, str, None]]]:
        samples = []

        for ent in doc.extras["ne"]:
            valid_types = self.types_mapping.get(ent.type, {None})

            if len(valid_types) == 1:
                sample = next(iter(valid_types))
            else:
                sample = self._extract_features(doc, ent, include_labels)

            samples.append((ent, sample))

        return samples

    def _extract_features(self, doc: Document, ent: Entity, include_labels=False):
        ent_sent_idx = doc.get_entity_sent_idx(ent)

        start_token = doc.sentences[ent_sent_idx].start_token
        end_token = doc.sentences[ent_sent_idx].end_token

        features = {
            **self.token_feature_extractor.extract_features_from_doc(doc, start_token, end_token),
            **self._get_attention_features(doc, ent, start_token, end_token),
            **self._get_classifier_features(doc, ent)
        }

        if include_labels:
            label = _map_ne(doc, ent)
            features["labels"] = self.label_converter[label]

        ent_mask = [0] * len(self.label_converter)
        for key in self.types_mapping[ent.type]:
            ent_mask[self.label_converter[key]] = 1

        features["labels_mask"] = ent_mask

        features["indices"] = [[ent.start_token - start_token, ent.end_token - start_token]]

        return features

    def _get_attention_features(
            self, doc: Document, ent: Entity, start_token: int, end_token: int) -> dict:

        attention_features = {}

        wrt_span = (ent.start_token, ent.end_token, doc.get_entity_sent_idx(ent))
        position_features = self.token_position_fe.extract_features_from_doc(doc, start_token, end_token, wrt_span)
        attention_features.update(position_features)

        attention_features.update(
            self._get_entity_features(doc, ent, self.attention_features_converters, "attention"))

        return attention_features

    def _get_classifier_features(self, doc: Document, ent: Entity) -> dict:
        return self._get_entity_features(doc, ent, self.classifier_features_converters, "classifier")

    @staticmethod
    def _get_entity_features(doc, ent: Entity, converters, name_postfix):
        features = {}

        feature_name = "ne_type_in_{}".format(name_postfix)
        converter = converters.get(feature_name, None)
        if converter is not None:
            features[feature_name] = converter[ent.type]

        return features

    def get_labels_size(self):
        return len(self.label_converter)

    def get_type(self, val: int):
        return self.reversed_label_converter[val]

    def get_padding_value_and_rank(self, name):
        if name == "labels":
            return 0, 0

        if name == "labels_mask":
            return 0, 1

        if name == "indices":
            return 0, 2

        if name in self.attention_features_converters:
            return self.attention_features_converters[name]['$PADDING$'], 0

        if name in self.classifier_features_converters:
            return self.classifier_features_converters[name]['$PADDING$'], 0

        value = self.token_position_fe.get_padding_value_and_rank(name)
        if value is not None:
            return value

        return self.token_feature_extractor.get_padding_value_and_rank(name)

    def save(self, out_path):
        save_with_pickle(self, out_path, "feature_extractor.pkl")

    @staticmethod
    def load(path):
        return load_with_pickle(path, "feature_extractor.pkl")


class GroupingFeatureExtractor:
    def __init__(self, feature_extractor, *,
                 group_len_feature: str = "chain_len", group_level_features: Iterable[str] = tuple()):
        self.__feature_extractor = feature_extractor
        self.__group_len_feature = group_len_feature
        self.__group_level_features = group_level_features

    def extract_features_from_docs(self, docs, docs_groups) -> Iterable:
        def extract_samples(doc, groups):
            group_samples = self.extract_features_from_doc(doc, groups, include_labels=True)
            return [sample for _, sample in group_samples if isinstance(sample, dict)]

        return FuncIterable(lambda: chain.from_iterable(starmap(extract_samples, zip(docs, docs_groups))))

    def extract_features_from_doc(self, doc: Document, groups: List[Tuple[Any]], *, include_labels=False) \
            -> List[Tuple[Tuple[Any], Union[dict, str, None]]]:
        samples = self.__feature_extractor.extract_features_from_doc(doc, include_labels=include_labels)
        obj2features = {obj: feats for obj, feats in samples}
        return [(group, self._extract_features(group, obj2features, include_labels)) for group in groups]

    def _extract_features(self, group: Tuple[Any], obj2features: Dict[Any, Union[dict, str, None]], include_labels=False):
        if not len(group):
            warn("Empty group")
            return None
        first_features = obj2features[group[0]]
        group_features = map(obj2features.get, group)
        if isinstance(first_features, dict):
            merged = self._merge_dicts(group_features)
            merged[self.__group_len_feature] = len(group)
            for group_level in self.__group_level_features:
                if merged[group_level]:
                    merged[group_level] = self._validate_equals(iter(merged[group_level]), group_level)
                else:
                    warn(f"{group_level} feature is not presented in nested feature extractor")
            if include_labels:
                merged["labels"] = self._validate_equals(iter(merged["labels"]), "labels")
            return merged
        else:
            return self._validate_equals(group_features)

    @staticmethod
    def _merge_dicts(group_features: Iterator[Union[dict, str, None]]) -> dict:
        result = defaultdict(list)
        for obj_features in group_features:
            if not isinstance(obj_features, dict):
                raise Exception(f"Inconsistent group features")
            for key, value in obj_features.items():
                result[key].append(value)
        return result

    @staticmethod
    def _validate_equals(group_values: Iterator[Any], name: str = 'features') -> Any:
        value = next(group_values)
        if not all(features == value for features in group_values):
            raise Exception(f"Inconsistent group {name}")
        return value

    def get_labels_size(self):
        return self.__feature_extractor.get_labels_size()

    def get_type(self, val: int):
        return self.__feature_extractor.get_type(val)

    def get_padding_value_and_rank(self, name):
        padding_rank = self.__feature_extractor.get_padding_value_and_rank(name)
        if padding_rank is not None:
            padding, rank = padding_rank
            if name in self.__group_level_features or name == "labels":
                return padding, rank
            else:
                return padding, rank + 1

        if name == self.__group_len_feature:
            return 0, 0

    def save(self, out_path):
        save_with_pickle(self, out_path, "feature_extractor.pkl")

    @staticmethod
    def load(path):
        return load_with_pickle(path, "feature_extractor.pkl")
