import random
from typing import List, Tuple

from derek.common.helper import FuncIterable
from derek.common.io import save_with_pickle, load_with_pickle
from derek.coref.feature_extraction.entity_feature_extractor import ne_type_for_token
from derek.data.helper import find_span_head_token, get_sentence_distance_between_entities
from derek.data.model import Document, Entity


class CorefFeatureExtractor:
    def __init__(self, entity_fe, sampling_strategy, rel_converter, classifier_converters, agreement_types):
        self.entity_fe = entity_fe
        self.rel_converter = rel_converter
        self.rel_reversed_converter = rel_converter.get_reversed_converter()
        self.sampling_strategy = sampling_strategy
        self.classifier_converters = classifier_converters
        self.agreement_types = agreement_types

    def extract_features_from_docs_iterator(self, docs, use_filter=False, drop_negative=0):
        def apply():
            for doc in docs:
                doc_samples, _ = self.extract_features_from_doc(doc, use_filter, include_labels=True)
                for sample in doc_samples:
                    if self.rel_reversed_converter[sample['labels']] is not None or random.random() >= drop_negative:
                        yield sample

        return FuncIterable(apply)

    def extract_features_from_docs(self, docs, use_filter=False):
        samples = []

        for doc in docs:
            doc_samples, _ = self.extract_features_from_doc(doc, use_filter, include_labels=True)
            samples += doc_samples

        return samples

    def extract_features_from_doc(self, doc: Document, use_filter=False, *, include_labels=False):

        entity_pairs = []
        candidates = self.sampling_strategy.apply(doc, use_filter=use_filter, include_labels=include_labels)
        samples, window_entity_pairs = self._extract_features(doc, candidates, include_labels=include_labels)

        entity_pairs.append(window_entity_pairs)

        return samples, entity_pairs

    def _extract_features(self, doc: Document, rels_list: List[Tuple[Entity, Entity, str]], *,
                          include_labels=False):
        samples = []
        entity_pairs = []
        for e1, e2, rel_type in rels_list:
            rel_feats = self._extract_pair_features(doc, e1, e2)
            if include_labels:
                rel_feats["labels"] = self.rel_converter[rel_type]
            samples.append(rel_feats)
            entity_pairs.append((e1, e2))

        return samples, entity_pairs

    def _extract_pair_features(self, doc: Document, e1: Entity, e2: Entity):
        context_entities = self._get_context_entities(doc, e1, e2)
        ret = {'seq_len': len(context_entities)}
        for entity in context_entities:
            features = self.entity_fe.extract_features(doc, entity)
            for name, val in features.items():
                ret.setdefault(name, []).append(val)
        ret.update(self._get_classifier_features(doc, e1, e2))
        return ret

    @staticmethod
    def _get_context_entities(doc: Document, e1: Entity, e2: Entity) -> List[Entity]:
        entities = []
        for entity in doc.entities:
            if e1.start_token <= entity.start_token and entity.end_token <= e2.end_token:
                head = find_span_head_token(doc, entity)
                entities.append((entity, head))
        entities = sorted(entities, key=lambda x: x[1])
        return list(map(lambda x: x[0], entities))

    def _get_classifier_features(self, doc: Document, e1: Entity, e2: Entity) -> dict:
        classifier_feats = {}
        extractors = [self._get_agreement, self._get_string_match,
                      self._get_mention_distance, self._get_mention_interrelation, self._get_entity_distance,
                      self._get_token_distance, self._get_sentence_distance, self._get_entity_types, self._get_ne_types]

        for extractor in extractors:
            classifier_feats.update(extractor(doc, e1, e2))
        return classifier_feats

    def _get_ne_types(self, doc: Document, e1: Entity, e2: Entity):
        ret = {}
        if 'head_ne_types_0' not in self.classifier_converters:
            return ret

        for i, entity in enumerate((e1, e2)):
            name = "head_ne_types_{}".format(i)
            head = find_span_head_token(doc, entity)
            ret[name] = self.classifier_converters[name][ne_type_for_token(doc, head)]
        return ret

    def _get_entity_types(self, doc: Document, e1: Entity, e2: Entity):
        ret = {}
        if 'entities_types_in_classifier_0' not in self.classifier_converters:
            return ret

        for i, entity in enumerate((e1, e2)):
            name = "entities_types_in_classifier_{}".format(i)
            ret[name] = self.classifier_converters[name][entity.type]
        return ret

    def _get_agreement(self, doc: Document, e1: Entity, e2: Entity):
        ret = {}

        e1_main_token = find_span_head_token(doc, e1)
        e2_main_token = find_span_head_token(doc, e2)

        e1_feats = doc.token_features['feats'][e1_main_token]
        e2_feats = doc.token_features['feats'][e2_main_token]

        for key in self.agreement_types:
            name = key + '_agreement'
            if name in self.classifier_converters:
                if key in e1_feats and key in e2_feats and e1_feats[key] == e2_feats[key]:
                    label = 'agreement'
                elif key not in e1_feats or key not in e2_feats:
                    label = 'unknown'
                else:
                    label = 'disagreement'
                ret[name] = self.classifier_converters[name][label]
        return ret

    def _get_string_match(self, doc: Document, e1: Entity, e2: Entity):
        ret = {}

        if 'head_str_match' in self.classifier_converters:
            e1_main_token = find_span_head_token(doc, e1)
            e2_main_token = find_span_head_token(doc, e2)
            ret['head_str_match'] = self.classifier_converters['head_str_match'][
                doc.token_features['lemmas'][e1_main_token] == doc.token_features['lemmas'][e2_main_token]]

        e1_words = doc.token_features['lemmas'][e1.start_token: e1.end_token]
        e2_words = doc.token_features['lemmas'][e2.start_token: e2.end_token]
        if 'exact_str_match' in self.classifier_converters:
            ret['exact_str_match'] = self.classifier_converters['exact_str_match'][set(e1_words) == set(e2_words)]
        if 'partial_str_match' in self.classifier_converters:
            ret['partial_str_match'] = self.classifier_converters['partial_str_match'][bool(set(e1_words) &
                                                                                            set(e2_words))]
        if 'ordered_exact_str_match' in self.classifier_converters:
            ret['ordered_exact_str_match'] = self.classifier_converters['ordered_exact_str_match'][e1_words == e2_words]
        if 'ordered_partial_str_match' in self.classifier_converters:
            ret['ordered_partial_str_match'] = self.classifier_converters['ordered_partial_str_match'][
                self._get_ordered_partial_match(e1_words, e2_words) or
                self._get_ordered_partial_match(e2_words, e1_words)]
        return ret

    def _get_mention_distance(self, doc: Document, e1: Entity, e2: Entity):
        if 'mention_distance' not in self.classifier_converters:
            return {}
        distance = 0
        start = min((e1.end_token, e2.end_token))
        end = max((e1.start_token, e2.start_token))
        for entity in doc.entities:
            if start <= entity.start_token and entity.end_token <= end:
                distance += 1
        return {'mention_distance': self.classifier_converters['mention_distance'][distance]}

    def _get_token_distance(self, doc: Document, e1: Entity, e2: Entity):
        if 'entities_token_distance_in_classifier' not in self.classifier_converters:
            return {}
        return {'entities_token_distance_in_classifier':
                    self.classifier_converters['entities_token_distance_in_classifier'][e1.token_distance_to(e2)]}

    def _get_sentence_distance(self, doc: Document, e1: Entity, e2: Entity):
        if 'entities_sent_distance_in_classifier' not in self.classifier_converters:
            return {}
        return {
            'entities_sent_distance_in_classifier':
                self.classifier_converters['entities_sent_distance_in_classifier']
                [get_sentence_distance_between_entities(doc, e1, e2)]
        }

    def _get_mention_interrelation(self, doc: Document, e1: Entity, e2: Entity):
        if 'mention_interrelation' not in self.classifier_converters:
            return {}
        if e1.start_token <= e2.start_token and e2.end_token <= e1.end_token:
            interrelation_label = "CONTAINS"
        elif e2.start_token <= e1.start_token and e1.end_token <= e2.end_token:
            interrelation_label = "CONTAINED"
        elif e2.start_token <= e1.start_token < e2.end_token or e1.start_token <= e2.start_token < e1.end_token:
            interrelation_label = "INTERSECTS"
        else:
            interrelation_label = "SEPARATED"

        return {'mention_interrelation': self.classifier_converters['mention_interrelation'][interrelation_label]}

    def _get_ordered_partial_match(self, e1_words, e2_words):
        e1_idx = 0
        for idx, e2_word in enumerate(e2_words):
            if e1_idx >= len(e1_words):
                break
            if e1_words[e1_idx] == e2_word:
                e1_idx += 1

        return e1_idx == len(e1_words)

    def _get_entity_distance(self, doc: Document, e1: Entity, e2: Entity):
        if 'classifier_entity_distance' not in self.classifier_converters:
            return {}
        distance = 0

        start = min((e1.end_token, e2.end_token))
        end = max((e1.start_token, e2.start_token))

        for entity in doc.entities:
            if entity.end_token > end:
                break
            if entity.start_token >= start:
                distance += 1
        return {'classifier_entity_distance': self.classifier_converters['classifier_entity_distance'][distance]}

    def get_labels_size(self):
        return len(self.rel_converter)

    def get_type(self, val: int):
        return self.rel_reversed_converter[val]

    def get_padding_value_and_rank(self, name):
        if name == "labels":
            return 0, 0
        if name == "seq_len":
            return 0, 0

        if name in self.classifier_converters:
            return self.classifier_converters[name]["$PADDING$"], 0

        padding_rank = self.entity_fe.get_padding_value_and_rank(name)
        if padding_rank is None:
            return None

        padding, rank = padding_rank
        return padding, rank + 1

    def save(self, out_path):
        save_with_pickle(self, out_path, "feature_extractor.pkl")

    @staticmethod
    def load(path):
        return load_with_pickle(path, "feature_extractor.pkl")
