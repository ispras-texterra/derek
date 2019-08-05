from collections import Iterable
from itertools import chain

from derek.common.helper import FuncIterable, from_namespace, namespaced
from derek.common.io import save_with_pickle, load_with_pickle
from derek.data.model import Document, Sentence


class ParserFeatureExtractor:
    def __init__(
            self, shared_feature_extractor, arc_converter, token_position_fe,
            attention_features_converters, classifier_features_converters, sampling_strategy):

        self.shared_feature_extractor = shared_feature_extractor
        self.arc_converter = arc_converter
        self.token_position_fe = token_position_fe
        self.attention_features_converters = attention_features_converters
        self.classifier_features_converters = classifier_features_converters
        self.sampling_strategy = sampling_strategy

    def extract_features_from_docs(self, docs) -> Iterable:
        return FuncIterable(lambda: chain.from_iterable(map(self.extract_features_from_doc, docs)))

    def extract_features_from_doc(self, doc: Document):
        ret = []
        for sent_idx in range(len(doc.sentences)):
            sent_samples = self._extract_samples_from_sent(doc, sent_idx)
            ret.extend(sent_samples)

        return ret

    def _extract_samples_from_sent(self, doc: Document, sent_idx: int):
        sent_samples = []
        sent = doc.sentences[sent_idx]

        for parent_idx_in_sent, child_idx_in_sent, arc_type in self.sampling_strategy.generate(doc, sent):
            parent_wrt_span = (
                parent_idx_in_sent + sent.start_token, parent_idx_in_sent + sent.start_token + 1, sent_idx)
            child_wrt_span = (
                child_idx_in_sent + sent.start_token, child_idx_in_sent + sent.start_token + 1, sent_idx)

            sample = {
                **self.shared_feature_extractor.extract_features_from_doc(
                    doc, sent.start_token, sent.end_token, parent_wrt_span, child_wrt_span),
                **self._get_attention_features(doc, sent, parent_idx_in_sent, child_idx_in_sent),
                **self._get_classifier_features(doc, sent, parent_idx_in_sent, child_idx_in_sent),
                "labels": self.arc_converter[arc_type],
                "indices": [[parent_idx_in_sent, parent_idx_in_sent + 1], [child_idx_in_sent, child_idx_in_sent + 1]]
            }

            sent_samples.append(sample)

        return sent_samples

    def _get_attention_features(self, doc: Document, sent: Sentence, t1_idx: int, t2_idx: int) -> dict:
        attention_features = {}
        for namespace, idx in zip(['head', 'dep'], [t1_idx, t2_idx]):
            position_features = self.token_position_fe.extract_features_from_doc(
                doc, sent.start_token, sent.end_token, (idx, idx + 1, doc.get_token_sent_idx(idx)))
            attention_features.update(namespaced(position_features, namespace))

        attention_features.update(
            self._get_arc_features(doc, sent, t1_idx, t2_idx, self.attention_features_converters, "attention"))

        return attention_features

    def _get_classifier_features(self, doc: Document, sent: Sentence, t1_idx: int, t2_idx: int) -> dict:
        return self._get_arc_features(doc, sent, t1_idx, t2_idx, self.classifier_features_converters, "classifier")

    @staticmethod
    def _get_arc_features(doc: Document, sent: Sentence, t1_idx: int, t2_idx: int, converters, name_postfix):
        features = {}

        feature_name = "arc_token_distance_in_{}".format(name_postfix)
        converter = converters.get(feature_name, None)
        if converter is not None:
            features[feature_name] = converter[abs(t1_idx - t2_idx)]

        return features

    def get_labels_size(self):
        return len(self.arc_converter)

    def get_padding_value_and_rank(self, name):
        if name == "labels":
            return 0, 0

        if name == "indices":
            return 0, 2

        if name in self.classifier_features_converters:
            return self.classifier_features_converters[name]["$PADDING$"], 0

        if name in self.attention_features_converters:
            return self.attention_features_converters[name]['$PADDING$'], 0

        for namespace in ["head", "dep"]:
            value = self.token_position_fe.get_padding_value_and_rank(from_namespace(name, namespace))
            if value is not None:
                return value

        return self.shared_feature_extractor.get_padding_value_and_rank(name)

    def save(self, out_path):
        save_with_pickle(self, out_path, "feature_extractor.pkl")

    @staticmethod
    def load(path):
        return load_with_pickle(path, "feature_extractor.pkl")
