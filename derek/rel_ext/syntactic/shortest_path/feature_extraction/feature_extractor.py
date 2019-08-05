from itertools import chain
from logging import getLogger

from derek.common.helper import FuncIterable
from derek.rel_ext.feature_extraction.spans_feature_extractor import SpansCommonFeatureExtractor
from derek.common.io import save_with_pickle, load_with_pickle
from derek.data.model import Document, Sentence
from derek.data.sdp_extraction import compute_sdp, compute_sdp_subtree

logger = getLogger('logger')


def generate_sdp_task_feature_extractor(props: dict, shared_feature_extractor: SpansCommonFeatureExtractor):
    pos_types = props.get("POS_types", None)
    pos_types = {("NN", "NN"), ("NNS", "NNS"), ("NN", "NNS"), ("FW", "NN"), ("FW", "NNS"), ("FW", "FW")} \
        if pos_types is None else \
        set(tuple(json_lst for json_lst in pos_types))

    labels_extr_strategy = _SDPSubtreeLabelsExtractionStrategy() if props.get("predict_subtree", False) \
        else _SDPLabelsExtractionStrategy()

    return SDPTaskFeatureExtractor(shared_feature_extractor, pos_types, labels_extr_strategy)


class _SDPLabelsExtractionStrategy:
    @staticmethod
    def extract(doc: Document, sent: Sentence, t1_idx: int, t2_idx: int):
        sdp = compute_sdp(doc, sent, t1_idx, t2_idx)
        return [int(idx in sdp) for idx in range(sent.start_token, sent.end_token)]

    @staticmethod
    def get_labels_num():
        return 2

    @staticmethod
    def get_padding_value_and_rank():
        return 0, 1


class _SDPSubtreeLabelsExtractionStrategy:
    @staticmethod
    def extract(doc: Document, sent: Sentence, t1_idx: int, t2_idx: int):
        sdp = compute_sdp(doc, sent, t1_idx, t2_idx)
        sdp_subtree = compute_sdp_subtree(doc, sent, sdp)
        sdp = set(sdp)

        return [1 if idx in sdp else 2 if idx in sdp_subtree else 0 for idx in range(sent.start_token, sent.end_token)]

    @staticmethod
    def get_labels_num():
        return 3

    @staticmethod
    def get_padding_value_and_rank():
        return 0, 1


class SDPTaskFeatureExtractor:
    def __init__(self, feature_extractor, pos_types, labels_extraction_strategy):
        self.feature_extractor = feature_extractor
        self.pos_types = pos_types
        self.labels_extraction_strategy = labels_extraction_strategy

    def extract_features_from_docs(self, docs):
        return FuncIterable(
            lambda: chain.from_iterable(map(lambda a: self.extract_features_from_doc(a, include_labels=True)[0], docs)))

    def extract_features_from_doc(self, doc: Document, include_labels=False):
        samples = []
        token_pairs = []
        for sent_idx in range(len(doc.sentences)):
            sent_samples, sent_token_pairs = self._extract_features(doc, sent_idx, include_labels)
            samples.extend(sent_samples)
            token_pairs.append(sent_token_pairs)

        return samples, token_pairs

    def _extract_features(self, doc: Document, sent_idx: int, include_labels):
        samples = []
        token_pairs = []
        sent = doc.sentences[sent_idx]

        for t1_idx in range(sent.start_token, sent.end_token):
            for t2_idx in range(t1_idx + 1, sent.end_token):
                t1_pos = doc.token_features["pos"][t1_idx]
                t2_pos = doc.token_features["pos"][t2_idx]

                if (t1_pos, t2_pos) not in self.pos_types and (t2_pos, t1_pos) not in self.pos_types:
                    continue

                t1_wrt_span = (t1_idx, t1_idx + 1, sent_idx)
                t2_wrt_span = (t2_idx, t2_idx + 1, sent_idx)

                sample = self.feature_extractor.extract_features_from_doc(
                    doc, sent.start_token, sent.end_token, t1_wrt_span, t2_wrt_span)
                sample["indices"] = [t1_idx - sent.start_token, t2_idx - sent.start_token]

                if include_labels:
                    sample["labels"] = self.labels_extraction_strategy.extract(doc, sent, t1_idx, t2_idx)

                samples.append(sample)
                token_pairs.append((t1_idx, t2_idx))

        return samples, token_pairs

    def get_labels_size(self):
        return self.labels_extraction_strategy.get_labels_num()

    def get_padding_value_and_rank(self, name):
        if name == "labels":
            return self.labels_extraction_strategy.get_padding_value_and_rank()
        if name == 'indices':
            return 0, 1
        return self.feature_extractor.get_padding_value_and_rank(name)

    def save(self, out_path):
        save_with_pickle(self, out_path, "feature_extractor.pkl")

    @staticmethod
    def load(path):
        return load_with_pickle(path, "feature_extractor.pkl")
