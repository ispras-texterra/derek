from typing import Iterable

from derek.common.feature_extraction.factory import generate_token_feature_extractor
from derek.common.feature_extraction.token_position_feature_extractor import generate_token_position_feature_extractor
from derek.common.helper import namespaced, from_namespace
from derek.data.model import Document


def generate_spans_common_feature_extractor(docs: Iterable[Document], props: dict, char_padding_size: int = 0):
    token_common_fe, token_common_meta = generate_token_feature_extractor(docs, props, char_padding_size)
    token_position_fe, token_position_meta = generate_token_position_feature_extractor(props)
    token_position_meta = token_position_meta.namespaced('span1') + token_position_meta.namespaced('span2')
    token_common_meta.basic_meta += token_position_meta

    return SpansCommonFeatureExtractor(token_common_fe, token_position_fe), token_common_meta


class SpansCommonFeatureExtractor:
    def __init__(self, token_common_feature_extractor, token_position_feature_extractor):
        self.common_fe = token_common_feature_extractor
        self.position_fe = token_position_feature_extractor

    # wrt_span is (token_start, token_end, sent_idx)
    def extract_features_from_doc(self, doc: Document, start_token, end_token, wrt_span_1, wrt_span_2):
        return {
            **self.common_fe.extract_features_from_doc(doc, start_token, end_token),
            **namespaced(self.position_fe.extract_features_from_doc(doc, start_token, end_token, wrt_span_1), "span1"),
            **namespaced(self.position_fe.extract_features_from_doc(doc, start_token, end_token, wrt_span_2), "span2")
        }

    def get_padding_value_and_rank(self, name):
        for namespace in ("span1", "span2"):
            value = self.position_fe.get_padding_value_and_rank(from_namespace(name, namespace))
            if value is not None:
                return value

        return self.common_fe.get_padding_value_and_rank(name)
