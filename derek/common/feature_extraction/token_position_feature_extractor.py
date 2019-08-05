from derek.common.feature_extraction.converters import create_signed_integers_converter, \
    create_signed_log_integers_converter, create_categorical_converter, create_unsigned_integers_converter
from derek.common.feature_extraction.factory_helper import get_categorical_meta_converters
from derek.common.feature_extraction.helper import encode_sequence
from derek.data.helper import get_marked_tokens_on_root_path_for_span
from derek.data.model import Document, TokenSpan


def generate_token_position_feature_extractor(props):
    meta, converters = get_categorical_meta_converters(_get_features(props))
    return TokenPositionFeatureExtractor(converters), meta


class TokenPositionFeatureExtractor:
    def __init__(self, converters):
        self.converters = converters

    # wrt_span is (token_start, token_end, sent_idx)
    def extract_features_from_doc(self, doc: Document, start_token, end_token, wrt_span):
        features = {}

        factories = [
            ("token_position", _get_token_positions_to_span),
            ("token_log_position", _get_token_positions_to_span),
            ("sent_position", _get_sentence_positions_to_span),
            ("at_root_dt_path", _get_tokens_on_path_from_root_to_span),
            ("root_dt_path_position", lambda d, f, l, s: _get_tokens_on_path_from_root_to_span(d, f, l, s, True))
        ]

        for feature_name, feature_factory in factories:
            converter = self.converters.get(feature_name, None)
            if converter is not None:
                features[feature_name] = encode_sequence(
                    feature_factory(doc, start_token, end_token, wrt_span), converter)
        return features

    def get_padding_value_and_rank(self, name):
        if name in self.converters:
            return self.converters[name]['$PADDING$'], 1


def _get_sentence_positions_to_span(doc: Document, start_token: int, end_token: int, wrt_span):
    _, _, sent_idx = wrt_span
    return [sent_idx - doc.get_token_sent_idx(idx) for idx in range(start_token, end_token)]


def _get_token_positions_to_span(doc: Document, start_token: int, end_token: int, wrt_span):
    start, end, _ = wrt_span
    token_span = TokenSpan(start, end)
    return [token_span.token_distance_to(TokenSpan(idx, idx + 1)) for idx in range(start_token, end_token)]


def _get_tokens_on_path_from_root_to_span(
        doc: Document, start_token: int, end_token: int, wrt_span, add_distance=False):
    return get_marked_tokens_on_root_path_for_span(doc, wrt_span, add_distance=add_distance)[start_token:end_token]


def _get_features(props):
    features = {}

    size = props.get("token_position_size", -1)
    if size >= 0:
        feature = {
            "converter": create_signed_integers_converter(props["max_word_distance"])
        }
        if size > 0:
            feature["embedding_size"] = size
        features["token_position"] = feature

    size = props.get("token_log_position_size", -1)
    if size >= 0:
        feature = {
            "converter": create_signed_log_integers_converter(props["max_word_distance"])
        }
        if size > 0:
            feature["embedding_size"] = size
        features["token_log_position"] = feature

    size = props.get("sent_position_size", -1)
    if size >= 0:
        feature = {
            "converter": create_signed_integers_converter(props["max_sent_distance"])
        }
        if size > 0:
            feature["embedding_size"] = size
        features["sent_position"] = feature

    size = props.get("at_root_dt_path_size", -1)
    if size >= 0:
        feature = {
            "converter": create_categorical_converter({False, True})
        }
        if size > 0:
            feature["embedding_size"] = size
        features["at_root_dt_path"] = feature

    size = props.get("root_dt_path_position_size", -1)
    if size >= 0:
        feature = {
            "converter": create_unsigned_integers_converter(props["max_dt_depth"], additional_labels={False})
        }
        if size > 0:
            feature["embedding_size"] = size
        features["root_dt_path_position"] = feature

    return features
