from derek.common.feature_extraction.helper import encode_sequence, encode_sequence3d
from derek.data.model import Document


class TokenFeatureExtractor:
    def __init__(self, we_converters_preprocessors, word_level_features, char_level_features, vectors_keys):

        self.we_converters_preprocessors = we_converters_preprocessors
        self.word_level_features = word_level_features
        self.char_level_features = char_level_features
        self.vectors_keys = vectors_keys

    def extract_features_from_doc(self, doc: Document, start_token_idx, end_token_idx):
        ret = {
            "seq_len": end_token_idx - start_token_idx
        }

        for name, (converter, preprocessor) in self.we_converters_preprocessors.items():
            tokens = doc.tokens[start_token_idx:end_token_idx]
            tokens = [preprocessor(token) if preprocessor is not None else token for token in tokens]
            ret[name] = encode_sequence(tokens, converter)

        for key in self.vectors_keys:
            ret[key] = doc.token_features[key][start_token_idx:end_token_idx]

        for name, converter in self.word_level_features.items():
            ret[name] = encode_sequence(doc.token_features[name][start_token_idx:end_token_idx], converter)

        if 'chars' in self.char_level_features:
            chars = TokenFeatureExtractor._get_chars_features(
                doc, start_token_idx, end_token_idx, self.char_level_features['chars']['padding_size'])
            ret['chars'] = encode_sequence3d(chars, self.char_level_features['chars']['converter'])

        return ret

    def get_padding_value_and_rank(self, name):
        if name == "seq_len":
            return 0, 0

        if name in self.we_converters_preprocessors:
            return self.we_converters_preprocessors[name][0]['$PADDING$'], 1

        if name in self.vectors_keys:
            # padding must have the same type as values
            return 0.0, 2

        if name in self.word_level_features:
            return self.word_level_features[name]['$PADDING$'], 1

        if name in self.char_level_features:
            return self.char_level_features[name]['converter']['$PADDING$'], 2

    @staticmethod
    def _get_chars_features(doc: Document, start_token_idx: int, end_token_idx: int, padding_size: int):
        chars = []
        for token in doc.tokens[start_token_idx: end_token_idx]:
            chars.append(TokenFeatureExtractor._add_feature_padding(token, padding_size))
        return chars

    @staticmethod
    def _add_feature_padding(val: list, size: int):
        padding = ['$PADDING$'] * size
        return padding + list(val) + padding
