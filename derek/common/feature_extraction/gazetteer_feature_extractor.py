from os.path import isfile
from typing import Set

from derek.common.feature_extraction.converters import create_categorical_converter
from derek.common.feature_extraction.factory_helper import get_categorical_meta_converters
from derek.common.feature_extraction.helper import encode_sequence
from derek.data.model import Document
from derek.data.processing_helper import StandardTokenProcessor


def generate_gazetteers_feature_extractors(props: dict):
    features = {}
    gazetteer_feature_extractors = {}
    converter = create_categorical_converter({True, False}, zero_padding=True, has_oov=False)

    for index, config in enumerate(props.get('gazetteers', [])):
        gazetteer_name = f"gazetteer_{index}"
        features[gazetteer_name] = {'converter': converter}
        if config.get('emb_size', -1) > 0:
            features[gazetteer_name]['embedding_size'] = config['emb_size']

        gazetteer = _read_gazetteer(config["path"])
        gazetteer_feature_extractors[gazetteer_name] = GazetteerFeatureExtractor(
            gazetteer, StandardTokenProcessor.from_props(config), converter, config.get("lemmatize", False))

    meta, _ = get_categorical_meta_converters(features)
    return meta, gazetteer_feature_extractors


def _read_gazetteer(path) -> set:
    assert isfile(path), f"Dictionary path '{path}' is not a real path"
    with open(path, 'r', encoding='utf-8')as f:
        token_set = set(f.read().strip().split("\n"))
    return token_set


class GazetteerFeatureExtractor:
    # TODO: include lemmatization in StandardTokenProcessor
    def __init__(self, gazetteer: Set[str], token_processor: StandardTokenProcessor, converter: dict, lemmatize: bool):
        self._gazetteer = gazetteer
        self._processor = token_processor
        self._converter = converter
        self._lemmatize = lemmatize

    def extract_features(self, doc: Document, start_token_idx: int, end_token_idx: int):
        tokens = doc.token_features["lemmas"] if self._lemmatize else doc.tokens
        token_slice = map(self._processor, tokens[start_token_idx:end_token_idx])
        return encode_sequence(map(self._gazetteer.__contains__, token_slice), self._converter)

    def get_padding_value_and_rank(self):
        return self._converter["$PADDING$"], 1
