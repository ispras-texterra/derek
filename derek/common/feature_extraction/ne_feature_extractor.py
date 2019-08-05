from typing import Iterable

from derek.common.feature_extraction.converters import create_categorical_converter
from derek.common.feature_extraction.factory_helper import get_categorical_meta_converters, collect_entities_types
from derek.common.feature_extraction.helper import encode_sequence
from derek.data.model import Document, Sentence
from derek.ner.feature_extraction.labelling_strategies import get_labelling_strategy


def generate_ne_feature_extractor(docs: Iterable[Document], props: dict):
    features = {}
    labelling_strategy = None

    if props.get("ne_emb_size", -1) >= 0:
        types = collect_entities_types(docs, extras=True)
        labelling_strategy = get_labelling_strategy(props.get("ne_labelling_strategy", "IO"))
        features['ne'] = {
            'converter': create_categorical_converter(labelling_strategy.get_possible_categories(types), has_oov=True)
        }
        if props["ne_emb_size"] != 0:
            features["ne"]['embedding_size'] = props["ne_emb_size"]

    meta, converters = get_categorical_meta_converters(features)
    return NEFeatureExtractor(converters, labelling_strategy), meta


class NEFeatureExtractor:
    def __init__(self, converters, labelling_strategy):
        self.converters = converters
        self.labelling_strategy = labelling_strategy

    def extract_features_from_doc(self, doc: Document, start_token, end_token):
        features = {}
        if "ne" in self.converters:
            features["ne"] = encode_sequence(self._get_ne_types(doc, start_token, end_token), self.converters["ne"])
        return features

    def get_padding_value_and_rank(self, name):
        if name in self.converters:
            return self.converters[name]["$PADDING$"], 1

    def _get_ne_types(self, doc, start_token, end_token):
        # TODO We create fake sentence because labeller requires sentence as input
        sent = Sentence(start_token, end_token)
        return self.labelling_strategy.encode_labels(sent, doc.extras['ne'].contained_in(sent))
