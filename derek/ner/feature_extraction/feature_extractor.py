from itertools import chain
from typing import Iterable, List

from derek.common.feature_extraction.converters import create_categorical_converter
from derek.common.feature_extraction.factory import generate_token_feature_extractor, TokenFeatureExtractor
from derek.common.feature_extraction.factory_helper import collect_entities_types
from derek.common.feature_extraction.ne_feature_extractor import generate_ne_feature_extractor, NEFeatureExtractor
from derek.common.helper import FuncIterable
from derek.common.io import load_with_pickle, save_with_pickle
from derek.data.model import Document, Sentence, Entity
from derek.common.feature_extraction.helper import encode_sequence
from derek.data.transformers import DocumentTransformer
from derek.ner.feature_extraction.augmentations import EntitiesUnquoteAugmentor
from derek.ner.feature_extraction.labelling_strategies import get_labelling_strategy


def generate_feature_extractor(docs: Iterable[Document], props: dict, char_padding_size: int = 0):
    types_to_unquote = props.get("types_to_unquote", [])
    unquote_prob = props.get("prob_to_unquote", 0.0)

    if types_to_unquote and unquote_prob:
        # concat augmented docs with original ones to be sure all possible features are processed by FE factories
        augmentor = EntitiesUnquoteAugmentor(1.0, types_to_unquote)
        prev_docs = docs
        docs = FuncIterable(lambda: chain(prev_docs, map(augmentor.transform, prev_docs)))

    token_feature_extractor, token_features_meta = generate_token_feature_extractor(docs, props, char_padding_size)

    ne_feature_extractor, ne_meta = generate_ne_feature_extractor(docs, props)
    token_features_meta.basic_meta += ne_meta

    ent_types = collect_entities_types(docs)

    labelling_strategy = get_labelling_strategy(props.get("labelling_strategy", "BIO"))
    labels_converter = create_categorical_converter(
        labelling_strategy.get_possible_categories(ent_types),
        zero_padding=False
    )
    prob_augmentor = EntitiesUnquoteAugmentor(unquote_prob, types_to_unquote)
    feature_extractor = NERFeatureExtractor(
        token_feature_extractor, ne_feature_extractor, labelling_strategy, labels_converter, prob_augmentor)

    return feature_extractor, token_features_meta


class NERFeatureExtractor:
    def __init__(
            self, token_feature_extractor: TokenFeatureExtractor, ne_feature_extractor: NEFeatureExtractor,
            labelling_strategy, labels_converter, augmentor: DocumentTransformer):
        self.token_feature_extractor = token_feature_extractor
        self.ne_feature_extractor = ne_feature_extractor
        self.labelling_strategy = labelling_strategy
        self.labels_converter = labels_converter
        self.reversed_labels_converter = labels_converter.get_reversed_converter()
        self.augmentor = augmentor

    def extract_features_from_docs(self, docs) -> Iterable:
        return FuncIterable(
            lambda: chain.from_iterable(
                map(lambda d: self.extract_features_from_doc(self.augmentor.transform(d), True), docs)))

    def extract_features_from_doc(self, doc: Document, include_labels=False):
        samples = []

        for sent_idx, sent in enumerate(doc.sentences):
            sample = self.token_feature_extractor.extract_features_from_doc(doc, sent.start_token, sent.end_token)
            sample.update(self.ne_feature_extractor.extract_features_from_doc(doc, sent.start_token, sent.end_token))
            if include_labels:
                labels = self.labelling_strategy.encode_labels(sent, doc.entities.contained_in(sent))
                sample["labels"] = encode_sequence(labels, self.labels_converter)

            samples.append(sample)

        return samples

    def get_labels_size(self):
        return len(self.labels_converter)

    def encoded_labels_to_entities(self, sent: Sentence, encoded_labels: List[int]) -> List[Entity]:
        return self.labelling_strategy.decode_labels(
            sent, encode_sequence(encoded_labels, self.reversed_labels_converter))

    def get_padding_value_and_rank(self, name):
        if name == "labels":
            # take outside entity label as padding value
            outside_ent_label = self.labelling_strategy.outside_ent_label
            return self.labels_converter[outside_ent_label], 1

        ne_padding = self.ne_feature_extractor.get_padding_value_and_rank(name)
        if ne_padding is not None:
            return ne_padding

        return self.token_feature_extractor.get_padding_value_and_rank(name)

    def save(self, out_path):
        save_with_pickle(self, out_path, "feature_extractor.pkl")

    @staticmethod
    def load(path):
        return load_with_pickle(path, "feature_extractor.pkl")
