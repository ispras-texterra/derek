from abc import ABCMeta, abstractmethod
from typing import Iterable, List
from warnings import warn

from derek.common.feature_extraction.helper import Direction
from derek.common.feature_computation.computers import get_dt_depths_feature, get_sentence_borders_feature, \
    get_entities_types_and_depths_features, get_dt_breakups_feature, get_dt_deltas_feature, get_morph_features
from derek.data.model import Document
from derek.common.io import save_with_pickle, load_with_pickle


class AbstractFeatureComputer(metaclass=ABCMeta):
    def create_features_for_docs(self, docs: Iterable[Document]) -> List[Document]:
        return [self.create_features_for_doc(doc) for doc in docs]

    def create_features_for_doc(self, doc: Document) -> Document:
        return doc.with_additional_token_features(self._get_features(doc))

    @abstractmethod
    def _get_features(self, doc: Document) -> dict:
        pass

    def save(self, out_path):
        save_with_pickle(self, out_path, "feature_computer.pkl")

    @staticmethod
    def load(path):
        return load_with_pickle(path, "feature_computer.pkl")


class SyntacticFeatureComputer(AbstractFeatureComputer):
    def __init__(self, morph_features):
        self.morph_features = morph_features

    def _get_features(self, doc: Document) -> dict:
        new_token_features = dict()

        if "dt_head_distances" not in doc.token_features:
            warn("SyntacticFeatureComputer was called on doc without dependency tree, some features won't be included")
        else:
            new_token_features["dt_depths"] = get_dt_depths_feature(doc)

            for direction in [Direction.FORWARD, Direction.BACKWARD]:
                new_token_features["dt_breakups_" + direction.value] = get_dt_breakups_feature(doc, direction=direction)
                new_token_features["dt_deltas_" + direction.value] = get_dt_deltas_feature(
                    doc, direction=direction, precomputed_depths=new_token_features["dt_depths"])

        new_token_features["borders"] = get_sentence_borders_feature(doc)
        new_token_features.update(get_morph_features(doc, self.morph_features))

        return new_token_features


class EntityBasedFeatureComputer(AbstractFeatureComputer):
    def _get_features(self, doc: Document) -> dict:
        ent_types, ent_depths = get_entities_types_and_depths_features(doc)
        return {"entities_types": ent_types, "entities_depths": ent_depths}


class CompositeFeatureComputer(AbstractFeatureComputer):
    def __init__(self, feature_computers: Iterable[AbstractFeatureComputer]):
        self.__fcs = tuple(feature_computers)

    def _get_features(self, doc: Document) -> dict:
        return dict(item for fc in self.__fcs for item in fc._get_features(doc).items())


class FeatureComputer(CompositeFeatureComputer):
    def __init__(self, morph_features):
        super().__init__((SyntacticFeatureComputer(morph_features), EntityBasedFeatureComputer()))
