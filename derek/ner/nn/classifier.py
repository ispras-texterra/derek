from typing import Iterable, List, Callable

import tensorflow as tf
from logging import getLogger

from derek.common.feature_computation.manager import SyntacticFeatureComputer
from derek.common.feature_extraction.factory import DEFAULT_FEATS_LIST
from derek.common.io import save_with_pickle, load_with_pickle
from derek.common.nn.tf_io import save_classifier, load_classifier
from derek.common.nn.tf_utils import TFSessionAwareClassifier, TFSessionAwareTrainer
from derek.common.nn.trainers import predict_for_samples, \
    train_for_samples, get_char_padding_size, TaskTrainMeta, get_decayed_lr, get_const_controller
from derek.common.nn.batchers import get_standard_batcher_factory, get_batcher_from_props
from derek.data.model import Document, Entity
from derek.common.nn.graph_factory import build_graphs_with_shared_encoder
from derek.ner.nn.graph_factory import build_task_graph_meta
from derek.ner.feature_extraction.feature_extractor import generate_feature_extractor, NERFeatureExtractor
from derek.common.helper import FuncIterable
from derek.ner.post_processing import unify_types_of_similar_entities

logger = getLogger('logger')


class _Classifier:
    # TODO make prediction batch size configurable
    _PREDICTION_BATCH_SIZE = 400

    def __init__(self, graph, feature_extractor: NERFeatureExtractor, feature_computer, session, saver, post_processor):
        self.graph = graph
        self.extractor = feature_extractor
        self.feature_computer = feature_computer
        self.session = session
        self.saver = saver
        self.post_processor = post_processor

    def predict_doc(self, doc: Document) -> List[Entity]:
        doc = self.feature_computer.create_features_for_doc(doc)
        sent_samples = self.extractor.extract_features_from_doc(doc)

        batcher = get_standard_batcher_factory(
            sent_samples, self._PREDICTION_BATCH_SIZE, self.extractor.get_padding_value_and_rank)

        # we have only predictions as output
        sent_labels, = predict_for_samples(self.graph, self.session, ["predictions"], batcher)

        predicted = []

        for sent, labels in zip(doc.sentences, sent_labels):
            # remove padding
            predicted.extend(self.extractor.encoded_labels_to_entities(sent, labels[:len(sent)]))

        if self.post_processor is not None:
            predicted = self.post_processor(doc, predicted)

        return predicted

    def save(self, out_path):
        save_classifier(out_path, self.extractor, self.feature_computer, self.graph, self.session, self.saver)
        save_with_pickle(self.post_processor, out_path, "post_processor")

    @classmethod
    def load(cls, path, session):
        extractor, feature_computer, graph, saver = load_classifier(
            path, NERFeatureExtractor, SyntacticFeatureComputer, session)

        return cls(graph, extractor, feature_computer, session, saver, load_with_pickle(path, "post_processor"))


class NERClassifier(TFSessionAwareClassifier):
    def __init__(self, model_path: str):
        super().__init__()
        self.path = model_path

    def _load(self):
        return _Classifier.load(self.path, self._session)


class NERTrainer(TFSessionAwareTrainer):
    def __init__(self, props: dict):
        super().__init__(props)
        self.props = props

    def train(
            self, docs: Iterable[Document], unlabeled_docs: Iterable[Document] = None,
            early_stopping_callback: Callable[[NERClassifier, int], bool] = lambda c, e: False):

        feature_computer = SyntacticFeatureComputer(self.props.get('morph_feats_list', DEFAULT_FEATS_LIST))
        precomputed_docs = FuncIterable(lambda: map(feature_computer.create_features_for_doc, docs))

        char_padding_size = get_char_padding_size(self.props)
        feature_extractor, meta = generate_feature_extractor(precomputed_docs, self.props, char_padding_size)
        # we have only one graph
        graph, = build_graphs_with_shared_encoder(
            self.props, meta, [build_task_graph_meta(self.props, feature_extractor.get_labels_size())])

        init = tf.global_variables_initializer()
        self._session.run(init)

        samples = feature_extractor.extract_features_from_docs(precomputed_docs)
        saver = tf.train.Saver(save_relative_paths=True)

        if self.props.get("unify_similar_entities_types", False):
            processor = unify_types_of_similar_entities
        else:
            processor = None

        classifier = _Classifier(graph, feature_extractor, feature_computer, self._session, saver, processor)

        batcher_factory = get_batcher_from_props(
            samples, self.props["batcher"], feature_extractor.get_padding_value_and_rank, True, True)

        train_meta = TaskTrainMeta(
            "NER", graph, batcher_factory,
            {
                "learning_rate": get_decayed_lr(self.props["learning_rate"], self.props.get("lr_decay", 0)),
                "dropout_rate": get_const_controller(self.props.get("dropout", 1.0))
            },
            classifier, early_stopping_callback)

        train_for_samples(self._session, self.props["epoch"], [train_meta])
