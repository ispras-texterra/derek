from itertools import chain
from logging import getLogger
from typing import Iterable, List, Callable, Tuple

import tensorflow as tf

from derek.common.feature_computation.manager import SyntacticFeatureComputer
from derek.common.feature_extraction.factory import DEFAULT_FEATS_LIST
from derek.common.helper import FuncIterable
from derek.common.io import save_with_pickle, load_with_pickle
from derek.common.nn.batchers import get_standard_batcher_factory, get_batcher_from_props
from derek.common.nn.graph_factory import build_graphs_with_shared_encoder
from derek.common.nn.tf_io import save_classifier, load_classifier
from derek.common.nn.tf_utils import TFSessionAwareClassifier, TFSessionAwareTrainer
from derek.common.nn.trainers import predict_for_samples, \
    train_for_samples, get_char_padding_size, TaskTrainMeta, get_decayed_lr, get_const_controller
from derek.data.model import Document, Entity
from derek.ner.feature_extraction.feature_extractor import generate_feature_extractor, NERFeatureExtractor
from derek.ner.nn.graph_factory import build_task_graph_meta
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

    def predict_docs_with_scores(self, docs: List[Document]) -> Tuple[List[List[Entity]], List[List[float]]]:
        docs = self.feature_computer.create_features_for_docs(docs)
        samples = chain.from_iterable(map(self.extractor.extract_features_from_doc, docs))
        batcher = get_standard_batcher_factory(
            samples, self._PREDICTION_BATCH_SIZE, self.extractor.get_padding_value_and_rank)

        sent_labels, scores = predict_for_samples(self.graph, self.session, ["predictions", "scores"], batcher)
        sent_labels, scores = iter(sent_labels), iter(scores)

        docs_predicted_entities, docs_confidences = [], []

        for doc in docs:
            predicted_entities, sentences_confidences = [], []

            for sent, labels, score in zip(doc.sentences, sent_labels, scores):
                # remove padding
                predicted_entities.extend(self.extractor.encoded_labels_to_entities(sent, labels[:len(sent)]))
                sentences_confidences.append(score)

            if self.post_processor is not None:
                predicted_entities = self.post_processor(doc, predicted_entities)

            docs_predicted_entities.append(predicted_entities)
            docs_confidences.append(sentences_confidences)

        return docs_predicted_entities, docs_confidences

    def predict_docs(self, docs: List[Document]) -> List[List[Entity]]:
        return self.predict_docs_with_scores(docs)[0]

    def predict_doc_with_scores(self, doc: Document) -> Tuple[List[Entity], List[float]]:
        """
        :return: (List[Entity] found in doc, List[float] containing sequence labelling score for each Sentence in doc)
        """
        docs_entities, docs_confidences = self.predict_docs_with_scores([doc])
        return docs_entities[0], docs_confidences[0]

    def predict_doc(self, doc: Document) -> List[Entity]:
        predicted, _ = self.predict_doc_with_scores(doc)
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
