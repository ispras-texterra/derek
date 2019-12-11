import os
from collections import namedtuple
from logging import getLogger
from typing import List, Callable

import tensorflow as tf

from derek.common.feature_computation.manager import FeatureComputer
from derek.common.feature_extraction.factory import DEFAULT_FEATS_LIST, generate_token_feature_extractor
from derek.common.helper import FuncIterable
from derek.common.io import CacheMPManager, save_with_pickle, load_with_pickle
from derek.common.logger import update_progress
from derek.common.nn.batchers import get_standard_batcher_factory
from derek.common.nn.tf_io import save_classifier, load_classifier
from derek.common.nn.tf_utils import TFSessionAwareClassifier, TFSessionAwareTrainer
from derek.common.nn.trainers import predict_for_samples, train_for_samples, \
    get_char_padding_size, TaskTrainMeta, get_decayed_lr, get_const_controller
from derek.coref.feature_extraction.factory import create_coref_feature_extractor
from derek.coref.feature_extraction.feature_extractor import CorefFeatureExtractor
from derek.coref.feature_extraction.heuristics import CLASSIFIERS, ComplexClassifier
from derek.coref.nn.graph_factory import build_coref_graph
from derek.data.model import Relation, Document

logger = getLogger("logger")


def get_coref_batcher_factory(samples, size, feature_extractor, need_printing, need_shuffling, buffer_size=1000):
    return get_standard_batcher_factory(
        samples, size, feature_extractor.get_padding_value_and_rank,
        print_progress=need_printing, need_shuffling=need_shuffling, buffer_size=buffer_size)


class _Classifier:
    def __init__(self, graph: dict, extractor: CorefFeatureExtractor, feature_computer: FeatureComputer,
                 session, saver, classifiers=None):
        self.session = session
        self.graph = graph
        self.extractor = extractor
        self.feature_computer = feature_computer
        self.saver = saver
        self.classifiers = classifiers

    def predict_docs(self, docs: list, *, print_progress=False, include_probs=False) -> dict:
        rels = {}
        for i, doc in enumerate(docs):
            result = self.predict_doc(doc, include_probs)

            rels[doc.name] = result
            if print_progress:
                update_progress((i + 1) / len(docs))

        return rels

    def predict_doc(self, doc, include_probs=False):
        doc = self.feature_computer.create_features_for_doc(doc)

        # parallel lists for segment features and segment entity pairs for all doc segments
        samples, entity_pairs = self.extractor.extract_features_from_doc(doc, use_filter=True)
        entity_pairs = sum(entity_pairs, [])
        outputs = ["predictions"]

        if include_probs:
            outputs.append("scores")

        out = predict_for_samples(
            self.graph, self.session, outputs,
            get_coref_batcher_factory(samples, 300, self.extractor, False, False))  # labels, [scores]

        relations = self._collect_pair_results(out[0], entity_pairs)
        relations = self._get_relations(relations)

        posprocessing_result = self.classifiers.apply(doc)

        posprocessing_rels = set()
        for (e1, e2), scores in posprocessing_result.items():
            label = max(scores, key=scores.get)
            if label is not None:
                posprocessing_rels.add(Relation(e1, e2, label))

        relations |= posprocessing_rels

        try:
            relations |= doc.relations
        except ValueError:
            pass

        ret = relations

        if include_probs:
            scores = self._collect_pair_results(out[1], entity_pairs)
            scores = self._get_scores(scores)
            scores = {**scores, **posprocessing_result}
            try:
                scores = {**scores, **get_known_rel_scores(doc.relations)}
            except ValueError:
                pass
            ret = (relations, scores)

        return ret

    def _collect_pair_results(self, out, entity_pairs):
        ret = {}
        for result, entity_pair in zip(out, entity_pairs):
            ret[entity_pair] = result
        return ret

    def _get_scores(self, scores: dict) -> dict:
        ret = {}
        for pair, pair_scores in scores.items():
            scores_dict = {}
            for i, score in enumerate(pair_scores):
                scores_dict[self.extractor.get_type(i)] = score
            ret[pair] = scores_dict
        return ret

    def _get_relations(self, predictions: dict) -> set:
        rels = set()
        for (e1, e2), label in predictions.items():
            rel_type = self.extractor.get_type(label)
            if rel_type is not None:
                rels.add(Relation(e1, e2, rel_type))
        return rels

    def save(self, out_path):
        save_classifier(out_path, self.extractor, self.feature_computer, self.graph, self.session, self.saver)
        save_with_pickle(self.classifiers, out_path, "classifiers.pkl")

    @classmethod
    def load(cls, path, session):
        extractor, feature_computer, graph, saver = load_classifier(
            path, CorefFeatureExtractor, FeatureComputer, session)
        classifiers = load_with_pickle(path, "classifiers.pkl")
        return cls(graph, extractor, feature_computer, session, saver, classifiers)


class CorefClassifier(TFSessionAwareClassifier):
    def __init__(self, model_path: str):
        super().__init__()
        self.path = model_path

    def _load(self):
        return _Classifier.load(self.path, self._session)


def get_known_rel_scores(rels):
    ret = {}
    for rel in rels:
        ret[(rel.first_entity, rel.second_entity)] = {None: 0, "COREF": 1}
    return ret


class CorefTrainer(TFSessionAwareTrainer):
    _TaskMeta = namedtuple("_TaskMeta", ["task_name", "feature_extractor", "props", "metas"])

    def __init__(self, props: dict):
        super().__init__(props)
        self.props = props
        self._feature_computer = FeatureComputer(self.props.get('morph_feats_list', DEFAULT_FEATS_LIST))

    def train(
            self, docs: List[Document],
            early_stopping_callback: Callable[[CorefClassifier, int], bool] = lambda c, e: False):

        def log_meta(meta, msg):
            logger.info("{}\n{}".format(msg, meta))

        docs = self._get_precomputed_docs(docs)

        common_fe, common_meta = self._create_common_fe(docs)
        meta = self._init_meta(common_fe, docs)

        log_meta(common_meta, "common features")

        log_meta(meta.metas.encoder, "{} task specific encoder features".format(meta.task_name))
        log_meta(meta.metas.attention, "{} attention features".format(meta.task_name))
        log_meta(meta.metas.classifier, "{} classifier features".format(meta.task_name))

        print("Extracting features")

        self._get_samples(
            docs, meta, "SAMPLES_CACHE_PATH",
            lambda s: self._build_and_train(common_meta, meta, s, early_stopping_callback))

    def _get_precomputed_docs(self, docs):
        return FuncIterable(lambda: map(self._feature_computer.create_features_for_doc, docs))

    def _create_common_fe(self, docs):

        char_padding_size = get_char_padding_size(self.props)

        return generate_token_feature_extractor(docs, self.props, char_padding_size)

    def _init_meta(self, common_fe, docs):
        feature_extractor, metas = create_coref_feature_extractor(docs, self.props, common_fe)
        return self._TaskMeta("coref", feature_extractor, self.props, metas)

    @staticmethod
    def _get_samples(docs, meta, samples_cache_path, samples_strategy):
        samples = meta.feature_extractor.extract_features_from_docs_iterator(
            docs, meta.props.get('use_filter', False), meta.props.get('drop_negative', 0))

        if samples_cache_path in os.environ:
            with CacheMPManager(iter(samples), os.environ[samples_cache_path]) as samples:
                samples_strategy(samples)
        else:
            samples_strategy(samples)

    def _build_and_train(self, common_meta, coref_meta, samples, early_stopping_callback):
        graph = build_coref_graph(self.props, common_meta, [coref_meta])

        init = tf.global_variables_initializer()
        self._session.run(init)

        self._train_regular(coref_meta, graph, samples, early_stopping_callback)

    def _train_regular(self, meta, graph, samples, early_stopping_callback):
        epoch = self.props["epoch"]

        print("Training for {} epochs".format(epoch))

        train_meta = self._init_train_meta(meta, graph, samples, early_stopping_callback)

        train_metas = [train_meta]
        train_for_samples(self._session, epoch, train_metas)

    def _init_train_meta(self, meta, graph, samples, early_stopping_callback):
        feature_extractor = meta.feature_extractor

        saver = tf.train.Saver(save_relative_paths=True, max_to_keep=100)

        classifiers = []
        for name in meta.props.get('classifiers', []):
            classifiers.append(CLASSIFIERS[name])

        classifier = _Classifier(
            graph, feature_extractor, self._feature_computer, self._session, saver, ComplexClassifier(classifiers))

        batcher_factory = get_coref_batcher_factory(
            samples, meta.props["batch_size"], feature_extractor, True, True, meta.props.get("buffer_size", 1000))

        return _init_train_meta(meta, graph, batcher_factory, classifier, early_stopping_callback)


def _init_train_meta(meta, graph, batcher_factory, classifier, early_stopping_callback):
    controllers = {
        "learning_rate": get_decayed_lr(meta.props["learning_rate"], meta.props.get("lr_decay", 0)),
        "dropout_rate": get_const_controller(meta.props.get("dropout", 1.0)),
    }

    return TaskTrainMeta(meta.task_name, graph, batcher_factory, controllers, classifier, early_stopping_callback)
