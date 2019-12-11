from itertools import chain
from logging import getLogger
from typing import Iterable, List, Callable
from collections import namedtuple
from warnings import warn

import tensorflow as tf

from derek.common.feature_computation.manager import SyntacticFeatureComputer, EntityBasedFeatureComputer, \
    CompositeFeatureComputer, AbstractFeatureComputer
from derek.common.feature_extraction.factory import DEFAULT_FEATS_LIST
from derek.common.helper import FuncIterable
from derek.common.io import save_with_pickle, load_with_pickle
from derek.common.logger import update_progress
from derek.common.nn.graph_factory import build_graphs_with_shared_encoder
from derek.common.nn.tf_io import save_classifier, load_classifier
from derek.common.nn.tf_utils import TFSessionAwareClassifier, TFSessionAwareTrainer
from derek.common.nn.trainers import predict_for_samples, train_for_samples, \
    get_char_padding_size, TaskTrainMeta, multitask_scheduler, get_decayed_lr, get_const_controller, \
    get_epoch_shifting_wrapper
from derek.common.nn.batchers import get_bucketed_batcher_factory, get_standard_batcher_factory
from derek.data.model import Relation, Document
from derek.data.entities_collapser import EntitiesCollapser
from derek.rel_ext.feature_extraction.spans_feature_extractor import generate_spans_common_feature_extractor
from derek.rel_ext.feature_extraction.factory import generate_feature_extractor
from derek.rel_ext.feature_extraction.feature_extractor import RelExtFeatureExtractor
from derek.rel_ext.nn.graph_factory import build_task_graph_meta, RelExtTaskGraphMeta
from derek.rel_ext.syntactic.parser.feature_extraction.factory import generate_feature_extractor as parser_fe_factory
from derek.rel_ext.syntactic.shortest_path.feature_extraction.feature_extractor import \
    generate_sdp_task_feature_extractor
from derek.rel_ext.syntactic.shortest_path.nn.graph_factory import build_task_graph_meta as build_sdp_task_graph_meta, \
    SDPMeta

logger = getLogger("logger")


def get_bucketed_batcher(samples, size, feature_extractor, bucket_samples_len, need_printing, need_shuffling, buffer_size=10000):
    return get_bucketed_batcher_factory(
        samples, size, feature_extractor.get_padding_value_and_rank, lambda s: s['seq_len'] // bucket_samples_len,
        print_progress=need_printing, need_shuffling=need_shuffling, buffer_size=buffer_size)


def get_standard_batcher(samples, size, feature_extractor, need_printing, need_shuffling, buffer_size=10000):
    return get_standard_batcher_factory(
        samples, size, feature_extractor.get_padding_value_and_rank,
        print_progress=need_printing, need_shuffling=need_shuffling, buffer_size=buffer_size)


class _Classifier:
    def __init__(self, graph: dict, extractor: RelExtFeatureExtractor, feature_computer: AbstractFeatureComputer,
                 session, saver, collapser: EntitiesCollapser):
        self.session = session
        self.graph = graph
        self.extractor = extractor
        self.feature_computer = feature_computer
        self.saver = saver
        self.collapser = collapser

    def predict_docs(self, docs: list, *, print_progress=False, include_probs=False) -> dict:
        rels = {}

        for i, doc in enumerate(docs):
            rels[doc.name] = self.predict_doc(doc, include_probs)
            if print_progress:
                update_progress((i + 1) / len(docs))

        return rels

    def predict_doc(self, doc, include_probs=False):
        doc, direct_mapping = self.collapser.transform_with_mapping(doc)
        reversed_mapping = {v: k for k, v in direct_mapping.items()}
        doc = self.feature_computer.create_features_for_doc(doc)

        samples, entity_pairs = self.extractor.extract_features_from_doc(doc)
        entity_pairs = [(reversed_mapping[e1], reversed_mapping[e2]) for e1, e2 in entity_pairs]
        outputs = ["predictions"]

        if include_probs:
            outputs.append("scores")

        out = predict_for_samples(
            self.graph, self.session, outputs,
            get_standard_batcher(samples, 300, self.extractor, False, False))  # labels, [scores]

        relations = self._collect_pair_results(out[0], entity_pairs)
        relations = self._get_relations(relations)
        ret = relations

        if include_probs:
            scores = self._collect_pair_results(out[1], entity_pairs)
            scores = self._get_scores(scores)
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
        save_with_pickle(self.collapser, out_path, "collapser")

    @classmethod
    def load(cls, path, session):
        extractor, feature_computer, graph, saver = load_classifier(
            path, RelExtFeatureExtractor, CompositeFeatureComputer, session)

        return cls(graph, extractor, feature_computer, session, saver, load_with_pickle(path, "collapser"))


class RelExtClassifier(TFSessionAwareClassifier):
    def __init__(self, model_path: str):
        super().__init__()
        self.path = model_path

    def _load(self):
        return _Classifier.load(self.path, self._session)


class RelExtTrainer(TFSessionAwareTrainer):
    _TaskMeta = namedtuple("_TaskMeta", ["task_name", "feature_extractor", "props", "taskgraphmeta"])

    def __init__(self, props: dict):
        super().__init__(props)
        self.props = props
        self._syntactic_fc = SyntacticFeatureComputer(self.props.get('morph_feats_list', DEFAULT_FEATS_LIST))
        self._feature_computer = CompositeFeatureComputer((self._syntactic_fc, EntityBasedFeatureComputer()))
        self._collapser = EntitiesCollapser(self.props.get("types_to_collapse", set()))

    def train(
            self, docs: List[Document], unlabeled_docs: Iterable[Document] = None,
            early_stopping_callback: Callable[[RelExtClassifier, int], bool] = lambda c, e: False):

        collapsed_docs = FuncIterable(lambda: map(self._collapser.transform, docs))
        prec_docs = self._get_precomputed_docs(collapsed_docs, self._feature_computer)
        if unlabeled_docs is not None:
            unlabeled_docs = self._get_precomputed_docs(unlabeled_docs, self._syntactic_fc)

        shared_meta, rel_ext_meta, auxiliary_metas = self._init_metas(prec_docs, unlabeled_docs)

        print("Extracting features")
        rel_ext_samples = rel_ext_meta.feature_extractor.extract_features_from_docs(prec_docs)
        auxiliary_samples = \
            [list(task_meta.feature_extractor.extract_features_from_docs(unlabeled_docs))
             for task_meta in auxiliary_metas]

        self._build_and_train(
            shared_meta, rel_ext_meta, rel_ext_samples, auxiliary_metas, auxiliary_samples, early_stopping_callback)

    @staticmethod
    def _get_precomputed_docs(docs, feature_computer):
        return FuncIterable(lambda: map(feature_computer.create_features_for_doc, docs))

    def _init_metas(self, rel_ext_docs, unlabeled_docs):
        char_padding_size = get_char_padding_size(self.props)
        concatted_docs = FuncIterable(lambda: chain(rel_ext_docs, [] if unlabeled_docs is None else unlabeled_docs))

        shared_fe, shared_meta = generate_spans_common_feature_extractor(
            concatted_docs, self.props['shared'], char_padding_size)

        self._log_meta(shared_meta, "shared features")

        rel_ext_task_meta = self._init_rel_ext_task_meta(rel_ext_docs, shared_fe)
        parser_task_meta = self._init_parser_task_meta(unlabeled_docs, shared_fe)
        sdp_task_meta = self._init_sdp_task_meta(unlabeled_docs, shared_fe)

        auxiliary_metas = list(filter(lambda x: x is not None, [parser_task_meta, sdp_task_meta]))
        if not auxiliary_metas and unlabeled_docs is not None:
            warn("Unlabeled docs provided without auxiliary configs")

        return shared_meta, rel_ext_task_meta, auxiliary_metas

    @staticmethod
    def _log_meta(meta, msg):
        logger.info("{}\n{}".format(msg, meta))

    @staticmethod
    def _log_relext_metas(metas, task_name):
        RelExtTrainer._log_meta(metas.encoder, "{} task specific encoder features".format(task_name))
        RelExtTrainer._log_meta(metas.attention, "{} attention features".format(task_name))
        RelExtTrainer._log_meta(metas.classifier, "{} classifier features".format(task_name))

    @staticmethod
    def _to_relext_taskgraph_meta(task_name, props, metas, out_size, needs_labels_mask):
        return build_task_graph_meta(RelExtTaskGraphMeta(task_name, props, metas, out_size, needs_labels_mask))

    def _init_rel_ext_task_meta(self, docs, shared_fe):
        task_name = "rel_ext"
        rel_ext_fe, rel_ext_metas = generate_feature_extractor(docs, self.props, shared_fe)
        self._log_relext_metas(rel_ext_metas, task_name)
        task_graph_meta = self._to_relext_taskgraph_meta(
            task_name, self.props, rel_ext_metas, rel_ext_fe.get_labels_size(), True)
        return self._TaskMeta(task_name, rel_ext_fe, self.props, task_graph_meta)

    def _init_parser_task_meta(self, docs, shared_fe):
        task_name = "parser"
        parser_props = self.props.get(f"{task_name}_config", {})
        if not parser_props:
            return None

        if docs is None:
            raise Exception("No unlabeled docs provided but parser_config is in props")

        parser_fe, parser_metas = parser_fe_factory(docs, parser_props, shared_fe)
        self._log_relext_metas(parser_metas, task_name)
        task_graph_meta = self._to_relext_taskgraph_meta(
            task_name, parser_props, parser_metas, parser_fe.get_labels_size(), False)
        return self._TaskMeta(task_name, parser_fe, parser_props, task_graph_meta)

    def _init_sdp_task_meta(self, docs, shared_fe):
        task_name = "sdp"
        sdp_props = self.props.get(f"{task_name}_config", {})
        if not sdp_props:
            return None

        if docs is None:
            raise Exception("No unlabeled docs provided but sdp_config is in props")

        sdp_fe = generate_sdp_task_feature_extractor(sdp_props, shared_fe)
        return self._TaskMeta(
            task_name, sdp_fe, sdp_props, build_sdp_task_graph_meta(SDPMeta(sdp_props, sdp_fe.get_labels_size())))

    def _build_and_train(
            self, shared_meta, rel_ext_meta, rel_ext_samples, auxiliary_metas, auxiliary_samples,
            early_stopping_callback):

        graphs = self._build_graphs(shared_meta, rel_ext_meta, auxiliary_metas)
        rel_ext_graph = graphs[0]
        auxiliary_graphs = graphs[1:]

        saver = tf.train.Saver(save_relative_paths=True, max_to_keep=100)
        classifier = _Classifier(
            rel_ext_graph, rel_ext_meta.feature_extractor, self._feature_computer, self._session, saver, self._collapser)

        init = tf.global_variables_initializer()
        self._session.run(init)

        pretrained_epochs = []
        for meta, samples, graph in zip(auxiliary_metas, auxiliary_samples, auxiliary_graphs):
            pretrained_epochs.append(self._pretrain_auxiliary(meta, graph, samples))

        self._warmup_relext(rel_ext_meta, rel_ext_graph, rel_ext_samples, classifier, early_stopping_callback)
        self._train_regular(rel_ext_meta, rel_ext_graph, rel_ext_samples, classifier,
                            auxiliary_metas, auxiliary_graphs, auxiliary_samples, pretrained_epochs,
                            early_stopping_callback)

    def _build_graphs(self, shared_meta, rel_ext_meta, auxiliary_metas):
        graph_metas = [rel_ext_meta.taskgraphmeta] + [m.taskgraphmeta for m in auxiliary_metas]
        return build_graphs_with_shared_encoder(self.props, shared_meta, graph_metas)

    def _pretrain_auxiliary(self, meta, graph, samples):
        epoch = self.props.get(f"{meta.task_name}_pretraining_epoch", 0)
        if epoch > 0:
            print(f"{meta.task_name} pretraining for {epoch} epochs")
            train_meta = _init_train_meta(meta, graph, samples, False)
            train_for_samples(self._session, epoch, [train_meta])

        return epoch

    def _warmup_relext(self, meta, graph, samples, classifier, early_stopping_callback):
        epoch = self.props.get("freeze_shared_ce_epoch", 0)
        if epoch <= 0:
            return

        print("Rel-ext warm up for {} epochs".format(epoch))

        train_meta = _init_train_meta(meta, graph, samples, True, classifier, early_stopping_callback)
        train_for_samples(self._session, epoch, [train_meta])

    def _train_regular(self, rel_ext_meta, rel_ext_graph, rel_ext_samples, classifier,
                       auxiliary_metas, auxiliary_graphs, auxiliary_samples, pretrained_epochs,
                       early_stopping_callback):

        freeze_epoch = self.props.get("freeze_shared_ce_epoch", 0)
        epoch = self.props["epoch"] - freeze_epoch

        print("Training for {} epochs".format(epoch))

        rel_ext_train_meta = _init_train_meta(
            rel_ext_meta, rel_ext_graph, rel_ext_samples, False, classifier, early_stopping_callback, freeze_epoch)

        train_metas = [rel_ext_train_meta]
        # 1 sample of rel-ext per train step
        schedule = [1]

        for meta, graph, samples, epoch_shift in zip(
                auxiliary_metas, auxiliary_graphs, auxiliary_samples, pretrained_epochs):

            ratio = self.props.get(f"{meta.task_name}_samples_ratio", 0)
            if ratio <= 0:
                continue
            print(f"{meta.task_name} multitask with ratio {ratio}")

            train_metas.append(_init_train_meta(meta, graph, samples, False, epoch_shift=epoch_shift))
            schedule.append(ratio)

        train_for_samples(self._session, epoch, train_metas, multitask_scheduler(schedule))


def _init_train_meta(
        meta, graph, samples, freeze_shared_ce, classifier=None,
        early_stopping_callback=lambda c, e: False, epoch_shift=None):

    batcher_factory = get_bucketed_batcher(
        samples, meta.props["batch_size"], meta.feature_extractor, meta.props.get("bucket_length", 7), True, True,
        meta.props.get("buffer_size", 10000)
    )

    controllers = {
        "learning_rate": get_decayed_lr(meta.props["learning_rate"], meta.props.get("lr_decay", 0)),
        "dropout_rate": get_const_controller(meta.props.get("dropout", 1.0)),
        "freeze_shared_ce": get_const_controller(freeze_shared_ce)
    }
    if epoch_shift is not None:
        controllers = {
            name: get_epoch_shifting_wrapper(ctrl, epoch_shift)
            for name, ctrl in controllers.items()
        }
    return TaskTrainMeta(meta.task_name, graph, batcher_factory, controllers, classifier, early_stopping_callback)
