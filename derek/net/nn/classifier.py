from collections import defaultdict
from operator import itemgetter
from typing import Iterable, Dict, Union, List, Callable, Tuple

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
from derek.coref.data.model import CoreferenceChain
from derek.data.model import Document, Entity
from derek.common.nn.graph_factory import build_graphs_with_shared_encoder
from derek.data.entities_collapser import EntitiesCollapser
from derek.ner.post_processing.process_similar_entities import chain_similar_entities
from derek.net.feature_extraction.feature_extractor import generate_feature_extractor, NETFeatureExtractor, \
    GroupingFeatureExtractor
from derek.net.nn.graph_factory import build_task_graph_meta, NETTaskGraphMeta
from derek.common.helper import FuncIterable

logger = getLogger('logger')


class _Classifier:
    # TODO make prediction batch size configurable
    _PREDICTION_BATCH_SIZE = 300

    def __init__(self, graph, feature_extractor: GroupingFeatureExtractor, feature_computer, session, saver,
                 grouper_collapser: '_GrouperCollapser'):
        self.graph = graph
        self.extractor = feature_extractor
        self.feature_computer = feature_computer
        self.session = session
        self.saver = saver
        self.grouper_collapser = grouper_collapser

    def predict_doc(self, doc: Document) -> Dict[Entity, Union[str, None]]:
        ret = {}

        collapsed_doc, groups, reversed_mapping = self.grouper_collapser.prepare_doc_with_collapsing(doc)
        doc = self.feature_computer.create_features_for_doc(collapsed_doc)
        chain_samples = self.extractor.extract_features_from_doc(doc, groups)
        entities_to_predict, samples_to_predict = [], []

        for chain, sample in chain_samples:
            if isinstance(sample, dict):
                entities_to_predict.append(chain)
                samples_to_predict.append(sample)
            else:
                ret.update(dict.fromkeys(map(reversed_mapping.get, chain), sample))

        batcher = get_standard_batcher_factory(
            samples_to_predict, self._PREDICTION_BATCH_SIZE, self.extractor.get_padding_value_and_rank)

        # we have only predictions as output
        predicted_labels, = predict_for_samples(self.graph, self.session, ["predictions"], batcher)

        predicted_types = [self.extractor.get_type(label) for label in predicted_labels]

        for entities, predicted_type in zip(entities_to_predict, predicted_types):
            ret.update(dict.fromkeys(map(reversed_mapping.get, entities), predicted_type))

        return ret

    def save(self, out_path):
        save_classifier(out_path, self.extractor, self.feature_computer, self.graph, self.session, self.saver)
        save_with_pickle(self.grouper_collapser, out_path, "grouper_collapser")

    @classmethod
    def load(cls, path, session):
        extractor, feature_computer, graph, saver = load_classifier(
            path, NETFeatureExtractor, SyntacticFeatureComputer, session)

        return cls(graph, extractor, feature_computer, session, saver, load_with_pickle(path, "grouper_collapser"))


class NETClassifier(TFSessionAwareClassifier):
    def __init__(self, model_path: str):
        super().__init__()
        self.path = model_path

    def _load(self):
        return _Classifier.load(self.path, self._session)


class NETTrainer(TFSessionAwareTrainer):
    def __init__(self, props: dict):
        super().__init__(props)
        self.props = {**props, "concat_shared": True}

    def train(
            self, docs: Iterable[Document], unlabeled_docs: Iterable[Document] = None,
            early_stopping_callback: Callable[[NETClassifier, int], bool] = lambda c, e: False):

        feature_computer = SyntacticFeatureComputer(self.props.get('morph_feats_list', DEFAULT_FEATS_LIST))

        if self.props.get("unify_similar_entities_types", False):
            grouper = chain_similar_entities
            get_bucket_for_sample = lambda s: int(s["chain_len"] == 1)
        else:
            grouper = chain_individual_entities
            get_bucket_for_sample = lambda s: s["seq_len"][0] // self.props["batcher"]["bucket_length"]

        grouper_collapser = _GrouperCollapser(
            CoreferenceChainGrouper(grouper),
            EntitiesCollapser(self.props.get("types_to_collapse", set()), collapse_with_ne=True))

        docs_groups = FuncIterable(
            lambda: map(itemgetter(0, 1), map(grouper_collapser.prepare_doc_with_collapsing, docs)))
        collapsed_docs = FuncIterable(lambda: map(itemgetter(0), docs_groups))
        precomputed_docs = FuncIterable(lambda: map(feature_computer.create_features_for_doc, collapsed_docs))
        groups = FuncIterable(lambda: map(itemgetter(1), docs_groups))

        char_padding_size = get_char_padding_size(self.props)
        feature_extractor, metas, token_meta = generate_feature_extractor(
            precomputed_docs, self.props, char_padding_size)
        feature_extractor = GroupingFeatureExtractor(feature_extractor, group_level_features=["labels_mask"])

        # reuse because this task is kinda unary rel-ext
        task_graph_meta = NETTaskGraphMeta("NET", self.props, metas, feature_extractor.get_labels_size(), True)
        # we have only one graph
        graph, = build_graphs_with_shared_encoder(self.props, token_meta, [build_task_graph_meta(task_graph_meta)], rank=3)

        init = tf.global_variables_initializer()
        self._session.run(init)

        samples = list(feature_extractor.extract_features_from_docs(precomputed_docs, groups))
        saver = tf.train.Saver(save_relative_paths=True)

        classifier = _Classifier(graph, feature_extractor, feature_computer, self._session, saver, grouper_collapser)

        batcher_factory = get_batcher_from_props(
            samples, self.props["batcher"], feature_extractor.get_padding_value_and_rank, True, True,
            get_bucket_for_sample)

        train_meta = TaskTrainMeta(
            "NET", graph, batcher_factory,
            {
                "learning_rate": get_decayed_lr(self.props["learning_rate"], self.props.get("lr_decay", 0)),
                "dropout_rate": get_const_controller(self.props.get("dropout", 1.0))
            },
            classifier, early_stopping_callback)

        train_for_samples(self._session, self.props["epoch"], [train_meta])


class CoreferenceChainGrouper(object):
    def __init__(self, grouper: Callable[[Document, List[Entity]], List[CoreferenceChain]]):
        self.__grouper = grouper

    def __call__(self, doc: Document) -> List[Tuple[Entity]]:
        chains = self.__grouper(doc, doc.extras['ne'])
        ret = []

        for chain in chains:
            typed_chain = defaultdict(list)
            for ent in chain.entities:
                typed_chain[ent.type].append(ent)

            ret.extend(map(itemgetter(1), sorted(typed_chain.items(), key=itemgetter(0))))

        return ret


def chain_individual_entities(doc: Document, entities: List[Entity]) -> List[CoreferenceChain]:
    return [CoreferenceChain([entity]) for entity in entities]


class _GrouperCollapser:
    def __init__(self, grouper: CoreferenceChainGrouper, collapser: EntitiesCollapser):
        self.__grouper = grouper
        self.__collapser = collapser

    def prepare_doc_with_collapsing(self, doc: Document):
        groups = self.__grouper(doc)
        collapsed_doc, mapping = self.__collapser.transform_with_mapping(doc)
        collapsed_groups = [tuple(map(mapping.get, group)) for group in groups]
        reversed_mapping = {v: k for k, v in mapping.items()}
        return collapsed_doc, collapsed_groups, reversed_mapping
