from operator import itemgetter

import tensorflow as tf
from logging import getLogger
from collections import namedtuple

from derek.common.nn.tf_trainers import get_train_op
from derek.common.nn.tf_layers import create_context_encoder, apply_dense_layer_to_nd_tensor, concat_tensors_list, \
    dropout
from derek.common.nn.aggregation_strategies import BahdanauAttention, LuongAttention, MaxPoolingAggregation, \
    MeanPoolingAggregation, IndexedAggregation
from derek.common.nn.tf_losses import get_loss
from derek.common.feature_extraction.features_meta import BasicFeaturesMeta
from derek.rel_ext.feature_extraction.feature_meta import AttentionFeaturesMeta
from derek.common.nn.features_lookups import collect_basic_features
from derek.common.nn.graph_factory import TaskGraphMeta

logger = getLogger('logger')

RelExtTaskGraphMeta = namedtuple(
    'RelExtTaskGraphMeta', ['task_name', 'props', 'metas', 'out_size', 'needs_labels_mask'])


def build_task_graph_meta(task_meta: RelExtTaskGraphMeta):
    return TaskGraphMeta(task_meta.task_name, lambda e, we, inp, o: _build_graph(task_meta, e, we, inp, o))


def _build_graph(task_meta: RelExtTaskGraphMeta, shared_encoding, word_embeddings, shared_inputs, optimizer):
    graph = {"inputs": dict(shared_inputs)}
    graph["inputs"]["labels"] = tf.placeholder(tf.int32, shape=[None], name='labels')
    dropout_rate = graph["inputs"]["dropout_rate"]
    seq_len = graph["inputs"]["seq_len"]

    # [bs, 2 entities, start and end indices]
    graph["inputs"]["indices"] = tf.placeholder(tf.int32, shape=[None, 2, 2], name='indices')

    labels_mask = None
    if task_meta.needs_labels_mask:
        labels_mask = tf.placeholder(tf.int32, shape=[None, task_meta.out_size], name="labels_mask")
        graph["inputs"]["labels_mask"] = labels_mask

    context_encoding_non_linearity_size = task_meta.props.get("context_encoding_non_linearity_size", -1)
    if context_encoding_non_linearity_size > 0:
        shared_encoding = apply_dense_layer_to_nd_tensor(
            shared_encoding, context_encoding_non_linearity_size, "context_encoding_non_linearity")

    context_encoding = create_context_encoding(task_meta, graph, shared_encoding, word_embeddings)
    if context_encoding is None:
        raise Exception("No context_encoding provided in {} task".format(task_meta.task_name))

    attention_3d_features, attention_2d_features = collect_attention_features(task_meta.metas.attention, graph)

    hidden = aggregate_encoding(
        context_encoding, seq_len, graph["inputs"]["indices"],
        attention_3d_features, attention_2d_features, task_meta.task_name, dropout_rate, task_meta.props)

    if hidden is None:
        raise Exception("No aggregation strategy provided")

    hidden = tf.nn.dropout(hidden, dropout_rate)
    classifier_features = collect_classifier_features(task_meta.metas.classifier, graph)
    if classifier_features is not None:
        logger.info("Size of classifier features in {} task = {}".format(
            task_meta.task_name, classifier_features.shape[1].value))
        hidden = tf.concat([hidden, classifier_features], axis=-1)

    logger.info("Size of hidden vector in {} task = {}".format(task_meta.task_name, hidden.shape[1].value))

    classification_dense_size = task_meta.props.get('classification_dense_size', 0)
    if classification_dense_size > 0:
        hidden = tf.layers.dense(hidden, classification_dense_size, tf.nn.sigmoid)

    loss, label, scores = get_loss(hidden, graph["inputs"]["labels"], task_meta.out_size, task_meta.props, labels_mask)
    train_op = get_train_op(loss, optimizer, task_meta.props)

    graph.update({
        'losses': [loss],
        'train_ops': [train_op],
        'outputs': {
            "predictions": label,
            "scores": scores
        }
    })

    return graph


def create_context_encoding(task_meta: RelExtTaskGraphMeta, graph, shared_encoding, word_embeddings, rank=2):
    seq_len = graph["inputs"]["seq_len"]
    dropout_rate = graph["inputs"]["dropout_rate"]

    cont, cat, placeholders = collect_basic_features(task_meta.metas.encoder, rank)

    if task_meta.props.get("add_we", False):
        cont = concat_tensors_list([cont, word_embeddings])

    if task_meta.props.get("add_shared", False):
        cont = concat_tensors_list([cont, shared_encoding])

    encoder_features = concat_tensors_list([dropout(cont, dropout_rate), cat])
    graph["inputs"].update(placeholders)

    fw, bw, features_encoding = None, None, None
    if encoder_features is not None:
        graph["inputs"].update(placeholders)

        logger.info("Size of specific encoder features in {} task = {}".format(
            task_meta.task_name, encoder_features.shape[rank].value))

        fw, bw, features_encoding = create_context_encoder(
            encoder_features, seq_len,
            skip_connection=task_meta.props.get("specific_encoder_skip_connection", False),
            encoding_type=task_meta.props.get("specific_encoder_type", "lstm"),
            encoding_size=task_meta.props.get("specific_encoder_size", 0),
            kernel_size=task_meta.props.get("specific_encoder_kernel_size", None),
            layers_num=task_meta.props.get("specific_encoder_layers_num", None),
            max_len=task_meta.props.get("specific_encoder_transformer_max_len", None),
            dropout=1 - dropout_rate)

    if task_meta.props.get("concat_we", False):
        features_encoding = concat_tensors_list([features_encoding, word_embeddings])
        fw, bw = None, None

    if task_meta.props.get("concat_shared", False):
        features_encoding = concat_tensors_list([features_encoding, shared_encoding])
        fw, bw = None, None

    return fw, bw, features_encoding


def aggregate_encoding(
        context_encoding, seq_len, indices, attention_3d_features, attention_2d_features, task_name, dp_rate, props):
    if attention_3d_features is None and attention_2d_features is None:
        logger.info("No attention features in props in {} task".format(task_name))

    if attention_3d_features is not None:
        logger.info("Size of attention 3D features in {} task = {}".format(
            task_name, attention_3d_features.shape[2].value))

    if attention_2d_features is not None:
        logger.info("Size of attention 2D features in {} task = {}".format(
            task_name, attention_2d_features.shape[1].value))

    strategies = {

        "attention": lambda prop: create_attention_mechanism(
            context_encoding[2], seq_len, attention_3d_features, attention_2d_features, dp_rate, prop),

        "max_pooling": lambda prop: create_pooling_mechanism(
            context_encoding, seq_len, attention_3d_features, attention_2d_features, dp_rate, {**prop, "type": "max"}),

        "mean_pooling": lambda prop: create_pooling_mechanism(
            context_encoding, seq_len, attention_3d_features, attention_2d_features, dp_rate, {**prop, "type": "mean"}),

        "take_spans": lambda prop: create_spans_aggregation(
            context_encoding, indices, attention_3d_features, attention_2d_features),

        "last_hiddens": lambda prop: create_last_hiddens_aggregation(
            context_encoding, seq_len, attention_3d_features, attention_2d_features)
    }

    if "aggregation" not in props:
        return None

    aggregations = [strategies[name](prop) for name, prop in sorted(props["aggregation"].items(), key=itemgetter(0))]
    return concat_tensors_list(aggregations)


def create_attention_mechanism(
        context_encoding, seq_len, attention_3d_features, attention_2d_features, dropout_placeholder, props):

    attention_input = tf.nn.dropout(context_encoding, dropout_placeholder)
    if attention_3d_features is not None:
        attention_input = tf.concat([attention_input, attention_3d_features], axis=2)

    attention_aggregation_dense_size = props.get("dense_size", -1)
    if attention_aggregation_dense_size > 0:
        aggregation_tensor = apply_dense_layer_to_nd_tensor(
            context_encoding, attention_aggregation_dense_size, "att_aggr")
    else:
        aggregation_tensor = context_encoding

    if attention_2d_features is not None:
        queries = tf.tile(tf.expand_dims(attention_2d_features, 1), [1, tf.shape(context_encoding)[1], 1])
    else:
        queries = None

    attention_type = props.get("type", "bahdanau")
    if attention_type == "bahdanau":
        keys = concat_tensors_list([attention_input, queries])
        return BahdanauAttention.from_props(props)((keys, aggregation_tensor, seq_len))
    elif attention_type == "luong":
        if queries is None:
            raise Exception("No attention 2d features provided for Luong attention")
        return LuongAttention.from_props(props)((attention_input, queries, aggregation_tensor, seq_len))
    else:
        raise Exception("Unknown attention type")


def collect_attention_features(attention_meta: AttentionFeaturesMeta, graph, rank=2):
    dropout_rate = graph["inputs"]["dropout_rate"]
    cont_3d, cat_3d, placeholders_3d = collect_basic_features(attention_meta.get_token_features_meta(), rank)
    cont_2d, cat_2d, placeholders_2d = collect_basic_features(attention_meta.get_relation_features_meta(), rank-1)
    graph["inputs"].update({**placeholders_3d, **placeholders_2d})
    return concat_tensors_list([dropout(cont_3d, dropout_rate), cat_3d]),\
        concat_tensors_list([dropout(cont_2d, dropout_rate), cat_2d])


def collect_classifier_features(features_meta: BasicFeaturesMeta, graph):
    cont, cat, placeholders = collect_basic_features(features_meta, 1)
    graph["inputs"].update(placeholders)
    return concat_tensors_list([dropout(cont, graph["inputs"]["dropout_rate"]), cat])


def create_pooling_mechanism(
        context_encoding, seq_len, attention_3d_features, attention_2d_features, dropout_placeholder, props):

    _, _, context_encoding = context_encoding

    if attention_2d_features is not None:
        attention_2d_features = tf.tile(tf.expand_dims(attention_2d_features, 1), [1, tf.shape(context_encoding)[1], 1])

    pooling_input = concat_tensors_list([context_encoding, attention_3d_features, attention_2d_features])

    if "dense_size" in props:
        pooling_input = tf.nn.dropout(pooling_input, dropout_placeholder)
        pooling_input = apply_dense_layer_to_nd_tensor(pooling_input, props["dense_size"], "pooling_dense")

    if props["type"] == "max":
        aggregation = MaxPoolingAggregation()
    elif props["type"] == "mean":
        aggregation = MeanPoolingAggregation()
    else:
        raise Exception(f"{props['type']} pooling is not supported")

    return aggregation((pooling_input, seq_len))


def create_spans_aggregation(context_encoding, indices, attention_3d_features, attention_2d_features):
    fw, bw, _ = context_encoding
    if fw is None or bw is None:
        raise Exception("Spans aggregation requires encoding with forward and backward pass")

    if attention_3d_features is not None:
        logger.info("Attention 3d features provided in spans aggregation will be ignored")

    fw_at_ends = IndexedAggregation()((fw, indices[:, :, 1] - 1))
    bw_at_starts = IndexedAggregation()((bw, indices[:, :, 0]))

    concatted = tf.reshape(
        tf.concat((fw_at_ends, bw_at_starts), axis=-1), [-1, indices.shape[1] * (fw.shape[2] + bw.shape[2])])

    return concat_tensors_list([concatted, attention_2d_features])


def create_last_hiddens_aggregation(context_encoding, seq_len, attention_3d_features, attention_2d_features):
    seq_len = tf.reshape(seq_len, [-1, 1, 1])
    indices = tf.concat([tf.zeros_like(seq_len), seq_len], axis=-1)
    return create_spans_aggregation(context_encoding, indices, attention_3d_features, attention_2d_features)
