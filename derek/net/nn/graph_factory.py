import tensorflow as tf
from logging import getLogger
from collections import namedtuple

from derek.common.nn.aggregation_strategies import BahdanauAttention
from derek.common.nn.tf_trainers import get_train_op
from derek.common.nn.tf_layers import apply_dense_layer_to_nd_tensor
from derek.common.nn.tf_losses import get_loss
from derek.common.nn.graph_factory import TaskGraphMeta
from derek.rel_ext.nn.graph_factory import create_context_encoding, collect_attention_features, \
    aggregate_encoding, collect_classifier_features

logger = getLogger('logger')

NETTaskGraphMeta = namedtuple(
    'NETTaskGraphMeta', ['task_name', 'props', 'metas', 'out_size', 'needs_labels_mask'])


def build_task_graph_meta(task_meta: NETTaskGraphMeta):
    return TaskGraphMeta(task_meta.task_name, lambda e, we, inp, o: _build_graph(task_meta, e, we, inp, o))


def _build_graph(task_meta: NETTaskGraphMeta, shared_encoding, word_embeddings, shared_inputs, optimizer):
    graph = {"inputs": dict(shared_inputs)}
    graph["inputs"]["labels"] = tf.placeholder(tf.int32, shape=[None], name='labels')
    dropout_rate = graph["inputs"]["dropout_rate"]
    seq_len = graph["inputs"]["seq_len"]
    chain_len = graph["inputs"]["chain_len"]

    # [bs, chain_len, 1 entity, start and end indices]
    graph["inputs"]["indices"] = tf.placeholder(tf.int32, shape=[None, None, 1, 2], name='indices')

    labels_mask = None
    if task_meta.needs_labels_mask:
        labels_mask = tf.placeholder(tf.int32, shape=[None, task_meta.out_size], name="labels_mask")
        graph["inputs"]["labels_mask"] = labels_mask

    context_encoding_non_linearity_size = task_meta.props.get("context_encoding_non_linearity_size", -1)
    if context_encoding_non_linearity_size > 0:
        shared_encoding = apply_dense_layer_to_nd_tensor(
            shared_encoding, context_encoding_non_linearity_size, "context_encoding_non_linearity")

    context_encoding = create_context_encoding(task_meta, graph, shared_encoding, word_embeddings, 3)
    if context_encoding is None:
        raise Exception("No context_encoding provided in {} task".format(task_meta.task_name))

    attention_3d_features, attention_2d_features = collect_attention_features(task_meta.metas.attention, graph, 3)
    max_chain_len = tf.shape(context_encoding[2])[1]

    hidden = aggregate_encoding(
        tuple(map(_unfold_chains, context_encoding)), tf.reshape(seq_len, [-1]),
        tf.reshape(graph["inputs"]["indices"], shape=[-1, 1, 2]),
        _unfold_chains(attention_3d_features), _unfold_chains(attention_2d_features), task_meta.task_name,
        dropout_rate, task_meta.props)

    if hidden is None:
        raise Exception("No aggregation provided")

    hidden = tf.nn.dropout(hidden, dropout_rate)

    hidden = tf.reshape(hidden, [-1, max_chain_len, hidden.shape[-1].value])

    hidden = BahdanauAttention(-1)((hidden, hidden, chain_len))
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


def _unfold_chains(tensor: tf.Tensor):
    if tensor is None:
        return None
    shape = tf.shape(tensor)  # [batch, chain, ..., dim]
    dim = tensor.shape[-1].value
    return tf.reshape(tensor, tf.concat([[-1], shape[2:-1], [dim]], axis=-1))
