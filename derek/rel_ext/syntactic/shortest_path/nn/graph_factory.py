from collections import namedtuple

import tensorflow as tf
from logging import getLogger

from derek.common.nn.aggregation_strategies import IndexedAggregation
from derek.common.nn.tf_layers import apply_dense_layer_to_nd_tensor
from derek.common.nn.tf_losses import get_seq_labelling_loss
from derek.common.nn.tf_trainers import get_train_op
from derek.common.nn.graph_factory import TaskGraphMeta

logger = getLogger('logger')

SDPMeta = namedtuple("SDPMeta", ['props', 'out_size'])


def build_task_graph_meta(task_meta: SDPMeta):
    return TaskGraphMeta("SDP", lambda e, _, inp, o: _build_graph(task_meta, e, inp, o))


def _build_graph(meta: SDPMeta, shared_encoding, shared_inputs, optimizer):
    graph = {"inputs": dict(shared_inputs), "outputs": dict()}
    dropout_rate = graph["inputs"]["dropout_rate"]
    seq_len = graph["inputs"]["seq_len"]
    gold_labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
    indices = tf.placeholder(tf.int32, shape=[None, 2], name='indices')

    graph["inputs"]["labels"] = gold_labels
    graph["inputs"]["indices"] = indices

    output = shared_encoding

    context_encoding_non_linearity_size = meta.props.get("context_encoding_non_linearity_size", -1)
    if context_encoding_non_linearity_size > 0:
        output = apply_dense_layer_to_nd_tensor(
            shared_encoding, context_encoding_non_linearity_size, "context_encoding_non_linearity")

    queries = IndexedAggregation()((shared_encoding, indices))

    if meta.props.get("query_dense_size", -1) > 0:
        queries = tf.layers.dense(queries, meta.props["query_dense_size"], activation=tf.nn.leaky_relu)

    tiled_queries = tf.tile(tf.expand_dims(queries, 1), [1, tf.shape(shared_encoding)[1], 1])
    output = tf.concat([output, tiled_queries], axis=-1)
    output = tf.nn.dropout(output, dropout_rate)

    loss, label, _ = get_seq_labelling_loss(output, seq_len, gold_labels, meta.out_size, meta.props)
    train_op = get_train_op(loss, optimizer, meta.props)

    graph["outputs"]["predictions"] = label
    graph['losses'] = [loss]
    graph['train_ops'] = [train_op]

    return graph
