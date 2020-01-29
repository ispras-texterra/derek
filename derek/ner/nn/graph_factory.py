import tensorflow as tf
from logging import getLogger

from derek.common.nn.tf_losses import get_seq_labelling_loss
from derek.common.nn.tf_trainers import get_train_op
from derek.common.nn.graph_factory import TaskGraphMeta

logger = getLogger('logger')


def build_task_graph_meta(props: dict, out_size: int):
    return TaskGraphMeta("NER", lambda e, _, inp, o: _build_graph(props, out_size, e, inp, o))


def _build_graph(props: dict, out_size: int, shared_encoding, shared_inputs, optimizer):
    graph = {"inputs": dict(shared_inputs), "outputs": dict()}
    seq_len = graph["inputs"]["seq_len"]

    gold_labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
    graph["inputs"]["labels"] = gold_labels

    shared_encoding = tf.nn.dropout(shared_encoding, graph["inputs"]["dropout_rate"])
    loss, label, scores = get_seq_labelling_loss(shared_encoding, seq_len, gold_labels, out_size, props)
    train_op = get_train_op(loss, optimizer, props)

    graph["outputs"]["predictions"] = label
    graph['losses'] = [loss]
    graph['train_ops'] = [train_op]
    graph["outputs"]["scores"] = scores

    return graph
