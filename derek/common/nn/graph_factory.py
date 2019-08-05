import tensorflow as tf
from logging import getLogger
from collections import namedtuple
from typing import List

from derek.common.nn.tf_trainers import get_optimizer
from derek.common.nn.tf_layers import create_context_encoder, create_filtering_layer, concat_tensors_list, dropout
from derek.common.feature_extraction.features_meta import TokenFeaturesMeta
from derek.common.nn.features_lookups import collect_basic_features, collect_char_features, collect_word_embeddings

logger = getLogger('logger')

TaskGraphMeta = namedtuple('TaskGraphMeta', ['task_name', 'factory'])


def build_graphs_with_shared_encoder(
        props: dict, shared_features_meta: TokenFeaturesMeta, task_specific_metas: List[TaskGraphMeta], rank: int = 2):
    dropout_rate = tf.placeholder_with_default(1.0, [], 'dropout')
    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    seq_len = tf.placeholder(tf.int32, shape=[None] * (rank - 1), name='seq_len')
    optimizer = get_optimizer(props, learning_rate)

    freeze_shared_ce = tf.placeholder_with_default(False, [], 'freeze_shared_ce')

    shared_inputs = {
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        "seq_len": seq_len,
        'freeze_shared_ce': freeze_shared_ce
    }

    # TODO: we can use "seq_len" key to store list of input tensor lengths across each dimension.
    # this change will affect lots of other code, so it should be done as separate task
    if rank > 2:
        shared_inputs["chain_len"] = tf.placeholder(tf.int32, shape=[None] * (rank - 2), name='chain_len')
    if rank > 3:
        raise Exception(f"can't build graph with input rank {rank}")

    word_embeddings, we_placeholders = collect_word_embeddings(shared_features_meta.we_meta, input_dims=rank)
    cont, cat, basic_placeholders = collect_basic_features(shared_features_meta.basic_meta, input_dims=rank)
    char_features, char_placeholders = collect_char_features(
        shared_features_meta.char_meta, dropout_rate, input_dims=rank,
        kernel_sizes=props.get("char_kernel_sizes", []),
        kernel_num_features=props.get("char_kernel_num_features", []))

    cont = concat_tensors_list([word_embeddings, cont, char_features])
    shared_features = concat_tensors_list([dropout(cont, dropout_rate), cat])

    if shared_features is None:
        raise Exception('No shared token features given.')

    shared_inputs.update({**we_placeholders, **basic_placeholders, **char_placeholders})

    logger.info("Size of shared context encoder features = {}".format(shared_features.shape[-1]))

    shared_features = create_filtering_layer(shared_features, **props)
    _, _, shared_context_encoding = create_context_encoder(
        shared_features, seq_len, encoding_type=props.get("encoding_type", "lstm"),
        encoding_size=props.get("encoding_size", 0), skip_connection=props.get("skip_connection", False),
        kernel_size=props.get("encoder_kernel_size", None), layers_num=props.get("encoder_layers_num", None),
        max_len=props.get("transformer_max_len", None), dropout=1 - dropout_rate)

    shared_context_encoding = tf.cond(
        freeze_shared_ce,
        true_fn=lambda: tf.stop_gradient(shared_context_encoding),
        false_fn=lambda: shared_context_encoding)

    logger.info("Size of shared context encoding = {}".format(shared_context_encoding.shape[-1].value))

    task_graphs = []
    for task_meta in task_specific_metas:
        with tf.variable_scope(task_meta.task_name):
            task_graph = task_meta.factory(shared_context_encoding, word_embeddings, shared_inputs, optimizer)

        task_graphs.append(task_graph)

    return task_graphs
