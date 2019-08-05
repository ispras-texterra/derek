from logging import getLogger

import tensorflow as tf

from derek.common.feature_extraction.features_meta import TokenFeaturesMeta
from derek.common.nn.features_lookups import collect_token_features, collect_basic_features
from derek.common.nn.tf_layers import create_birnn_layer, concat_tensors_list, dropout
from derek.common.nn.tf_losses import get_loss
from derek.common.nn.tf_trainers import get_train_op, get_optimizer
from derek.rel_ext.nn.graph_factory import collect_classifier_features

logger = getLogger('logger')


def build_coref_graph(props: dict, entity_encoder_meta: TokenFeaturesMeta, task_meta: list):
    dropout_rate = tf.placeholder_with_default(1.0, [], 'dropout')
    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    gold_idx = tf.placeholder(tf.int32, shape=[None], name='gold_labels')
    seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len')
    entity_seq_len = tf.placeholder(tf.int32, shape=[None, None], name='entity_seq_len')
    optimizer = get_optimizer(props, learning_rate)

    freeze_shared_ce = tf.placeholder_with_default(False, [], 'freeze_shared_ce')  # XXX fid a way to remove this

    task_meta = task_meta[0]

    graph = {
        "inputs": {
            'labels': gold_idx,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            "seq_len": seq_len,
            "entity_seq_len": entity_seq_len,
            'freeze_shared_ce': freeze_shared_ce
        }
    }

    entity_cont, entity_cat, inputs = collect_token_features(
        entity_encoder_meta, dropout_rate, kernel_sizes=props.get("char_kernel_sizes", []),
        kernel_num_features=props.get("char_kernel_num_features", []), input_dims=3)
    entity_encoder_features = concat_tensors_list([dropout(entity_cont, dropout_rate), entity_cat])

    logger.info("Size of entity encoder features = {}".format(entity_encoder_features.shape[-1]))

    graph["inputs"].update(inputs)

    encoder_cont, encoder_cat, inputs = collect_basic_features(task_meta.metas.encoder, 2)
    graph["inputs"].update(inputs)

    entity_hidden = _create_entity_encoding(entity_encoder_features, entity_seq_len, props)

    entity_hidden = concat_tensors_list([
        dropout(concat_tensors_list([encoder_cont, entity_hidden]), dropout_rate),
        encoder_cat])

    logger.info("Size of entity encoding = {}".format(entity_hidden.shape[-1].value))

    _, _, context_encoding_state = create_birnn_layer(
        entity_hidden, seq_len, props['encoding_type'], props['encoding_size'], "context_encoding")

    hidden = tf.concat((context_encoding_state[0].h, context_encoding_state[1].h), axis=-1)
    hidden = tf.nn.dropout(hidden, dropout_rate)

    classifier_features = collect_classifier_features(task_meta.metas.classifier, graph)
    if classifier_features is not None:
        logger.info("Size of classifier features  = {}".format(classifier_features.shape[1].value))
        hidden = tf.concat([hidden, classifier_features], axis=-1)

    logger.info("Size of hidden vector = {}".format(hidden.shape[-1].value))

    classification_dense_size = task_meta.props.get('classification_dense_size', 0)
    if classification_dense_size > 0:
        hidden = tf.layers.dense(hidden, classification_dense_size, tf.nn.sigmoid)

    loss, label, scores = get_loss(hidden, graph["inputs"]["labels"], task_meta.feature_extractor.get_labels_size(),
                                   task_meta.props)
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


def _create_entity_encoding(entity_encoder_features, entity_seq_len, props):
    feat_shape = tf.shape(entity_encoder_features)
    seq_shape = tf.shape(entity_seq_len)

    reshaped_feats = tf.reshape(entity_encoder_features, [feat_shape[0] * feat_shape[1], feat_shape[2],
                                                          entity_encoder_features.shape[3].value])
    reshaped_seq_len = tf.reshape(entity_seq_len, [seq_shape[0] * seq_shape[1]])
    if props['entity_encoding_type'] == 'rnn':
        _, _, context_encoding_state = create_birnn_layer(
            reshaped_feats, reshaped_seq_len, props['encoding_type'], props['entity_encoding_size'], "entity_encoding")

        last_states = []
        for state in context_encoding_state:
            last_states.append(tf.reshape(state.h, [feat_shape[0], feat_shape[1], state.h.shape[-1].value]))
        return tf.concat(last_states, -1)
    elif props['entity_encoding_type'] == 'mean':
        sum = tf.reduce_sum(reshaped_feats, axis=-2)
        lens = tf.where(tf.equal(reshaped_seq_len, tf.zeros_like(reshaped_seq_len)),
                        tf.ones_like(reshaped_seq_len),
                        reshaped_seq_len)
        lens = tf.tile(tf.expand_dims(tf.cast(lens, tf.float32), -1), [1, entity_encoder_features.shape[3].value])
        context_encoding = sum / lens
        return tf.reshape(context_encoding, [feat_shape[0], feat_shape[1], entity_encoder_features.shape[3].value])
