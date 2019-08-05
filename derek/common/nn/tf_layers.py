import numpy as np
import tensorflow as tf

from derek.common.nn.transformer import transformer


def create_char_cnn(input_tensor, kernel_sizes, kernel_num_features):
    input_shape = tf.shape(input_tensor)
    max_word_length = input_shape[-2]
    embed_size = input_tensor.shape[-1].value
    input_tensor = tf.reshape(input_tensor, [-1, 1, max_word_length, embed_size])

    pooleds = []
    for size, num_features in zip(kernel_sizes, kernel_num_features):
        K = tf.get_variable('char_kernel_' + str(size), [1, size, embed_size, num_features])

        conv2d = tf.nn.conv2d(input_tensor, K, [1, 1, 1, 1], 'VALID')

        pooled = tf.expand_dims(tf.reduce_max(conv2d, -2), -2)
        pooleds.append(pooled)

    pooled_aggr = tf.concat(pooleds, 3)
    pooled_aggr = tf.reshape(pooled_aggr, tf.concat([input_shape[:-2], [pooled_aggr.shape[3].value]], axis=-1))
    return pooled_aggr


def create_birnn(features, size, rnn_type, seq_len, name='bi_rnn'):
    with tf.variable_scope(name):
        if rnn_type == 'lstm':
            fw_cell = tf.nn.rnn_cell.LSTMCell(size)
            bw_cell = tf.nn.rnn_cell.LSTMCell(size)
        elif rnn_type == 'gru':
            fw_cell = tf.nn.rnn_cell.GRUCell(size)
            bw_cell = tf.nn.rnn_cell.GRUCell(size)
        else:
            raise AttributeError("Unknown rnn type!")
        (fw, bw), state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, features, sequence_length=seq_len, dtype=tf.float32)

        return fw, bw, state


def get_ma_hovy_initializer(tensor_size):
    return tf.random_uniform_initializer(-np.sqrt(3 / tensor_size), np.sqrt(3 / tensor_size))


def create_embedding_lookup(features_tensor, feature_info):
    emb_size = feature_info["embedding_size"]

    W = tf.get_variable(
        feature_info["name"] + "_embeddings_matrix", [feature_info["size"], emb_size],
        initializer=get_ma_hovy_initializer(emb_size))

    emb = tf.nn.embedding_lookup(W, features_tensor)
    return emb


def create_birnn_layer(inputs, seq_len, rnn_type, rnn_hidden_size, scope='bi_rnn'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        fw, bw, state = create_birnn(inputs, rnn_hidden_size, rnn_type, seq_len, name="hidden")
    return fw, bw, state


def create_filtering_layer(inputs, **config):
    filtering_strategy = config.get("filtering_strategy", "")

    if filtering_strategy:
        if filtering_strategy == "sigmoid":
            inputs = create_sigmoid_filtering(inputs)
        elif filtering_strategy == "highway":
            inputs = create_highway_layer(inputs)
        else:
            raise Exception("{} filtering not implemented".format(filtering_strategy))

    return inputs


def create_context_encoder(inputs, seq_len, **config):
    fw, bw, encoding = None, None, None

    encoding_type = config.get("encoding_type", "")
    encoding_size = config.get("encoding_size", 0)
    inputs_shape = tf.shape(inputs)
    inputs = tf.reshape(inputs, [-1, inputs_shape[-2], inputs.shape[-1].value])

    if encoding_type in {"lstm", "gru"} and encoding_size:
        seq_len = tf.reshape(seq_len, [-1])
        fw, bw, _ = create_birnn_layer(inputs, seq_len, encoding_type, encoding_size)
        encoding = tf.concat((fw, bw), 2)
    elif encoding_type == 'transformer':
        encoding = transformer(inputs, encoding_size, config.get("layers_num", None),
                               config.get("max_len", None), config.get('dropout', 0))
    elif encoding_type == 'cnn':
        encoding = cnn_context_encoding(inputs, encoding_size, config.get("kernel_size", None))
    elif encoding_type and encoding_size:
        raise Exception("{} context encoding not implemented".format(encoding_type))

    skip_connection = config.get("skip_connection", False)
    if skip_connection:
        if encoding is None:
            encoding = inputs
        else:
            encoding = tf.concat([inputs, encoding], 2)
        fw, bw = None, None

    if encoding is None:
        raise Exception("No models for context encoding were specified")

    return tuple(
        tf.reshape(enc, tf.concat([inputs_shape[:-1], [enc.shape[-1].value]], axis=-1)) if enc is not None else None
        for enc in [fw, bw, encoding])


def create_sigmoid_filtering(inputs):
    features_size = inputs.shape[-1].value

    reshaped_inputs = tf.reshape(inputs, [-1, features_size])

    filter_coefficients = tf.layers.dense(reshaped_inputs, 1, activation=tf.sigmoid, name="sigmoid_filtering_layer")

    output = tf.multiply(filter_coefficients, reshaped_inputs)

    return tf.reshape(output, tf.shape(inputs))


def create_highway_layer(inputs, carry_value=-1, activation=tf.tanh):
    features_size = inputs.shape[-1].value

    reshaped_inputs = tf.reshape(inputs, [-1, features_size])

    transform_gate = tf.layers.dense(reshaped_inputs, features_size, activation=tf.sigmoid,
                                     bias_initializer=tf.constant_initializer(carry_value),
                                     name="transform_gate")

    transformed_input = tf.layers.dense(reshaped_inputs, features_size, activation=activation, name="highway_gate")

    output = tf.multiply(transform_gate, transformed_input)+tf.multiply((1.0-transform_gate), reshaped_inputs)

    return tf.reshape(output, tf.shape(inputs))


def concat_tensors_list(tensor_list: list, axis: int = -1):
    tensor_list = list(filter(lambda t: t is not None, tensor_list))
    result = None
    if len(tensor_list) >= 2:
        result = tf.concat(tensor_list, axis)
    elif len(tensor_list) == 1:
        result = tensor_list[0]
    return result


def dropout(tensor, keep_prob):
    return tf.nn.dropout(tensor, keep_prob) if tensor is not None else None


def tile_to_sequence(tiled_tensor, sequence):
    if tiled_tensor is None:
        return None
    return tf.tile(tf.expand_dims(tiled_tensor, 1), [1, tf.shape(sequence)[1], 1])


def apply_dense_layer_to_nd_tensor(inputs, size, name="nd_dense_layer", activation=tf.nn.relu):
    features_size = inputs.shape[-1].value
    inputs_shape = tf.shape(inputs)

    reshaped_inputs = tf.reshape(inputs, [-1, features_size])
    dense_layer = tf.layers.dense(reshaped_inputs, size, activation=activation, name=name)

    return tf.reshape(dense_layer, tf.concat([inputs_shape[:-1], [size]], axis=-1))


def cnn_context_encoding(inputs, size, kernel_size):
    inputs = tf.expand_dims(inputs, 3)
    in_shape = tf.shape(inputs)
    padding = tf.zeros((in_shape[0], int((kernel_size - 1) / 2), inputs.shape[2].value, inputs.shape[3].value))
    # We assume that kernel_size is odd number
    inputs = tf.concat((padding, inputs, padding), axis=1)
    conv = tf.layers.conv2d(inputs, size, [kernel_size, inputs.shape[2].value], name='conv_encoding')

    return tf.squeeze(conv, axis=-2)
