import tensorflow as tf
import numpy as np


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, non_linearity_size: int, attention_per_feature: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._non_linearity_size = non_linearity_size
        self._attention_per_feature = attention_per_feature

    def build(self, input_shape):
        keys_shape, values_shape, seq_len_shape = input_shape

        if self._non_linearity_size > 0:
            self._non_linearity = tf.layers.Dense(self._non_linearity_size, activation=tf.tanh)
        else:
            self._non_linearity = tf.identity

        if self._attention_per_feature:
            coefficients_dim = values_shape[-1]
        else:
            coefficients_dim = 1

        self._dense = tf.layers.Dense(coefficients_dim)

    def call(self, inputs: tuple, **kwargs):
        """
            Computes bahdanau attention over provided keys, values and seq_len
            :param inputs: tuple (keys, values, seq_len). Each tensor of shape [batch_size, seq_len, d_i].
                seq_len - tensor of shape [batch_size]
        """
        keys, values, seq_len = inputs
        if self._non_linearity is not None:
            keys = self._non_linearity(keys)

        weights = self._dense(keys)
        coefficients = tf.nn.softmax(_mask_weights(weights, seq_len), axis=1)

        return tf.reduce_sum(values * coefficients, 1)

    def compute_output_shape(self, input_shape):
        keys_shape, values_shape, seq_len_shape = input_shape
        return [values_shape[0], values_shape[-1]]

    @classmethod
    def from_props(cls, props: dict):
        return cls(props.get("non_linearity_size", -1), props.get("attention_per_feature", False))


class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, normalise_weights: bool = False, **kwargs):
        self._normalise_weights = normalise_weights
        super().__init__(**kwargs)

    def build(self, input_shape):
        keys_shape, queries_shape, values_shape, seq_len_shape = input_shape
        self._dense = tf.layers.Dense(keys_shape[-1], use_bias=False)

    def call(self, inputs: tuple, **kwargs):
        """
            Computes Luong multiplicative attention over provided keys, queries, values and seq_len
            :param inputs: tuple (keys, queries, values, seq_len). Each tensor of shape [batch_size, seq_len, d_i].
                seq_len - tensor of shape [batch_size]
        """
        keys, queries, values, seq_len = inputs
        densed_queries = self._dense(queries)

        # [batch_size, seq_len, seq_len]
        weights = tf.matmul(densed_queries, keys, transpose_b=True)
        if self._normalise_weights:
            weights /= np.sqrt(keys.shape[-1])

        # [batch_size, seq_len]
        weights = tf.linalg.diag_part(weights)
        # [batch_size, seq_len, 1]
        weights = tf.expand_dims(weights, axis=-1)

        coefficients = tf.nn.softmax(_mask_weights(weights, seq_len), axis=1)
        return tf.reduce_sum(values * coefficients, 1)

    def compute_output_shape(self, input_shape):
        keys_shape, queries_shape, values_shape, seq_len_shape = input_shape
        return [values_shape[0], values_shape[-1]]

    @classmethod
    def from_props(cls, props: dict):
        return cls(props.get("normalise_weights", False))


def _mask_weights(weights, seq_len):
    # [[1....1 0 ... 0] ...] 0-s on padding tokens
    mask = tf.expand_dims(tf.sequence_mask(seq_len, dtype=tf.float32), -1)
    # 1-s on non-padding weights
    weights = weights * mask
    # -inf on padding weights
    minus_inf_mask = (1 - mask) * tf.constant(np.finfo(np.float32).min)
    return weights + minus_inf_mask


class MaxPoolingAggregation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs: tuple, **kwargs):
        """
        :param inputs: (input_tensor, seq_len)
        input_tensor with shape [batch_size, max_seq_len, x]
        seq_len with shape [batch_size]
        """
        input_tensor, seq_len = inputs
        mask = tf.expand_dims(tf.sequence_mask(seq_len, dtype=tf.float32), -1)
        # zeroed padding vectors
        input_tensor = mask * input_tensor
        # -inf on padding vectors
        padding_inf = tf.constant(np.finfo(np.float32).min) * (1 - mask) * tf.ones_like(input_tensor)

        return tf.reduce_max(input_tensor + padding_inf, axis=-2)

    def compute_output_shape(self, input_shape):
        input_tensor_shape, seq_len_shape = input_shape
        return [input_tensor_shape[0], input_tensor_shape[-1]]


class MeanPoolingAggregation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs: tuple, **kwargs):
        """
        :param inputs: (input_tensor, seq_len)
        input_tensor with shape [batch_size, max_seq_len, x]
        seq_len with shape [batch_size]
        """
        input_tensor, seq_len = inputs
        mask = tf.expand_dims(tf.sequence_mask(seq_len, dtype=tf.float32), -1)
        # zeroed padding vectors
        input_tensor = mask * input_tensor

        return tf.reduce_sum(input_tensor, axis=-2) / tf.expand_dims(tf.to_float(seq_len), axis=-1)

    def compute_output_shape(self, input_shape):
        input_tensor_shape, seq_len_shape = input_shape
        return [input_tensor_shape[0], input_tensor_shape[-1]]


class IndexedAggregation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs: tuple, **kwargs):
        """
        :param inputs: (input_tensor, indices)
        input_tensor with shape [batch_size, max_seq_len, x]
        indices with shape [batch_size, vectors_num]
        """
        input_tensor, indices = inputs
        return tf.reshape(tf.batch_gather(input_tensor, indices), [-1, input_tensor.shape[-1] * indices.shape[-1]])

    def compute_output_shape(self, input_shape):
        input_tensor_shape, indices_shape = input_shape
        return [input_tensor_shape[0], indices_shape[1] * input_tensor_shape[-1]]
