import tensorflow as tf

from derek.common.feature_extraction.features_meta import TokenFeaturesMeta, BasicFeaturesMeta, WordEmbeddingsMeta, \
    CharsFeaturesMeta
from derek.common.nn.tf_layers import create_embedding_lookup, concat_tensors_list, create_char_cnn


def collect_token_features(features_meta: TokenFeaturesMeta, dropout, input_dims=2, **kwargs):
    word_embeddings, we_placeholders = collect_word_embeddings(features_meta.we_meta, input_dims)
    basic_cont_features, basic_cat_features, bf_placeholders =\
        collect_basic_features(features_meta.basic_meta, input_dims)
    char_features, cf_placeholders =\
        collect_char_features(features_meta.char_meta, dropout, input_dims,
                              kernel_sizes=kwargs.get("kernel_sizes", []),
                              kernel_num_features=kwargs.get("kernel_num_features", []))

    token_cont_features = [word_embeddings, basic_cont_features, char_features]
    placeholders = {**we_placeholders, **bf_placeholders, **cf_placeholders}
    return concat_tensors_list(token_cont_features), basic_cat_features, placeholders


def collect_word_embeddings(features_meta: WordEmbeddingsMeta, input_dims=2):
    placeholders = {}
    word_embeddings = []
    for feature in features_meta.get_precomputed_features():
        word_placeholder = tf.placeholder(tf.int32, shape=[None] * input_dims, name=feature['name'] + '_placeholder')
        placeholders[feature['name']] = word_placeholder
        embedding_matrix = tf.Variable(feature['vectors'], dtype=tf.float32, trainable=feature['trainable'])
        word_embeddings.append(tf.nn.embedding_lookup(embedding_matrix, word_placeholder))

    return concat_tensors_list(word_embeddings), placeholders


def collect_basic_features(features_meta: BasicFeaturesMeta, input_dims=2):
    placeholders = {}
    embedded_features, one_hot_features = _collect_categorical_features(features_meta, placeholders, input_dims)
    embedded_features += _collect_vectorized_features(features_meta, placeholders, input_dims)
    return concat_tensors_list(embedded_features), concat_tensors_list(one_hot_features), placeholders


def _collect_categorical_features(features_meta: BasicFeaturesMeta, placeholders, input_dims):
    embedded_features = []

    for feature in features_meta.get_embedded_features():
        placeholder = tf.placeholder(tf.int32, shape=[None]*input_dims, name=feature['name'] + '_placeholder')
        emb = create_embedding_lookup(placeholder, feature)
        embedded_features.append(emb)
        placeholders[feature['name']] = placeholder

    one_hot_features = []

    for feature in features_meta.get_one_hot_features():
        placeholder = tf.placeholder(tf.int32, shape=[None]*input_dims, name=feature['name'] + '_placeholder')
        one_hot = tf.one_hot(placeholder, feature['size'])
        one_hot_features.append(one_hot)
        placeholders[feature['name']] = placeholder

    return embedded_features, one_hot_features


def _collect_vectorized_features(features_meta: BasicFeaturesMeta, placeholders, input_dims):
    vectorized_features = []

    for feature in features_meta.get_vectorized_features():
        placeholder = tf.placeholder(tf.float32, shape=[None]*input_dims + [feature['size']],
                                     name=feature['name'] + '_placeholder')
        vectorized_features.append(placeholder)
        placeholders[feature['name']] = placeholder

    return vectorized_features


def collect_char_features(features_meta: CharsFeaturesMeta, dropout, input_dims=2, **kwargs):
    if not features_meta.get_char_features():
        return None, {}

    placeholders = {}
    char_features = _init_char_features(features_meta, placeholders, input_dims)
    char_features = tf.nn.dropout(char_features, dropout)
    char_features = create_char_cnn(char_features, kwargs.get("kernel_sizes", []),
                                    kwargs.get("kernel_num_features", []))
    return char_features, placeholders


def _init_char_features(features_meta: CharsFeaturesMeta, placeholders, input_dims):
    char_features = features_meta.get_char_features()
    embedded_features = []
    for feature in char_features:
        placeholder = tf.placeholder(tf.int32, shape=[None]*(input_dims+1), name=feature['name'] + '_placeholder')
        placeholders[feature['name']] = placeholder
        embedded_features.append(create_embedding_lookup(placeholder, feature))
    return concat_tensors_list(embedded_features)

