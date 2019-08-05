import numpy as np
from typing import Iterable
from gensim.models.keyedvectors import Word2VecKeyedVectors

from derek.common.feature_extraction.features_meta import BasicFeaturesMeta
from derek.data.model import Document


def collect_chars_set(docs: Iterable[Document]):
    chars = set()
    for doc in docs:
        for token in doc.tokens:
            chars.update(token)
    return chars


def extract_tokens(docs: Iterable[Document], preprocessor=None, model: Word2VecKeyedVectors = None, trainable=False):
    if not trainable:
        return set(model.vocab)

    tokens = set()

    for doc in docs:
        tokens.update(doc.tokens)

    if preprocessor is not None:
        tokens = {preprocessor(token) for token in tokens}

    if model is not None:
        # same preprocessing is assumed to be done for provided model
        tokens.update(model.vocab)

    return tokens


def _get_random_embedding(vector_size):
    # Ma Hovy initialisation
    return np.random.uniform(-np.sqrt(3 / vector_size), np.sqrt(3 / vector_size), vector_size)


def init_vectors(token_dict: dict, vector_size, model: Word2VecKeyedVectors = None, trainable=False):
    ret = np.zeros([len(token_dict), vector_size], dtype=float)

    # sorting is applied to ensure reproducibility of results
    for token, idx in sorted(token_dict.items(), key=lambda x: str(x[0])):

        model_vector = None

        if model is not None and token in model:
            model_vector = model[token]

        if model_vector is not None:
            ret[idx] = model_vector
        elif trainable:
            ret[idx] = _get_random_embedding(vector_size)
        # else we leave 0-s initialization

    return ret


def collect_feature_labels(docs: Iterable[Document], feature: str):
    label_set = set()

    for doc in docs:
        label_set.update(doc.token_features[feature])

    return label_set


def collect_entities_types(docs, extras=False):
    ret = set()
    for doc in docs:
        for ent in (doc.extras['ne'] if extras else doc.entities):
            ret.add(ent.type)
    return ret


def get_categorical_meta_converters(features_config):
    return BasicFeaturesMeta(*init_categorical_features(features_config)),\
        get_converters_from_features_config(features_config)


def init_categorical_features(features_config: dict):
    embedded_features = []
    one_hot_features = []

    # sorting is applied to ensure reproducibility of results
    for feature, feature_dict in sorted(features_config.items(), key=lambda x: str(x[0])):
        size = len(feature_dict['converter'])

        ret = {"name": feature, "size": size}

        if 'embedding_size' in feature_dict:
            ret['embedding_size'] = feature_dict['embedding_size']
            embedded_features.append(ret)
        else:
            one_hot_features.append(ret)

    return embedded_features, one_hot_features


def get_converters_from_features_config(features_config):
    return {key: value["converter"] for key, value in features_config.items()}

