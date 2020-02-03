from logging import getLogger
from typing import Iterable, List
from warnings import warn

import numpy as np

from derek.common.feature_extraction.converters import create_unsigned_integers_converter, \
    create_signed_integers_converter, create_categorical_converter
from derek.common.feature_extraction.embeddings import embedding_readers
from derek.common.feature_extraction.feature_extractor import TokenFeatureExtractor
from derek.common.feature_extraction.features_meta import TokenFeaturesMeta, WordEmbeddingsMeta, BasicFeaturesMeta, \
    CharsFeaturesMeta
from derek.common.feature_extraction.gazetteer_feature_extractor import generate_gazetteers_feature_extractors
from derek.common.feature_extraction.helper import Direction
from derek.common.feature_extraction.factory_helper import extract_tokens, init_vectors, init_categorical_features, \
    collect_chars_set, collect_feature_labels, get_categorical_meta_converters
from derek.data.model import Document
from derek.data.processing_helper import StandardTokenProcessor

logger = getLogger('logger')
DEFAULT_FEATS_LIST = ['Gender', 'Animacy', 'Tense', 'Variant', 'Aspect', 'Mood', 'Polarity', 'Voice', 'Number',
                      'VerbForm', 'Foreign', 'Case', 'Person', 'Degree']


def generate_token_feature_extractor(docs: Iterable[Document], props: dict, char_padding_size: int = 0):
    we_converters_preprocessors, we_meta = _init_we_features(docs, props)

    morph_features = props.get('morph_feats_list', DEFAULT_FEATS_LIST)
    word_meta, word_converters = get_categorical_meta_converters(
        _init_word_level_features(docs, props, morph_features))

    gazetteer_meta, gazetteer_fes = generate_gazetteers_feature_extractors(props)
    word_meta += gazetteer_meta

    vectors_keys = props.get('vectors_keys', [])
    if len(vectors_keys) != len(set(vectors_keys)):
        raise Exception('"vectors_keys" should not contain duplicates')
    vectorized_features = _init_vectorized_features(docs, vectors_keys)

    if not vectorized_features and not we_converters_preprocessors:
        warn("Neither word embeddings nor vectorized features were specified")

    word_meta += BasicFeaturesMeta([], [], vectorized_features)

    char_level_features = _init_char_level_features(docs, props, char_padding_size)
    # We assume that char features are only embedded features
    char_features, _ = init_categorical_features(char_level_features)

    return TokenFeatureExtractor(
        we_converters_preprocessors, word_converters, char_level_features, vectors_keys, gazetteer_fes), \
        TokenFeaturesMeta(we_meta, word_meta, CharsFeaturesMeta(char_features))


def _init_we_features(docs: Iterable[Document], props: dict):
    we_converters_preprocessors = {}
    precomputed_features = []

    i = 0
    for model_config in props.get('models', []):
        reader_type = model_config.get("type", "w2v")
        ignore_errors = model_config.get("ignore_utf_errors", False)
        reader = embedding_readers[reader_type](errors='ignore' if ignore_errors else 'strict')

        logger.info(f"Loading {reader_type} model...")
        name = f'words_{i}'

        we_model = reader.read(model_config["path"])
        trainable = model_config.get("trainable", False)
        preprocessor = StandardTokenProcessor.from_props(model_config)

        tokens_set = extract_tokens(docs, preprocessor, we_model, trainable)
        converter = create_categorical_converter(tokens_set, has_oov=True)
        vectors = init_vectors(converter, we_model.vector_size, we_model, trainable)

        we_converters_preprocessors[name] = (converter, preprocessor)
        precomputed_features.append({
            'name': name,
            'vectors': vectors,
            'trainable': trainable
        })

        logger.info("Initialised embeddings ({}, {})".format(vectors.shape[0], vectors.shape[1]))
        i += 1

    if props.get("internal_emb_size", 0) != 0:
        name = f'words_{i}'

        doc_tokens = extract_tokens(docs, trainable=True)
        converter = create_categorical_converter(doc_tokens, has_oov=True)
        vectors = init_vectors(converter, props["internal_emb_size"], trainable=True)

        we_converters_preprocessors[name] = (converter, None)
        precomputed_features.append({
            'name': name,
            'vectors': vectors,
            'trainable': True
        })

    return we_converters_preprocessors, WordEmbeddingsMeta(precomputed_features)


def _init_char_level_features(docs: Iterable[Document], props: dict, char_padding_size: int):
    char_level_features = {}
    if props.get("char_embedding_size", -1) > 0:
        chars = collect_chars_set(docs)
        char_level_features['chars'] = {
            'converter': create_categorical_converter(chars, has_oov=True),
            'embedding_size': props["char_embedding_size"],
            'padding_size': char_padding_size
        }
    return char_level_features


def _init_word_level_features(docs: Iterable[Document], props: dict, morph_features: List[str]):
    features = {}

    if props.get("pos_emb_size", -1) >= 0:
        pos_types = collect_feature_labels(docs, "pos")
        features['pos'] = {
            'converter': create_categorical_converter(pos_types, has_oov=True)
        }
        if props["pos_emb_size"] != 0:
            features['pos']['embedding_size'] = props["pos_emb_size"]

    if props.get("borders_size", -1) >= 0:
        features['borders'] = {
            'converter': create_categorical_converter({'start', 'in', 'end'})
        }
        if props["borders_size"] != 0:
            features["borders"]['embedding_size'] = props["borders_size"]

    if props.get("dt_label_emb_size", -1) >= 0:
        dt_label_types = collect_feature_labels(docs, "dt_labels")
        features['dt_labels'] = {
            'converter': create_categorical_converter(dt_label_types, has_oov=True)
        }
        if props["dt_label_emb_size"] != 0:
            features["dt_labels"]['embedding_size'] = props["dt_label_emb_size"]

    if props.get("dt_distance_emb_size", -1) >= 0:
        features['dt_head_distances'] = {
            'converter': create_signed_integers_converter(props["max_dt_distance"])
        }
        if props["dt_distance_emb_size"] != 0:
            features["dt_head_distances"]['embedding_size'] = props["dt_distance_emb_size"]

    if props.get("dt_depth_emb_size", -1) >= 0:
        features['dt_depths'] = {
            'converter': create_unsigned_integers_converter(props["max_dt_depth"])
        }
        if props["dt_depth_emb_size"] != 0:
            features["dt_depths"]['embedding_size'] = props["dt_depth_emb_size"]

    max_dt_delta = props.get("max_dt_delta", 0)
    if max_dt_delta:
        for direction in [Direction.FORWARD, Direction.BACKWARD]:
            key = "dt_deltas_" + direction.value
            emb_size = props.get(key + "_emb_size", -1)
            if emb_size >= 0:
                features[key] = {
                    'converter': create_signed_integers_converter(max_dt_delta, additional_labels={"$START$"})
                }

                if emb_size != 0:
                    features[key]['embedding_size'] = emb_size

    for direction in [Direction.FORWARD, Direction.BACKWARD]:
        key = "dt_breakups_" + direction.value
        emb_size = props.get(key + "_emb_size", -1)
        if emb_size >= 0:
            features[key] = {
                'converter': create_categorical_converter(collect_feature_labels(docs, key))
            }

            if emb_size != 0:
                features[key]['embedding_size'] = emb_size

    if props.get('morph_feats_emb_size', -1) >= 0:
        for feat in morph_features:
            feat_types = collect_feature_labels(docs, feat)
            if not feat_types:
                continue
            features[feat] = {
                'converter': create_categorical_converter(feat_types, has_oov=True)
            }
            if props["morph_feats_emb_size"] != 0:
                features[feat]['embedding_size'] = props["morph_feats_emb_size"]

    return features


def _init_vectorized_features(docs: Iterable[Document], vectors_keys):
    first_doc = next(iter(docs))
    ret = []
    for key in vectors_keys:
        # get first document first token vector to set size of provided vectors
        vector = first_doc.token_features[key][0]
        if not isinstance(vector, np.ndarray):
            raise Exception(f'"{key}" is not vector feature')
        ret.append({"name": key, "size": len(vector)})
    return ret
