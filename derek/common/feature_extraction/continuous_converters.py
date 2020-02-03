from typing import Sized

import numpy as np

from derek.common.feature_extraction.embeddings import embedding_readers
from derek.data.processing_helper import StandardTokenProcessor


class WordEmbeddingConverter:
    def __init__(self, model_config):
        reader_type = model_config.get("type", "w2v")
        ignore_errors = model_config.get("ignore_utf_errors", False)
        reader = embedding_readers[reader_type](errors='ignore' if ignore_errors else 'strict')

        self._model = reader.read(model_config["path"])
        self._preprocessor = StandardTokenProcessor.from_props(model_config)
        self._oov = np.zeros(self._model.vector_size, dtype=float)

    def __getitem__(self, word):
        word = self._preprocessor(word)
        vector = self._model.get_vector_for_token(word)
        return vector if vector is not None else self._oov

    def __len__(self):
        return self._model.vector_size


class ExtrasIdentityConverter:
    def __init__(self, docs, name):
        example = next(iter(docs)).token_features[name][0]
        self._len = len(example) if isinstance(example, Sized) else 1

    def __getitem__(self, item):
        if not isinstance(item, Sized):
            item = [item]
        if len(item) != self._len:
            raise Exception('Token feats has different length!')
        return item

    def __len__(self):
        return self._len
