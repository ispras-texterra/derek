from typing import Sized

from gensim.models import KeyedVectors

import numpy as np


class WordEmbeddingConverter:
    def __init__(self, model):
        binary_flag = model.get("binary", True)
        self.lower_flag = model.get("lower", False)
        self.model = KeyedVectors.load_word2vec_format(model["path"], binary=binary_flag, datatype=float)
        self.oov = np.zeros(self.model.vector_size, dtype=float)

    def __getitem__(self, word):
        if self.lower_flag:
            word = word.lower()
        if word in self.model.vocab:
            return self.model[word]
        return self.oov

    def __len__(self):
        return self.model.vector_size


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
