from abc import abstractmethod, ABCMeta
from logging import getLogger
from os.path import exists
from typing import Dict, FrozenSet, Optional, Iterable

from numpy import ndarray, zeros

logger = getLogger('logger')


class EmbeddingsModel:
    def __init__(self, word2idx: Dict[str, int], vectors_matrice: ndarray):
        assert len(word2idx) > 0, "vocab can't be empty"
        assert len(word2idx) == vectors_matrice.shape[0], "vocab size must match with vectors matrice shape"

        self._w2idx = word2idx
        self._vectors_matrice = vectors_matrice

    @property
    def vector_size(self) -> int:
        return self._vectors_matrice.shape[1]

    @property
    def vocab(self) -> FrozenSet:
        return frozenset(self._w2idx)

    def get_vector_for_token(self, token: str) -> Optional[ndarray]:
        if token not in self._w2idx:
            return None

        return self._vectors_matrice[self._w2idx[token]]


class EmbeddingsReader(metaclass=ABCMeta):
    @abstractmethod
    def read(self, path: str) -> EmbeddingsModel:
        pass


def _read_w2v_strings(stream: Iterable[str], vocab_size: int, embedding_size: int):
    word2idx = {}
    matrice = zeros((vocab_size, embedding_size), dtype=float)

    for line in stream:
        line = line.strip()
        if not line:
            continue

        if len(word2idx) == vocab_size:
            logger.warning("Embeddings file have more entries than specified in the beginning, skipping rest")
            break

        splitted = line.split()
        word = splitted[0]

        if word in word2idx:
            logger.warning(f"{word} duplicate, ignoring all but first")
            continue

        embedding = [float(s) for s in splitted[1:]]

        next_idx = len(word2idx)
        word2idx[word] = next_idx
        matrice[next_idx] = embedding

    if len(word2idx) < vocab_size:
        logger.warning(f"Embeddings file have less entries than specified in the begginning: "
                       f"{len(word2idx)} instead of {vocab_size}")
        matrice = matrice[:vocab_size + 1]

    return word2idx, matrice


class Word2VecReader(EmbeddingsReader):
    def __init__(self, *, errors='strict'):
        self._errors = errors

    def read(self, path: str) -> EmbeddingsModel:
        if not exists(path):
            raise Exception(f"{path} do not exist")

        with open(path, "r", encoding="utf-8", errors=self._errors) as f:
            first_line = f.readline()
            vocab_size, embedding_size = (int(x) for x in first_line.strip().split())
            word2idx, matrice = _read_w2v_strings(f, vocab_size, embedding_size)

        return EmbeddingsModel(word2idx, matrice)


class GloveReader(EmbeddingsReader):
    def __init__(self, *, errors='strict'):
        self._errors = errors

    def read(self, path: str) -> EmbeddingsModel:
        if not exists(path):
            raise Exception(f"{path} do not exist")

        with open(path, "r", encoding="utf-8", errors=self._errors) as f:
            first_line_splitted = f.readline().strip().split()
            embedding_size = len(first_line_splitted) - 1
            vocab_size = 1 + sum(1 for line in f if line.strip())

        with open(path, "r", encoding="utf-8", errors=self._errors) as f:
            word2idx, matrice = _read_w2v_strings(f, vocab_size, embedding_size)

        return EmbeddingsModel(word2idx, matrice)


embedding_readers = {
    "w2v": Word2VecReader,
    "glove": GloveReader
}
