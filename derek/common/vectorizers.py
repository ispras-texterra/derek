from abc import abstractmethod, ABCMeta
from typing import List, Iterable
from contextlib import AbstractContextManager

import numpy as np

from derek.data.model import Document
from derek.data.processing_helper import StandardTokenProcessor
from derek.data.transformers import DocumentTransformer
from derek.pretraining_lms.nn.trainer import LMSClassifier


class AbstractVectorizer(AbstractContextManager, metaclass=ABCMeta):
    def __init__(self):
        self._entered = False

    def vectorize(self, doc: Document) -> List[np.array]:
        if not self._entered:
            raise Exception("Use this class only as context manager")

        return [np.array(v, dtype=float) for v in self._vectorize_doc(doc)]

    @abstractmethod
    def _vectorize_doc(self, doc: Document) -> List[np.array]:
        pass


class CompositeVectorizer(AbstractVectorizer, DocumentTransformer):
    def __init__(self, vectorizers: Iterable[AbstractVectorizer], key):
        super().__init__()
        self._vectorizers = tuple(vectorizers)
        if not self._vectorizers:
            raise Exception("No vectorizers provided")
        self._key = key

    def __enter__(self):
        for v in self._vectorizers:
            v.__enter__()

        self._entered = True
        return self

    def transform(self, doc: Document) -> Document:
        return doc.with_additional_token_features({self._key: self.vectorize(doc)})

    def _vectorize_doc(self, doc: Document) -> List[np.array]:
        doc_vectors = [v.vectorize(doc) for v in self._vectorizers]
        return [np.concatenate(vectors) for vectors in zip(*doc_vectors)]

    def __exit__(self, *exc):
        self._entered = False
        for v in self._vectorizers[::-1]:
            v.__exit__(*exc)


class PretrainVectorizer(AbstractVectorizer):
    def __init__(self, path):
        super().__init__()
        self._path = path
        self._clf_manager = LMSClassifier(self._path)
        self._clf = None

    def __enter__(self):
        self._clf = self._clf_manager.__enter__()
        self._entered = True

        return self

    def _vectorize_doc(self, doc: Document) -> List[np.array]:
        return self._clf.vectorize_doc(doc)

    @classmethod
    def from_props(cls, props: dict):
        return cls(props["path"])

    def __exit__(self, *exc):
        self._entered = False
        self._clf = None
        self._clf_manager.__exit__(*exc)


class FastTextVectorizer(AbstractVectorizer):
    def __init__(self, path, token_processor):
        super().__init__()
        self._path = path
        self._model = None
        from fasttext import load_model as _load_model
        self._load_model = _load_model
        self._token_processor = token_processor

    def __enter__(self):
        self._model = self._load_model(self._path)
        self._entered = True

        return self

    def _vectorize_doc(self, doc: Document) -> List[np.array]:
        return [self._model.get_word_vector(self._token_processor(token)) for token in doc.tokens]

    @classmethod
    def from_props(cls, props: dict):
        return cls(props["path"], StandardTokenProcessor.from_props(props))

    def __exit__(self, *exc):
        self._entered = False
        self._model = None
