from abc import ABCMeta, abstractmethod
from typing import Iterable, Dict, List, Any

from derek.data.model import Document, Sentence


class DocumentTransformer(metaclass=ABCMeta):
    def __enter__(self):
        return self

    @abstractmethod
    def transform(self, doc: Document) -> Document:
        pass

    def __exit__(self, *exc):
        pass


class SequenceDocumentTransformer(DocumentTransformer):
    def __init__(self, transformers: Iterable[DocumentTransformer]):
        self.__transformers_mngrs = tuple(transformers)

    def __enter__(self):
        self.__transformers = []
        for mngr in self.__transformers_mngrs:
            self.__transformers.append(mngr.__enter__())
        return self

    def transform(self, doc: Document) -> Document:
        for t in self.__transformers:
            doc = t.transform(doc)
        return doc

    def __exit__(self, *exc):
        self.__transformers = None
        for mngr in self.__transformers_mngrs[::-1]:
            mngr.__exit__(*exc)


class TokenFeaturesProvider(DocumentTransformer):
    def transform(self, doc: Document) -> Document:
        return doc.with_additional_token_features(self.get_doc_token_features(doc))

    def get_doc_token_features(self, doc: Document) -> Dict[str, List[Any]]:
        return self.get_token_features(doc.tokens, doc.sentences)

    @abstractmethod
    def get_token_features(self, tokens: List[str], sentences: List[Sentence]) -> Dict[str, List[Any]]:
        pass


class ExtrasProvider(DocumentTransformer):
    def transform(self, doc: Document) -> Document:
        return doc.with_additional_extras(self.get_doc_extras(doc))

    def get_doc_extras(self, doc: Document) -> Dict[str, Any]:
        return self.get_extras(doc.tokens, doc.sentences)

    @abstractmethod
    def get_extras(self, tokens: List[str], sentences: List[Sentence]) -> Dict[str, Any]:
        pass
