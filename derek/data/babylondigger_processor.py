from typing import Dict, Callable, Any
from collections import defaultdict
from warnings import warn


from babylondigger.processor import DocumentProcessorInteface
from babylondigger.datamodel import Document, Sentence, Token, NavigableToken
from derek.data.transformers import TokenFeaturesProvider


class BabylonDiggerProcessor(TokenFeaturesProvider):
    def __init__(self, processor: DocumentProcessorInteface, token_fes: Dict[str, Callable[[int, NavigableToken], Any]]):
        self.__processor_mngr = processor
        self.__token_fes = token_fes

    def __enter__(self):
        self.__processor = self.__processor_mngr.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__processor = None
        self.__processor_mngr.__exit__(exc_type, exc_val, exc_tb)

    def get_token_features(self, tokens, sentences):
        doc = _build_digger_document(tokens, sentences)
        doc = next(self.__processor.process([doc]))
        # Following code is quite similar to tools.doc_converters.digger_converter.convert_from_digger_to_derek
        # This implementation is more flexible.
        # We should unify babylondigger <-> derek documents converting in future.
        token_features = defaultdict(list)
        for i, token in enumerate(doc.tokens):
            for feat_name, feat_extractor in self.__token_fes.items():
                token_features[feat_name].append(feat_extractor(i, token))
        return filter_empty_features(token_features)

    @classmethod
    def from_processor(cls, processor: DocumentProcessorInteface, extra_fes: Dict[str, Callable[[int, NavigableToken], Any]] = None):
        fes = {
            'pos': lambda _, token: (token.pos.upos if token.pos else None),
            'dt_labels': lambda _, token: token.deprel,
            'dt_head_distances': lambda i, token:
                    ((token.head_index - i if token.head_index != -1 else 0) if token.head_index is not None else None),
            'lemmas': lambda _, token: token.lemma,
            'feats': lambda _, token: (token.pos.feats if token.pos else None)
        }
        if extra_fes:
            fes.update(extra_fes)
        return cls(processor, fes)


def filter_empty_features(token_features):
    result = {}
    for key, value in token_features.items():
        if all(x is None for x in value):
            continue
        else:
            if any(x is None for x in value):
                warn(f"'{key}' feature contains one or more None values")
            result[key] = value
    return result


def _build_digger_document(tokens, sentences):
    # This method is quite similar to tools.doc_converters.digger_converter.convert_from_derek_to_digger
    # But this method have no need to build derek document as we assume all features to be computed in this wrapper
    # We should unify babylondigger <-> derek documents converting in future.
    text = ' '.join(tokens)
    digger_tokens = []
    pointer = 0
    for token in tokens:
        digger_tokens.append(Token(pointer, pointer + len(token)))
        pointer += len(token) + 1
    digger_sentences = [Sentence(s.start_token, s.end_token) for s in sentences]
    return Document(text, digger_tokens, digger_sentences)


def digger_extras_extractor(key):
    return lambda _, token: (token.extras[key] if key in token.extras else None)
