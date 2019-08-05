from typing import Iterable
from warnings import warn
from random import random

from derek.data.model import Document, Entity, SortedSpansSet
from derek.data.processing_helper import QUOTES
from derek.data.collapser_helper import shift_borders_after_collapse, build_borders_dict, \
    create_objects_with_new_borders
from derek.data.transformers import DocumentTransformer


class EntitiesUnquoteAugmentor(DocumentTransformer):
    def __init__(self, probability: float, types_to_unquote: Iterable[str]):
        if not 0.0 <= probability <= 1.0:
            raise Exception("Augment prob must be in [0.0, 1.0]")

        self.__prob = probability
        self.__types = frozenset(types_to_unquote)

        if not self.__types:
            warn("No types provided to unquote")

    def transform(self, doc: Document) -> Document:
        if self.__prob == 0.0 or not self.__types:
            return doc

        borders_dicts = {
            'entities': build_borders_dict(doc.entities),
            'sentences': build_borders_dict(doc.sentences),
        }
        if 'ne' in doc.extras:
            borders_dicts['ne'] = build_borders_dict(doc.extras["ne"])

        if set(doc.extras.keys()).difference({"ne"}):
            raise Exception("Can only work with ne extras")

        quotes_idx = set()

        for ent in doc.entities:
            if ent.type not in self.__types or not self.__quoted(doc, ent):
                continue

            if random() < 1.0 - self.__prob:
                continue

            quotes_idx.add(ent.start_token - 1)
            quotes_idx.add(ent.end_token)

            ent_shifted_start, ent_shifted_end = borders_dicts["entities"][ent]
            for key, val in borders_dicts.items():
                shift_borders_after_collapse(val, ent_shifted_start - 1, ent_shifted_start, new_length=0)
                # shift second quote span after first quote replacement
                shift_borders_after_collapse(val, ent_shifted_end - 1, ent_shifted_end, new_length=0)

        new_tokens = [tok for idx, tok in enumerate(doc.tokens) if idx not in quotes_idx]
        new_sentences = create_objects_with_new_borders(doc.sentences, borders_dicts["sentences"])
        new_entities = create_objects_with_new_borders(doc.entities, borders_dicts["entities"])

        if "ne" in doc.extras:
            new_extras = {
                "ne": SortedSpansSet(create_objects_with_new_borders(doc.extras["ne"], borders_dicts["ne"]).values())
            }
        else:
            new_extras = None

        new_token_features = {
            k: [v for idx, v in enumerate(val) if idx not in quotes_idx] for k, val in doc.token_features.items()}

        return Document(doc.name, new_tokens, new_sentences.values(), doc.paragraphs, new_entities.values(),
                        token_features=new_token_features, extras=new_extras)

    @staticmethod
    def __quoted(doc: Document, ent: Entity):
        return ent.start_token > 0 and ent.end_token < len(doc.tokens) \
               and doc.tokens[ent.start_token - 1] in QUOTES and doc.tokens[ent.end_token] in QUOTES
