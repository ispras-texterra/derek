from numbers import Number
from typing import Union, Dict, Iterable, Set, Tuple
from warnings import warn
import numpy as np

from derek.data.collapser_helper import build_borders_dict, create_objects_with_new_borders, \
    shift_borders_after_collapse
from derek.data.model import Entity, Relation, Document, SortedSpansSet
from derek.data.transformers import DocumentTransformer


class EntitiesCollapser(DocumentTransformer):
    def __init__(self, entity_types_to_collapse: Iterable[str], collapse_with_ne=False):
        self.__entity_types_to_collapse = frozenset(entity_types_to_collapse)
        self.__collapse_with_ne = collapse_with_ne

        if not self.__entity_types_to_collapse:
            warn("No types to collapse, doc won't be changed")

    def transform(self, doc: Document) -> Document:
        doc, mapping = self.__transform(doc)
        return doc

    def transform_with_mapping(self, doc: Document) -> Tuple[Document, Dict[Entity, Entity]]:
        return self.__transform(doc)

    def __transform(self, doc: Document) -> Tuple[Document, Dict[Entity, Entity]]:
        if self.__collapse_with_ne:
            ents = doc.extras["ne"]
        else:
            ents = doc.entities

        if not self.__entity_types_to_collapse:
            return doc, {e: e for e in ents}

        return _collapse_entities_in_doc(doc, ents, self.__entity_types_to_collapse)

    def __str__(self):
        return f"{', '.join(sorted(self.__entity_types_to_collapse))} types "\
            f"{'NEs' if self.__collapse_with_ne else 'entities'} collapser"

    @classmethod
    def from_props(cls, props: dict):
        return cls(props.get("types_to_collapse", frozenset()), props.get("collapse_with_ne", False))


def _collapse_entities_in_doc(
        doc, entities_to_collapse: Iterable[Entity], entity_types_to_collapse: Union[set, frozenset]):

    if set(doc.extras.keys()).difference({"ne"}):
        raise Exception("Currently support only ne extras")

    # copy features not to affect default document
    tokens_to_process = list(doc.tokens)
    token_features_to_process = {k: list(v) for k, v in doc.token_features.items()}

    borders_to_change = {
        'entities_to_collapse': build_borders_dict(entities_to_collapse),
        'sentences': build_borders_dict(doc.sentences)
    }
    try:
        borders_to_change["entities"] = build_borders_dict(doc.entities)
    except ValueError:
        pass

    if "ne" in doc.extras:
        borders_to_change["ne"] = build_borders_dict(doc.extras["ne"])

    _collapse_entities_and_correct_features(
        entities_to_collapse, tokens_to_process, token_features_to_process,
        entity_types_to_collapse, borders_to_change)

    sentences_mapping = create_objects_with_new_borders(doc.sentences, borders_to_change['sentences'])
    collapsed_entities_mapping = create_objects_with_new_borders(
        entities_to_collapse, borders_to_change['entities_to_collapse'])

    if 'entities' in borders_to_change:
        doc_entities_mapping = create_objects_with_new_borders(doc.entities, borders_to_change['entities'])
        doc_entities = doc_entities_mapping.values()
    else:
        doc_entities = None

    if "ne" in doc.extras:
        ne_mapping = create_objects_with_new_borders(doc.extras["ne"], borders_to_change["ne"])
        extras = {"ne": SortedSpansSet(ne_mapping.values())}
    else:
        extras = None

    doc_to_process = Document(
        doc.name, tokens_to_process, sentences_mapping.values(), doc.paragraphs, doc_entities,
        token_features=token_features_to_process, extras=extras)

    try:
        relations = [Relation(doc_entities_mapping[r.first_entity], doc_entities_mapping[r.second_entity], r.type)
                     for r in doc.relations]
        doc_to_process = doc_to_process.with_relations(relations)
    except ValueError:
        pass

    return doc_to_process, collapsed_entities_mapping


def _collapse_entities_and_correct_features(
        entities_to_process, tokens_to_process, token_features_to_process, entity_types_to_collapse, new_borders):

    for ent in _get_entities_to_collapse(entities_to_process, entity_types_to_collapse):
        _collapse_nested_entities(ent, tokens_to_process, token_features_to_process, new_borders)


def _collapse_nested_entities(max_including_entity, tokens_to_process, token_features_to_process, new_borders):
    entity_start, entity_end = new_borders['entities_to_collapse'][max_including_entity]
    new_entity_type = max_including_entity.type
    _process_token_features(token_features_to_process, entity_start, entity_end, new_entity_type)

    del tokens_to_process[entity_start + 1:entity_end]

    for val in new_borders.values():
        shift_borders_after_collapse(val, entity_start, entity_end)

    tokens_to_process[entity_start] = f"${new_entity_type}$"


def _process_token_features(token_features: Dict[str, list], entity_start, entity_end, entity_type):
    for key, val in token_features.items():
        elem_types = set(type(v) for v in val)

        if len(elem_types) == 0:
            raise Exception(f"Empty {key} token features list")

        if len(elem_types) > 1:
            raise Exception(f"{key} token features have different types: {elem_types}")

        elem_type = next(iter(elem_types))

        if issubclass(elem_type, Number):
            replacement = 0
        elif issubclass(elem_type, dict):
            replacement = dict()
        elif issubclass(elem_type, str):
            replacement = f"${entity_type}$"
        elif issubclass(elem_type, np.ndarray):
            replacement = np.zeros_like(val[0])
        else:
            raise Exception(f"Collapsing of {elem_type} features is not supported")

        val[entity_start: entity_end] = [replacement]


def _get_entities_to_collapse(entities: Iterable[Entity], types_to_collapse: Set[str]):
    entities_to_collapse = []
    candidate_entities = [e for e in entities if e.type in types_to_collapse]
    processed_entities = set()
    types_to_collapse = sorted(types_to_collapse)  # sort to choose determined type for entities at same span

    for entity in candidate_entities:
        if entity in processed_entities:
            continue

        nested = set()
        at_same_span = set()

        for other_entity in candidate_entities:
            if entity.coincides(other_entity):
                at_same_span.add(other_entity)
                continue

            if entity.contains(other_entity):
                nested.add(other_entity)

            if other_entity.contains(entity):
                break
        else:
            processed_entities.update(at_same_span)
            processed_entities.update(nested)
            for t in types_to_collapse:
                for ent in at_same_span:
                    if ent.type == t:
                        entities_to_collapse.append(ent)
                        break

    return entities_to_collapse
