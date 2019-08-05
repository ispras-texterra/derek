from typing import Dict, Set, List

from derek.data.model import Document, SortedSpansSet


def _merge_pattern_to_replacements(merging_pattern: Dict[str, List[str]]) -> Dict[str, str]:
    replacements = {}

    for key, values in merging_pattern.items():
        for val in values:
            if val in replacements:
                raise Exception(f"Replacement for {val} was already provided")
            replacements[val] = key

    return replacements


class NERPreprocessor:
    def __init__(self, types_to_filter: Set[str], replacements: Dict[str, str]):
        self.__filter = types_to_filter
        self.__replacements = replacements

    def process_doc(self, doc: Document) -> Document:
        new_entities = []

        for ent in doc.entities:
            if ent.type in self.__filter:
                continue

            new_entities.append(ent.with_type(self.__replacements.get(ent.type, ent.type)))

        return doc.without_relations().without_entities().with_entities(new_entities)

    @classmethod
    def from_props(cls, props: dict):
        return cls(set(props.get("ent_types_to_filter", [])),
                   _merge_pattern_to_replacements(props.get("ent_types_merge_pattern", {})))


class NETPreprocessor:
    def __init__(
            self, filter_types: Set[str],
            replacements_for_ne: Dict[str, str], replacements_for_ents: Dict[str, str]):
        self.__filter = filter_types
        self.__ne_replacements = replacements_for_ne
        self.__ents_replacements = replacements_for_ents

    def process_doc(self, doc: Document) -> Document:
        new_entities = []
        nes = []

        for ent in doc.entities:
            if ent.type in self.__filter:
                continue

            new_ne_type = self.__ne_replacements.get(ent.type, ent.type)
            new_ent_type = self.__ents_replacements.get(ent.type, ent.type)

            new_entities.append(ent.with_type(new_ent_type))
            nes.append(ent.with_type(new_ne_type))

        return doc.without_relations().without_entities().with_entities(new_entities). \
            with_additional_extras({"ne": SortedSpansSet(nes)})

    @classmethod
    def from_props(cls, props: dict):
        return cls(set(props.get("ent_types_to_filter", [])),
                   _merge_pattern_to_replacements(props.get("ne_types_merge_pattern", {})),
                   _merge_pattern_to_replacements(props.get("ent_types_merge_pattern", {})))


__PREPROCESSOR_FACTORIES = {"ner": NERPreprocessor.from_props, "net": NETPreprocessor.from_props}


def preprocessor_for(task_name: str, props: dict):
    if task_name not in __PREPROCESSOR_FACTORIES:
        raise Exception(f"Preprocessing for {task_name} is not currently supported")
    return __PREPROCESSOR_FACTORIES[task_name](props)
