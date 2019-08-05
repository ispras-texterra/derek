from typing import List

from derek.data.model import Entity, Relation


class CoreferenceChain:
    def __init__(self, entities: List[Entity]):
        self.entities = tuple(sorted(entities))

    def __eq__(self, other):
        if isinstance(other, CoreferenceChain):
            return other.entities == self.entities
        return False

    def __hash__(self):
        return hash(self.entities)

    def to_relations(self, strategy: str) -> List[Relation]:
        if strategy == 'chain':
            return self.to_relations_chain()
        elif strategy == 'set':
            return self.to_relations_set()
        raise ValueError("Unknown strategy!")

    def to_relations_chain(self) -> List[Relation]:
        relations = []
        for prev_entity, entity in zip(self.entities[:-1], self.entities[1:]):
            relations.append(Relation(prev_entity, entity, "COREF"))
        return relations

    def to_relations_set(self) -> List[Relation]:
        relations = []
        for i, entity in enumerate(self.entities):
            for next_entity in self.entities[i+1:]:
                relations.append(Relation(entity, next_entity, "COREF"))
        return relations

