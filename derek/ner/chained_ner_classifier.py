from os import listdir
from os.path import isdir, join
from typing import List

from derek.data.model import Document, Entity
from derek.ner import NERClassifier
from derek.net import NETClassifier


class ChainedNERClassifier:
    def __init__(self, model_path: str):
        self.path = model_path

    def __enter__(self):
        if not self.__is_chained_model():
            self.__ner_manager, self.__net_manager = NERClassifier(self.path), None
            return self.__ner_manager.__enter__()

        self.__ner_manager = ChainedNERClassifier(join(self.path, "ner"))
        self.__ner = self.__ner_manager.__enter__()
        self.__net_manager = NETClassifier(join(self.path, "net"))
        self.__net = self.__net_manager.__enter__()

        return self

    def predict_doc(self, doc: Document) -> List[Entity]:
        entities = self.__ner.predict_doc(doc)
        entities_typing = self.__net.predict_doc(doc.with_additional_extras({"ne": entities}))
        ret = []

        for ent in entities:
            new_type = entities_typing.get(ent, None)
            if new_type is None:
                ret.append(ent)
            else:
                ret.append(ent.with_type(new_type))

        return ret

    def __exit__(self, *exc):
        if self.__net_manager is not None:
            self.__net_manager.__exit__(*exc)

        self.__ner_manager.__exit__(*exc)

        self.__net_manager, self.__ner_manager, self.__ner, self.__net = None, None, None, None

    def __is_chained_model(self):
        path_files = listdir(self.path)
        return set(path_files) == {"ner", "net"} and all(isdir(join(self.path, p)) for p in ["ner", "net"])
