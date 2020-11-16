from typing import List, Dict, Optional

from derek.common.io import get_batch_size
from derek.data.model import Entity, Document
from derek.ner.evaluation import evaluate as ner_evaluate


class _NERLikeClassifier:
    _PREDICTION_BATCH_SIZE = get_batch_size()

    def __init__(self, net_classifier):
        self.net_classifier = net_classifier

    def predict_docs(self, docs: List[Document], batch_size: int = _PREDICTION_BATCH_SIZE) -> List[List[Entity]]:
        predictions = self.net_classifier.predict_docs(docs, batch_size)
        return [self._type_entities(doc.extras["ne"], ent_typing) for doc, ent_typing in zip(docs, predictions)]

    def predict_doc(self, doc: Document) -> List[Entity]:
        return self.predict_docs([doc])[0]

    @staticmethod
    def _type_entities(entities: List[Entity], entities_typing: Dict[Entity, Optional[str]]):
        ret = []
        for ent in entities:
            prediction = entities_typing[ent]
            if prediction is not None:
                ret.append(ent.with_type(prediction))

        return ret


def evaluate(classifier, docs, need_stats):
    return ner_evaluate(_NERLikeClassifier(classifier), docs, need_stats)
