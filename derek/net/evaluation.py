from typing import List

from derek.data.model import Entity
from derek.ner.evaluation import evaluate as ner_evaluate


class _NERLikeClassifier:
    def __init__(self, net_classifier):
        self.net_classifier = net_classifier

    def predict_doc(self, doc) -> List[Entity]:
        ret = []
        entities_predictions = self.net_classifier.predict_doc(doc)

        for ent in doc.extras["ne"]:
            prediction = entities_predictions[ent]
            if prediction is not None:
                ret.append(ent.with_type(prediction))

        return ret


def evaluate(classifier, docs, need_stats):
    return ner_evaluate(_NERLikeClassifier(classifier), docs, need_stats)
