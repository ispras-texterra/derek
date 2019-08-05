from collections import defaultdict

from derek.data.model import Document


class ComplexFilter:
    def __init__(self, filters):
        self.filters = filters

    def apply(self, pairs, doc):
        return [pair for pair in pairs if not self._check(pair[0], pair[1], doc)]

    def _check(self, e1, e2, doc):
        return any(x.check(e1, e2, doc) for x in self.filters)


class ComplexClassifier:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def apply(self, doc: Document):
        scores = {}
        for i, e1 in enumerate(doc.entities):
            for e2 in doc.entities[i+1:]:
                for classifier in self.classifiers:
                    if classifier.check(e1, e2, doc):
                        label = classifier.get_label(e1, e2, doc)
                        scores[(e1, e2)] = defaultdict(float)
                        scores[(e1, e2)][label] = 1
                        break
        return scores


class ExactMatchPostprocessor:
    def check(self, e1, e2, doc):
        if e1 == e2 or e1.type != 'noun' or e2.type != 'noun':
            return False
        e1_words = doc.token_features['lemmas'][e1.start_token: e1.end_token]
        e2_words = doc.token_features['lemmas'][e2.start_token: e2.end_token]
        return set(e1_words) == set(e2_words)

    def get_label(self, e1, e2, doc):
        return "COREF"


class IntersectingMentionsPostprocessor:
    def check(self, e1, e2, doc):
        return e2.intersects(e1)

    def get_label(self, e1, e2, doc):
        return None


CLASSIFIERS = {
    'exact_match': ExactMatchPostprocessor(),
    'intersecting_mentions': IntersectingMentionsPostprocessor()
}
