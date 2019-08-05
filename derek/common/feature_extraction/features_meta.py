class TokenFeaturesMeta:
    def __init__(self, we_meta, basic_meta, char_meta):
        self.we_meta = we_meta
        self.basic_meta = basic_meta
        self.char_meta = char_meta

    def get_embedded_features(self):
        return self.basic_meta.get_embedded_features()

    def get_one_hot_features(self):
        return self.basic_meta.get_one_hot_features()

    def get_vectorized_features(self):
        return self.basic_meta.get_vectorized_features()

    def get_precomputed_features(self):
        return self.we_meta.get_precomputed_features()

    def get_char_features(self):
        return self.char_meta.get_char_features()

    def namespaced(self, namespace):
        return TokenFeaturesMeta(self.we_meta.namespaced(namespace),
                                 self.basic_meta.namespaced(namespace),
                                 self.char_meta.namespaced(namespace))

    def __repr__(self):
        return '\n'.join((repr(self.basic_meta), repr(self.we_meta), repr(self.char_meta)))


class BasicFeaturesMeta:
    def __init__(self, embedded_features, one_hot_features, vectorized_features=None):
        self.embedded_features = embedded_features
        self.one_hot_features = one_hot_features
        self.vectorized_features = vectorized_features if vectorized_features is not None else []

    def get_embedded_features(self):
        return self.embedded_features

    def get_one_hot_features(self):
        return self.one_hot_features

    def get_vectorized_features(self):
        return self.vectorized_features

    def namespaced(self, namespace):
        return BasicFeaturesMeta(_namespaced_features(self.embedded_features, namespace),
                                 _namespaced_features(self.one_hot_features, namespace),
                                 _namespaced_features(self.vectorized_features, namespace))

    def __repr__(self):
        return '\n'.join([f"Embedded features: {self.embedded_features}",
                          f"One hot features: {self.one_hot_features}",
                          f"Vectorized features: {self.vectorized_features}"])

    def __add__(self, other):
        if not isinstance(other, BasicFeaturesMeta):
            raise Exception("+ works only with BasicFeaturesMeta instances")

        return BasicFeaturesMeta(self.embedded_features + other.embedded_features,
                                 self.one_hot_features + other.one_hot_features,
                                 self.vectorized_features + other.vectorized_features)


class WordEmbeddingsMeta:
    def __init__(self, precomputed_features):
        self.precomputed_features = precomputed_features

    def get_precomputed_features(self):
        return self.precomputed_features

    def namespaced(self, namespace):
        return WordEmbeddingsMeta(_namespaced_features(self.precomputed_features, namespace))

    def __repr__(self):
        return "{} word embedding models".format(len(self.precomputed_features))


class CharsFeaturesMeta:
    def __init__(self, char_features):
        self.char_features = char_features

    def get_char_features(self):
        return self.char_features

    def namespaced(self, namespace):
        return CharsFeaturesMeta(_namespaced_features(self.char_features, namespace))

    def __repr__(self):
        return f"Char features: {self.char_features}"


def get_empty_basic_meta():
    return BasicFeaturesMeta([], [])


def _namespaced_features(features, namespace):
    ret = []
    for feature in features:
        new_feature = dict(feature)
        new_feature['name'] = namespace + '_' + feature['name']
        ret.append(new_feature)
    return ret
