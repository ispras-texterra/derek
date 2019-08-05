from collections import namedtuple

from derek.common.feature_extraction.features_meta import BasicFeaturesMeta, get_empty_basic_meta


class AttentionFeaturesMeta:
    def __init__(self, token_features_meta: BasicFeaturesMeta, relation_features_meta: BasicFeaturesMeta):
        self.token_features_meta = token_features_meta
        self.relation_features_meta = relation_features_meta

    def get_token_features_meta(self):
        return self.token_features_meta

    def get_relation_features_meta(self):
        return self.relation_features_meta

    def __repr__(self):
        return "Token level features:\n{}\nRelation level features:\n{}".format(
            self.token_features_meta, self.relation_features_meta)


Metas = namedtuple('Metas', ['encoder', 'attention', 'classifier'])


def get_empty_attention_meta():
    return AttentionFeaturesMeta(get_empty_basic_meta(), get_empty_basic_meta())
