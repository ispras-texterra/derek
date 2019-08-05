from collections import defaultdict

from bllipparser import RerankingParser
import StanfordDependencies

from derek.data.transformers import TokenFeaturesProvider


class BLLIPProcessor(TokenFeaturesProvider):
    def __init__(self, model_name=None):
        if model_name is None:
            model_name = 'GENIA+PubMed'
        self.model_name = model_name
        self.sd = StanfordDependencies.get_instance()

    def __enter__(self):
        self.bllip = RerankingParser.fetch_and_load(self.model_name, verbose=True)
        return self

    def __exit__(self, *exc):
        self.bllip = None

    def get_token_features(self, tokens, sentences):
        token_features = defaultdict(list)

        for sent in sentences:
            sent_tokens = tokens[sent.start_token: sent.end_token]
            ptb_tree = self.bllip.simple_parse(sent_tokens)
            ud_sent = self.sd.convert_tree(ptb_tree)

            token_features['pos'] = [ud_tok.pos for ud_tok in ud_sent]
            token_features['dt_labels'] = [ud_tok.deprel for ud_tok in ud_sent]
            token_features['dt_head_distances'] =\
                [ud_tok.head - (i + 1) if ud_tok.head != 0 else 0
                 for i, ud_tok in enumerate(ud_sent)]

        return token_features

    @classmethod
    def from_props(cls, props):
        return cls(props.get('model_name', None))
