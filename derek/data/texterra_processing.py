from typing import Optional, List

import texterra

from derek.data.helper import align_raw_entities
from derek.data.model import Sentence, SortedSpansSet, Entity
from derek.data.processing_helper import QUOTES
from derek.data.transformers import ExtrasProvider, TokenFeaturesProvider


def get_api(key_path: Optional[str]) -> texterra.API:
    if key_path is None:
        return texterra.API(host='http://localhost:8082/texterra/')
    else:
        with open(key_path, encoding="utf-8") as f:
            key = f.read().rstrip()
            return texterra.API(key)


class TexterraSegmentor:
    def __init__(self, texterra_key_path=None, lang=None):
        self.lang = lang
        self.texterra = get_api(texterra_key_path)

    def segment(self, raw_text):
        jsn = [{'text': raw_text}]
        params = {'targetType': ['token', 'sentence']}
        if self.lang is not None:
            params['language'] = self.lang

        response, = self.texterra.custom_query('nlp', params, json=jsn)
        if "token" not in response["annotations"]:
            return [], [], []

        token_annotations, sentence_annotations = response["annotations"]["token"], response["annotations"]["sentence"]
        raw_tokens = [(t["start"], t["end"]) for t in token_annotations]
        raw_sentences = [(t["start"], t["end"]) for t in sentence_annotations]
        tokens, sentences = self._process_tokens_and_sentences(raw_text, raw_tokens, raw_sentences)

        return tokens, sentences, raw_tokens

    @staticmethod
    def _process_tokens_and_sentences(raw_text, raw_tokens, raw_sentences):
        cur_sent_idx = 0
        sent_start = 0
        tokens, sentences = [], []

        for token in raw_tokens:
            tokens.append(raw_text[token[0]: token[1]])
            if raw_sentences[cur_sent_idx][1] <= token[1]:
                sentences.append(Sentence(sent_start, len(tokens)))
                sent_start = len(tokens)
                cur_sent_idx += 1

        if sent_start < len(tokens):
            sentences.append(Sentence(sent_start, len(tokens)))

        return tokens, sentences


class TexterraTextProcessor(TokenFeaturesProvider):
    def __init__(self, texterra_key_path=None, lang=None):
        self.lang = lang if lang is not None else ''
        self.texterra = get_api(texterra_key_path)

    def get_token_features(self, tokens, sentences):
        sents, _, _ = _get_space_joined_sentences(tokens, sentences)

        pos_sents = list(self.texterra.pos_tagging(sents, language=self.lang))
        pos_tags = _get_annotations(pos_sents, sents, None)

        dt_labels = []
        dt_head_distances = []
        syntax_trees = list(self.texterra.syntax_detection(sents, language=self.lang))
        for sent_tree in syntax_trees:
            for i in range(1, len(sent_tree.labels)):  # None is first label in dependency tree
                label = sent_tree.labels[i]
                distance = sent_tree.heads[i] - i
                dt_labels.append(label)
                dt_head_distances.append(0 if label == "ROOT" else distance)

        token_features = {
            'pos': pos_tags,
            'dt_labels': dt_labels,
            'dt_head_distances': dt_head_distances,
        }
        return token_features

    @classmethod
    def from_props(cls, props):
        return cls(props.get("key_path", None), props.get("lang", None))


class TexterraNerExtrasProvider(ExtrasProvider):
    def __init__(self, texterra_key_path=None, lang=None, remove_quotes=False):
        self.lang = lang if lang is not None else ''
        self.api = get_api(texterra_key_path)
        self.remove_quotes = remove_quotes

    def get_extras(self, tokens, sentences):
        sents, sent_starts, raw_tokens = _get_space_joined_sentences(tokens, sentences)
        ne_doc = list(self.api.named_entities(sents, language=self.lang))

        raw_entities = []
        for sent_start, ne_sent in zip(sent_starts, ne_doc):
            for ne in ne_sent:
                raw_entities.append({'id': str(len(raw_entities)), 'type': ne[-1],
                                     'start': sent_start + ne[0], 'end': sent_start + ne[1]})

        entities = align_raw_entities(raw_entities, raw_tokens)
        if self.remove_quotes:
            entities = self.__remove_quotes(tokens, entities)

        return {'ne': SortedSpansSet(entities)}

    @staticmethod
    def __remove_quotes(tokens: List[str], entities: List[Entity]) -> List[Entity]:

        ret = []
        for ent in entities:
            if len(ent) > 2 and tokens[ent.start_token] in QUOTES and tokens[ent.end_token - 1] in QUOTES:
                ent = ent.relocated(ent.start_token + 1, ent.end_token - 1)

            ret.append(ent)

        return ret

    @classmethod
    def from_props(cls, props):
        return cls(props.get("key_path", None), props.get("lang", None), props.get("remove_quotes", False))


class TexterraFeatsProcessor(TokenFeaturesProvider):
    def __init__(self, texterra_key_path=None, lang=None):
        self.lang = lang
        self.api = get_api(texterra_key_path)

    def get_token_features(self, tokens, sentences):
        sents, _, _ = _get_space_joined_sentences(tokens, sentences)
        jsn = [{'text': text} for text in sents]
        params = {'targetType': 'pos-token'}
        if self.lang is not None:
            params['language'] = self.lang

        responce = self.api.custom_query('nlp', params, json=jsn)
        feats = [self._parse_feats(res) for res in responce]
        ret = _get_annotations(feats, sents, {})
        return {'feats': ret}

    def _parse_feats(self, responce):
        ret = []
        for token in responce['annotations']['pos-token']:
            token_feats = {}
            for feat in token['value']['characters']:
                # We need to convert feats types names to use same configs for all processors
                token_feats.update(self._convert_to_dict(feat))
            ret.append((token['start'], token['end'], token_feats))
        return ret

    def _convert_to_dict(self, feat):
        name = feat['type'].capitalize()
        return {name: feat['tag']}

    @classmethod
    def from_props(cls, props):
        return cls(props.get("key_path", None), props.get("lang", None))


def _get_space_joined_sentences(tokens, sentences):
    raw_tokens = []
    sent_starts = [0]
    s_idx = 0
    ret = []
    sent_tokens = []
    for i, token in enumerate(tokens):
        start = 0 if not raw_tokens else raw_tokens[-1][1] + 1

        if i >= sentences[s_idx].end_token:
            s_idx += 1
            ret.append(' '.join(sent_tokens))
            sent_tokens = []
            sent_starts.append(start)
        raw_tokens.append((start, start + len(token)))
        sent_tokens.append(token)

    ret.append(' '.join(sent_tokens))
    return ret, sent_starts, raw_tokens


def _get_annotations(annotations, sents, default_val=None):
    ret = []
    for sent_annotations, sent in zip(annotations, sents):
        start = 0
        for word in sent.split():
            end = start + len(word)
            ret.append(_get_annotation(start, end, sent_annotations, default_val))
            start = end + 1
    return ret


def _get_annotation(start, end, raw_annotations, default_val=None):
    for annotation in raw_annotations:
        if annotation[0] <= start and end <= annotation[1]:
            return annotation[-1]
    return default_val
