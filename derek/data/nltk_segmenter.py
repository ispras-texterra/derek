from nltk import load
import nltk.tokenize as nltk

from derek.data.model import Sentence
from derek.data.processing_helper import fix_joined_tokens, expand_names, fix_joined_names, \
    eliminate_references_and_figures, fix_raw_tokens_after_elimination, fix_sentences_ends


class NLTKTextSegmenter:
    def __init__(self, *, nltk_model=None, post_processing=False):
        if nltk_model is None:
            nltk_model = 'tokenizers/punkt/english.pickle'

        self.sent_model = load(nltk_model)
        self.post = post_processing

    def segment(self, text):
        sentences = []
        tokens = []
        raw_tokens = []
        token_sent_start = 0

        if self.post:
            text, eliminations = eliminate_references_and_figures(text)

        for raw_sent_start, raw_sent_end in self.sent_model.span_tokenize(text):
            sent_raw_tokens = list(nltk._treebank_word_tokenizer.span_tokenize(text[raw_sent_start:raw_sent_end]))

            raw_tokens.extend((start + raw_sent_start, end + raw_sent_start) for start, end in sent_raw_tokens)
            tokens.extend(text[start + raw_sent_start:end + raw_sent_start] for start, end in sent_raw_tokens)
            sentences.append(Sentence(token_sent_start, token_sent_start + len(sent_raw_tokens)))

            token_sent_start += len(sent_raw_tokens)

        if self.post:
            tokens, sentences, raw_tokens = fix_joined_tokens(tokens, sentences, raw_tokens)
            tokens, sentences, raw_tokens = fix_joined_names(tokens, sentences, raw_tokens)
            tokens = expand_names(tokens)
            raw_tokens = fix_raw_tokens_after_elimination(raw_tokens, eliminations)
            sentences = fix_sentences_ends(tokens, sentences)

        return tokens, sentences, raw_tokens
