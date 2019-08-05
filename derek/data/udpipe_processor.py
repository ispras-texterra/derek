import ufal.udpipe as ud

from derek.data.model import Sentence


def load_udpipe_model(path):
    return ud.Model.load(path)


class UDPipeTextSegmentor:
    def __init__(self, model):
        self.model = model

    def segment(self, text):
        sent = ud.Sentence()
        tokenizer = self.model.newTokenizer('ranges')
        tokenizer.setText(text)

        sentences = []
        tokens = []
        raw_tokens = []
        sent_start = 0

        while tokenizer.nextSentence(sent):
            words = sent.words[1:]
            sent_raw_tokens = [(word.getTokenRangeStart(), word.getTokenRangeEnd()) for word in words]

            sentences.append(Sentence(sent_start, sent_start + len(sent_raw_tokens)))
            tokens += [text[raw_token[0]: raw_token[1]] for raw_token in sent_raw_tokens]

            raw_tokens += sent_raw_tokens
            sent_start += len(sent_raw_tokens)

        return tokens, sentences, raw_tokens
