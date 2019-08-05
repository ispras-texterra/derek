from babylondigger.datamodel import Document as DiggerDoc
from babylondigger.datamodel import Token, Pos
from babylondigger.datamodel import Sentence as DiggerSentence
from derek.data.model import Document, Paragraph, Sentence


def convert_from_digger_to_derek(diggerdoc: DiggerDoc, doc_name: str) -> Document:
    tokens = []
    token_features = {
        "pos": [],
        "dt_labels": [],
        "dt_head_distances":  [],
        "lemmas": [],
        "feats": []
    }

    for i, token in enumerate(diggerdoc.tokens):
        tokens.append(token.doc_text)
        token_features["pos"].append(token.pos.upos)
        token_features["dt_labels"].append(token.deprel)
        token_features["dt_head_distances"].append(token.head_index - i if token.head_index != -1 else 0)
        token_features["lemmas"].append(token.lemma)
        token_features["feats"].append(token.pos.feats)

    sentences = list(Sentence(sent.start, sent.end) for sent in diggerdoc.sentences_boundaries)
    # here we assume all doc sentences to be in 1 paragraph
    paragraphs = [Paragraph(0, len(sentences))]

    return Document(doc_name, tokens, sentences, paragraphs, token_features=token_features)


def convert_from_derek_to_digger(doc: Document) -> DiggerDoc:
    pos = doc.token_features.get("pos", None)
    dt_labels = doc.token_features.get("dt_labels", None)
    dt_distances = doc.token_features.get("dt_head_distances", None)
    lemmas = doc.token_features.get("lemmas", None)
    feats = doc.token_features.get("feats", None)

    dig_tokens = []
    text = ' '.join(doc.tokens)
    cur_tok_start = 0
    for i, token in enumerate(doc.tokens):
        token_feats = feats[i] if feats is not None else None
        token_pos = Pos(pos[i], feats=token_feats) if pos is not None else None
        token_deprel = dt_labels[i] if dt_labels is not None else None
        # -1 for root token
        token_head_index = (i + dt_distances[i] if dt_distances[i] else -1) if dt_distances is not None else None
        token_lemma = lemmas[i] if lemmas is not None else None
        dig_tokens.append(Token(cur_tok_start, cur_tok_start + len(token),
                                token_pos, token_lemma, token_deprel,
                                head_document_index=token_head_index))

        # all tokens divided by space
        cur_tok_start += 1 + len(token)

    dig_sentences = [DiggerSentence(sent.start_token, sent.end_token) for sent in doc.sentences]

    return DiggerDoc(text, dig_tokens, dig_sentences)
