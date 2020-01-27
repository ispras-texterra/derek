import json
from derek.data.model import Entity, Sentence, Paragraph, Document, Relation


def load_json_file_as_dict(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def make_document_from_json_file(file_path):
    d = load_json_file_as_dict(file_path)

    tokens = d.get('tokens', [])
    entities = d.get('entities', [])
    sentences = d.get('sentences', [])
    paragraphs = d.get('paragraphs', [])
    token_features = {}

    for feature in ['pos', 'entities_types', 'entities_depths', 'borders', 'dt_labels', 'dt_head_distances',
                    'dt_depths', 'dt_deltas_forward', 'dt_deltas_backward',
                    'dt_breakups_forward', 'dt_breakups_backward']:
        if feature in d:
            token_features[feature] = d[feature]

    relations = d.get('relations', [])

    doc_entities = []
    for ent in entities:
        id_, start_token, end_token, ent_type = tuple(ent)
        doc_entities.append(Entity(id_, start_token, end_token, ent_type))

    doc_sentences = []

    for sent in sentences:
        start_token, end_token = tuple(sent)
        doc_sentences.append(Sentence(start_token, end_token))

    doc_paragraphs = []

    for par in paragraphs:
        start_sentence, end_sentence = tuple(par)
        doc_paragraphs.append(Paragraph(start_sentence, end_sentence))

    doc_relations = []

    for rel in relations:
        e1 = None
        e2 = None
        e1_id, e2_id, rel_type = tuple(rel)

        for entity in doc_entities:
            if entity.id == e1_id:
                e1 = entity
            if entity.id == e2_id:
                e2 = entity

            if e1 is not None and e2 is not None:
                break

        doc_relations.append(Relation(e1, e2, rel_type))

    doc = Document("", tokens, doc_sentences, doc_paragraphs, token_features=token_features)
    if 'entities' in d:
        doc = doc.with_entities(doc_entities)
    if 'relations' in d:
        doc = doc.with_relations(doc_relations)
    return doc


def get_training_hook(docs):
    ret = []

    def evaluate(clf, _):
        for doc in docs:
            clf.predict_doc(doc)
        # change mutable object to validate this method was called during training
        ret.append(True)
        return False

    return evaluate, ret
