from typing import List, Set, Iterable
from warnings import warn

from derek.data.model import Document, Entity, Sentence, Relation, TokenSpan


def get_entity_distance_between_entities(doc: Document, e1: Entity, e2: Entity):
    distance = 0

    start = min((e1.end_token, e2.end_token))
    end = max((e1.start_token, e2.start_token))

    for entity in doc.entities:
        if entity.end_token > end:
            break
        if entity.start_token >= start:
            distance += 1
    return distance


def get_sentence_distance_between_entities(doc: Document, e1: Entity, e2: Entity):
    return abs(doc.get_entity_sent_idx(e1) - doc.get_entity_sent_idx(e2))


def get_sentence_relations(relations: Iterable[Relation], sent: Sentence):
    ret = set()

    for rel in relations:
        if sent.contains(rel.first_entity) and sent.contains(rel.second_entity):
            ret.add(rel)

    return ret


def in_paragraph(paragraph, entity_sentence_idx):
    return paragraph.start_sentence <= entity_sentence_idx < paragraph.end_sentence


def get_sentence_paragraph(doc: Document, sent_idx: int):
    for paragraph in doc.paragraphs:
        if paragraph.start_sentence <= sent_idx < paragraph.end_sentence:
            return paragraph
    return None


def get_max_paragraph_lengths(docs: list):
    max_token_len = 1
    max_sent_len = 1
    for doc in docs:
        for paragraph in doc.paragraphs:
            p_token_len = doc.sentences[paragraph.end_sentence - 1].end_token - \
                          doc.sentences[paragraph.start_sentence].start_token

            if max_token_len < p_token_len:
                max_token_len = p_token_len

            p_sent_len = paragraph.end_sentence - paragraph.start_sentence
            if max_sent_len < p_sent_len:
                max_sent_len = p_sent_len
    return max_token_len, max_sent_len


def adjust_sentences(sentences, entities):
    cur_sentence = 0
    sentences = sentences.copy()
    ret = []
    for entity in entities:
        while sentences[cur_sentence].end_token <= entity.start_token:
            ret.append(sentences[cur_sentence])
            cur_sentence += 1

        sentence = sentences[cur_sentence]
        if sentence.start_token <= entity.start_token and entity.end_token <= sentence.end_token:
            continue
        start = sentence.start_token
        while sentences[cur_sentence].end_token < entity.end_token:
            cur_sentence += 1
        sentences[cur_sentence] = Sentence(start, sentences[cur_sentence].end_token)

    ret += sentences[cur_sentence:]
    return ret


def align_raw_entities(raw_entities, raw_tokens):
    """
    :param raw_tokens: sorted list of tuples: (start, end)
    :param raw_entities: list of dicts: {'id', 'type', 'start', 'end'}
    :return: list of Entity objects
    """

    ret = []

    for raw_entity in raw_entities:
        start = None
        end = None

        for i, raw_token in enumerate(raw_tokens):
            if raw_token[0] <= raw_entity['start'] < raw_token[1]:
                start = i

            if raw_token[0] < raw_entity['end'] <= raw_token[1]:
                end = i

            # shifted entity cases
            # choose first token with start more than entity start
            if start is None and raw_token[0] > raw_entity["start"]:
                start = i

            # choose last token with end less than entity end
            if end is None and raw_token[0] >= raw_entity["end"]:
                end = i - 1

            if start is not None and end is not None:
                break

        assert start is not None and end is not None
        ret.append(Entity(raw_entity['id'], start, end + 1, raw_entity['type']))

    return ret


# TODO return set / dict
def get_marked_tokens_on_root_path_for_span(doc: Document, span, *, add_distance=False):
    start, end, span_sent_idx = span
    distances_to_root = doc.token_features["dt_head_distances"]
    main_span_token = find_span_head_token(doc, TokenSpan(start, end))

    res = [False] * len(doc.tokens)

    for sent_idx in range(len(doc.sentences)):
        if sent_idx != span_sent_idx:
            continue

        distance_from_span = 0
        res[main_span_token] = distance_from_span if add_distance else True
        distance_to_parent = distances_to_root[main_span_token]
        current_idx = main_span_token

        while distance_to_parent != 0:
            current_idx += distance_to_parent
            distance_from_span += 1
            res[current_idx] = distance_from_span if add_distance else True
            distance_to_parent = distances_to_root[current_idx]

    return res


def find_span_head_token(doc: Document, span: TokenSpan):
    """
        chooses root token of span as main if present or leftmost token with outer link
    """
    dt_head_distances = doc.token_features["dt_head_distances"]

    main_span_token = None

    for i in range(span.start_token, span.end_token):
        distance_to_parent = dt_head_distances[i]

        # root token
        if distance_to_parent == 0:
            main_span_token = i
            break

        parent = i + distance_to_parent

        # first outer link, check other span tokens for rootness
        if main_span_token is None and (parent < span.start_token or parent >= span.end_token):
            main_span_token = i

    return main_span_token


def collapse_intersecting_entities(entities: List[Entity], relations: Set[Relation]):
    # assume entities list is sorted with start token
    entities_to_process = list(entities)
    entities_mapping = {}
    new_entities = []

    while entities_to_process:
        ent1 = entities_to_process.pop(0)
        ent_end = ent1.end_token
        type_ent = ent1.type
        ents_to_collapse = []

        for ent2 in entities_to_process:
            if ent2.start_token >= ent_end:
                continue
            if ent1.type != ent2.type:
                warn(f"Intersecting entities have different types: {ent1} absorbed {ent2}")
                assert not ent1.coincides(ent2), "Two entities of different types on the same span"
                assert ent1.contains(ent2) or ent2.contains(ent1) or not ent1.intersects(ent2), \
                    "Two entities of different types are not embedded, intersecting only"

            ents_to_collapse.append(ent2)
            ent_end = max(ent_end, ent2.end_token)

            if len(ent2) > len(ent1):
                type_ent = ent2.type

        if not ents_to_collapse:
            new_ent = ent1
        else:
            new_ent = ent1.relocated(ent1.start_token, ent_end).with_type(type_ent)

        new_entities.append(new_ent)
        entities_mapping[ent1] = new_ent

        for ent2 in ents_to_collapse:
            entities_mapping[ent2] = new_ent
            entities_to_process.remove(ent2)

    new_relations = {
        Relation(entities_mapping[r.first_entity], entities_mapping[r.second_entity], r.type) for r in relations}

    # new entities list was constructed as sorted
    return new_entities, new_relations
