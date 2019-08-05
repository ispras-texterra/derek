from derek.data.model import Document, Relation
from typing import Set
from math import log2
from derek.data.helper import get_sentence_distance_between_entities
from derek.data.helper import get_sentence_relations
from derek.common.evaluation.metrics import ir_categorized_queries_score, ir_micro_scores
from derek.common.evaluation.helper import scores_to_dict, categorized_scores_to_dict


__CATEGORIZERS = [
    ("type", lambda d, r: r.type),
    ("args_types", lambda d, r: r.entities_types),
    ("args_token_log_dists", lambda d, r: round(log2(1 + r.first_entity.token_distance_to(r.second_entity)))),
    ("args_sent_dists", lambda d, r: get_sentence_distance_between_entities(d, r.first_entity, r.second_entity))
]


def evaluate(classifier, docs, need_stats):
    gold_rels = [doc.relations for doc in docs]
    # ensure that we do not use gold labels for prediction
    docs = [doc.without_relations() for doc in docs]
    pred = classifier.predict_docs(docs)
    predicted_rels = [pred[doc.name] for doc in docs]

    ovrl_precision, ovrl_recall, ovrl_f1 = ir_micro_scores(predicted_rels, gold_rels)
    main_score = ovrl_f1
    scores = scores_to_dict(ovrl_precision, ovrl_recall, ovrl_f1)

    for name, categorizer in __CATEGORIZERS:
        categories_scores = ir_categorized_queries_score(docs, predicted_rels, gold_rels, categorizer, ir_micro_scores)
        scores[name] = categorized_scores_to_dict(categories_scores)

    return main_score, scores,\
        (lambda i: __generate_stats(docs[i], predicted_rels[i], gold_rels[i])) if need_stats else None


def __generate_stats(doc: Document, predicted: Set[Relation], gold: Set[Relation]) -> str:
    return '\n'.join(__generate_sent_stats(doc, sent, predicted, gold) + '\n' * 2 + '=' * 40 + '\n'
                     for sent in doc.sentences)


def __generate_sent_stats(doc, sent, predicted, gold):
    sent_entities = doc.entities.contained_in(sent)
    sent_pred_rels = get_sentence_relations(predicted, sent)
    sent_gold_rels = get_sentence_relations(gold, sent)

    tp = sent_pred_rels.intersection(sent_gold_rels)
    fp = sent_pred_rels.difference(sent_gold_rels)
    fn = sent_gold_rels.difference(sent_pred_rels)

    ret = ' '.join(doc.tokens[sent.start_token:sent.end_token])
    ret += '\n\nEntities:\n'
    ret += '\n'.join(
        f'{ent.id} {ent.start_token - sent.start_token} {ent.end_token - sent.start_token} ' +
        ' '.join(doc.tokens[ent.start_token:ent.end_token]) for ent in sent_entities)

    ret += '\n\nFalse positives:\n'
    ret += '\n'.join(__generate_rel_stats(r) for r in fp)
    ret += '\n\nFalse negatives:\n'
    ret += '\n'.join(__generate_rel_stats(r) for r in fn)
    ret += '\n\nTrue positives:\n'
    ret += '\n'.join(__generate_rel_stats(r) for r in tp)

    return ret


def __generate_rel_stats(rel):
    return f"{rel.first_entity.id} {rel.second_entity.id} {rel.type}"
