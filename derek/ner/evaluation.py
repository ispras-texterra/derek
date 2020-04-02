from typing import Set
from derek.common.evaluation.metrics import ir_categorized_queries_score, ir_micro_scores
from derek.common.evaluation.helper import scores_to_dict, categorized_scores_to_dict
from derek.data.model import Document


__CATEGORIZERS = [
    ("type", lambda d, e: e[2])
]


def evaluate(classifier, docs, need_stats):
    gold_ents = [__ents_to_tuple_set(doc.entities) for doc in docs]
    # ensure classifier don't use entities and relations
    docs = [doc.without_relations().without_entities() for doc in docs]
    predicted_ents = list(map(__ents_to_tuple_set, classifier.predict_docs(docs)))

    ovrl_precision, ovrl_recall, ovrl_f1 = ir_micro_scores(predicted_ents, gold_ents)
    main_score = ovrl_f1
    scores = scores_to_dict(ovrl_precision, ovrl_recall, ovrl_f1)

    for name, categorizer in __CATEGORIZERS:
        categories_scores = ir_categorized_queries_score(docs, predicted_ents, gold_ents, categorizer, ir_micro_scores)
        scores[name] = categorized_scores_to_dict(categories_scores)

    return main_score, scores,\
        (lambda i: __generate_stats(docs[i], predicted_ents[i], gold_ents[i])) if need_stats else None


# we can't evaluate entities as is because they contain generated id and stored in list
def __ents_to_tuple_set(ents):
    return {(e.start_token, e.end_token, e.type) for e in ents}


def __generate_stats(doc: Document, predicted: Set[tuple], gold: Set[tuple]):
    predicted_labels, gold_labels = ["O"] * len(doc.tokens), ["O"] * len(doc.tokens)

    for ent_set, labels_list in ((predicted, predicted_labels), (gold, gold_labels)):
        for start_token, end_token, ent_type in ent_set:
            if end_token - start_token == 1:
                labels_list[start_token] = f"U-{ent_type}"
                continue

            labels_list[start_token] = f"B-{ent_type}"
            labels_list[start_token + 1:end_token-1] = [f"I-{ent_type}"] * (end_token - start_token - 2)
            labels_list[end_token - 1] = f"L-{ent_type}"

    ret = "Token\tPredicted Label\tGold Label\n\n"
    ret += "\n\n".join(
        "\n".join(f"{token}\t{pr_l}\t{g_l}" for token, pr_l, g_l in zip(
            doc.tokens[sent.start_token:sent.end_token],
            predicted_labels[sent.start_token:sent.end_token],
            gold_labels[sent.start_token:sent.end_token])) for sent in doc.sentences)

    return ret

