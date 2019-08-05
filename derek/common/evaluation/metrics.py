from itertools import chain
from typing import List, Callable
from collections import defaultdict


def f1_score(precision, recall):
    denominator = precision + recall
    return 2 * precision * recall / denominator if denominator > 0 else 0


def binary_precision_score(predicted: List[bool], gold: List[bool]):
    fp, fn, tp, tn = _compute_binary_errors(predicted, gold)
    denominator = tp + fp
    return tp / denominator if denominator > 0 else 0


def binary_recall_score(predicted: List[bool], gold: List[bool]):
    fp, fn, tp, tn = _compute_binary_errors(predicted, gold)
    denominator = tp + fn
    return tp / denominator if denominator > 0 else 0


def binary_f1_score(predicted: List[bool], gold: List[bool]):
    precision = binary_precision_score(predicted, gold)
    recall = binary_recall_score(predicted, gold)
    return f1_score(precision, recall)


def binary_accuracy(predicted: List[bool], gold: List[bool]):
    fp, fn, tp, tn = _compute_binary_errors(predicted, gold)
    denominator = tp + tn + fp + fn
    return (tp + tn) / denominator if denominator > 0 else 0


def binary_macro_avg_score(score_func, segments_predicted: List[List[bool]], segments_gold: List[List[bool]]):
    score = sum(score_func(pred, gold) for pred, gold in zip(segments_predicted, segments_gold))
    return score / len(segments_predicted) if len(segments_predicted) > 0 else 0


def binary_micro_avg_score(score_func, segments_predicted: List[List[bool]], segments_gold: List[List[bool]]):
    return score_func(list(chain.from_iterable(segments_predicted)), list(chain.from_iterable(segments_gold)))


def binary_macro_scores(segments_predicted: List[List[bool]], segments_gold: List[List[bool]]):
    macro_precision = binary_macro_avg_score(binary_precision_score, segments_predicted, segments_gold)
    macro_recall = binary_macro_avg_score(binary_recall_score, segments_predicted, segments_gold)
    macro_f1 = binary_macro_avg_score(binary_f1_score, segments_predicted, segments_gold)

    return macro_precision, macro_recall, macro_f1


def binary_micro_scores(segments_predicted: List[List[bool]], segments_gold: List[List[bool]]):
    micro_precision = binary_micro_avg_score(binary_precision_score, segments_predicted, segments_gold)
    micro_recall = binary_micro_avg_score(binary_recall_score, segments_predicted, segments_gold)
    micro_f1 = binary_micro_avg_score(binary_f1_score, segments_predicted, segments_gold)

    return micro_precision, micro_recall, micro_f1


def _compute_binary_errors(predicted: List[bool], gold: List[bool]):
    fp, fn, tp, tn = 0, 0, 0, 0

    for sample_pred, sample_gold in zip(predicted, gold):
        if sample_pred and sample_gold:
            tp += 1
        if sample_pred and not sample_gold:
            fp += 1
        if not sample_pred and sample_gold:
            fn += 1
        if not sample_pred and not sample_gold:
            tn += 1

    return fp, fn, tp, tn


def ir_precision_score(predicted: set, gold: set):
    return len(predicted.intersection(gold)) / len(predicted) if len(predicted) > 0 else 0


def ir_recall_score(predicted: set, gold: set):
    return len(predicted.intersection(gold)) / len(gold) if len(gold) > 0 else 0


def ir_f1_score(predicted: set, gold: set):
    intersection = predicted.intersection(gold)
    precision = len(intersection) / len(predicted) if len(predicted) > 0 else 0
    recall = len(intersection) / len(gold) if len(gold) > 0 else 0

    return f1_score(precision, recall)


def ir_macro_avg_score(score_func, segments_predicted: List[set], segments_gold: List[set]):
    score = 0

    for segment_predicted, segment_gold in zip(segments_predicted, segments_gold):
        score += score_func(segment_predicted, segment_gold)

    return score / len(segments_predicted) if len(segments_predicted) > 0 else 0


def ir_macro_scores(segments_predicted: List[set], segments_gold: List[set]):
    macro_precision = ir_macro_avg_score(ir_precision_score, segments_predicted, segments_gold)
    macro_recall = ir_macro_avg_score(ir_recall_score, segments_predicted, segments_gold)
    macro_f1 = ir_macro_avg_score(ir_f1_score, segments_predicted, segments_gold)

    return macro_precision, macro_recall, macro_f1


def ir_micro_avg_score(score_func, segments_predicted: List[set], segments_gold: List[set]):
    predicted = set((i, pred) for i, segment_pred in enumerate(segments_predicted) for pred in segment_pred)
    gold = set((i, gold) for i, segment_gold in enumerate(segments_gold) for gold in segment_gold)

    return score_func(predicted, gold)


def ir_micro_scores(segments_predicted: List[set], segments_gold: List[set]):
    micro_precision = ir_micro_avg_score(ir_precision_score, segments_predicted, segments_gold)
    micro_recall = ir_micro_avg_score(ir_recall_score, segments_predicted, segments_gold)
    micro_f1 = ir_micro_avg_score(ir_f1_score, segments_predicted, segments_gold)

    return micro_precision, micro_recall, micro_f1


def ir_categorized_queries_score(
        queries: List,
        queries_predicted: List[set], queries_gold: List[set],
        categorizer: Callable, score_func: Callable):

    queries_predicted_filtered = []
    queries_gold_filtered = []
    possible_categories = set()

    for query, query_pred, query_gold in zip(queries, queries_predicted, queries_gold):
        filtered_pred = _categorize_query_objects(query, query_pred, categorizer)
        filtered_gold = _categorize_query_objects(query, query_gold, categorizer)

        queries_predicted_filtered.append(filtered_pred)
        queries_gold_filtered.append(filtered_gold)
        possible_categories.update(filtered_gold.keys(), filtered_pred.keys())

    categories_results = {}

    for category in possible_categories:
        # defaultdict provide empty set if category is not present in query
        category_predicted = [query_pred[category] for query_pred in queries_predicted_filtered]
        category_gold = [query_gold[category] for query_gold in queries_gold_filtered]

        category_score = score_func(category_predicted, category_gold)
        categories_results[category] = category_score

    return categories_results


def _categorize_query_objects(query, objects: set, categorizer: Callable) -> defaultdict:
    filtered_objects = defaultdict(set)

    for obj in objects:
        filtered_objects[categorizer(query, obj)].add(obj)

    return filtered_objects
