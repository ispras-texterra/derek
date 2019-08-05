
def scores_to_dict(precision, recall, f1):
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def categorized_scores_to_dict(categories_scores):
    ret = {}
    for category, score in categories_scores.items():
        ret[str(category)] = scores_to_dict(*score)
    return ret
