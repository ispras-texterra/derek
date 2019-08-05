import os
from typing import Tuple, Callable

from derek.common.helper import ResultsStorage, SCORES_DICT_TYPE


def get_saving_hook(out_path, name):
    def save(classifier, epoch):
        model = name + '{:03}'.format(epoch)
        path = os.path.join(out_path, model)
        os.makedirs(path, exist_ok=True)
        classifier.save(path)

    return save


def get_specific_epoch_hook(hook, epoch):
    def func(classifier, e):
        if e == epoch:
            hook(classifier, e)

    return func


def get_best_model_picker() -> Tuple[Callable, ResultsStorage]:
    result = ResultsStorage()

    def evaluate(main_score: float, scores: SCORES_DICT_TYPE, func):
        if result.add_scores(main_score, scores):
            func()
        else:
            pass

    return evaluate, result
