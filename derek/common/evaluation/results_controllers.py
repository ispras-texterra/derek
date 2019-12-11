from functools import reduce
from typing import Dict, Union, Tuple, Callable, Generator

SCORES_DICT_TYPE = Dict[str, Union[float, 'SCORES_DICT']]


class ResultsStorage:
    def __init__(self):
        self.__best_score_idx = None
        self.__main_scores = []
        self.__scores_dicts = []

    def add_scores(self, main_score: float, scores_dict: SCORES_DICT_TYPE = None) -> bool:
        self.__main_scores.append(main_score)
        self.__scores_dicts.append(scores_dict)

        if self.__best_score_idx is None or main_score > self.__main_scores[self.__best_score_idx]:
            self.__best_score_idx = len(self.__main_scores) - 1

        return self.__best_score_idx == len(self.__main_scores) - 1

    @property
    def best_score_idx(self) -> int:
        return self.__best_score_idx

    @property
    def best_scores(self) -> Tuple[float, SCORES_DICT_TYPE]:
        return self.__main_scores[self.__best_score_idx], self.__scores_dicts[self.__best_score_idx]

    @property
    def mean_scores(self) -> Tuple[float, SCORES_DICT_TYPE]:
        return self.__mean(self.__main_scores), self.__compute_mean_scores()

    @property
    def main_scores(self):
        return list(self.__main_scores)

    @property
    def scores_dicts(self):
        return list(self.__scores_dicts)

    def __compute_mean_scores(self):
        return self.__take_mean_dict_of_list(reduce(self.__append_dict_val_to_dict_of_list, self.__scores_dicts, {}))

    @staticmethod
    def __mean(lst: list):
        return sum(lst) / len(lst)

    @staticmethod
    def __append_dict_val_to_dict_of_list(d1: dict, d2: SCORES_DICT_TYPE):
        res = {}

        for key, val in d2.items():
            if isinstance(val, dict):
                res[key] = ResultsStorage.__append_dict_val_to_dict_of_list(d1.get(key, {}), val)
            else:
                res[key] = d1.get(key, []) + [val]

        return res

    @staticmethod
    def __take_mean_dict_of_list(d: dict):
        res = {}

        for key, val in d.items():
            if isinstance(val, dict):
                res[key] = ResultsStorage.__take_mean_dict_of_list(val)
            else:
                res[key] = ResultsStorage.__mean(val)

        return res


def get_best_model_picker(improvement_controller: Generator[bool, bool, None]) -> Tuple[Callable, ResultsStorage]:
    result = ResultsStorage()
    next(improvement_controller)

    def evaluate(main_score: float, scores: SCORES_DICT_TYPE, func):
        improvement = result.add_scores(main_score, scores)

        if improvement:
            func()

        try:
            improvement_controller.send(improvement)
        except StopIteration:
            return True

        return False

    return evaluate, result


def history_improvement_controller(history_length: int) -> Generator[bool, bool, None]:
    assert history_length >= 0
    no_update_for = 0
    updated = yield None

    while no_update_for < history_length or updated:
        if updated:
            no_update_for = 0
            updated = yield True
        else:
            no_update_for += 1
            updated = yield False
