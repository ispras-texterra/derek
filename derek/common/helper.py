import collections
from functools import reduce
from typing import Iterator, Callable, Tuple, Dict, Union, List

SCORES_DICT_TYPE = Dict[str, Union[float, 'SCORES_DICT']]


class FuncIterable:
    def __init__(self, func):
        self.func = func

    def __iter__(self):
        return self.func()


class BlockIterator(collections.Iterator):

    def __init__(self, iterator: Iterator, block_size: int):
        self.iterator = iterator
        self.block_size = block_size

    def __next__(self) -> list:
        block = []

        try:
            while len(block) < self.block_size:
                block.append(next(self.iterator))
            return block
        except StopIteration:
            if block:
                return block
            raise StopIteration()


def from_namespace(name, namespace):
    return name[len(namespace)+1:] if name.startswith(namespace) else name


def namespaced(dictionary, namespace):
    return {namespace + '_' + key: value for key, value in dictionary.items()}


class _FunctionsComposer:
    def __init__(self, functions):
        self.functions = functions

    def __call__(self, *args, **kwargs):
        def _compose2(f, g):
            return lambda *a, **kw: f(g(*a, **kw))
        return reduce(_compose2, self.functions)(*args, **kwargs)


def compose_functions(functions: List[Callable]):
    if not functions:
        raise Exception("Provided empty list")

    return _FunctionsComposer(functions)


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
