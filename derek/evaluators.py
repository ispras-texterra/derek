from typing import Union, Callable, Any, List, Tuple

from derek.common.evaluation.results_controllers import SCORES_DICT_TYPE
from derek.data.model import Document
from derek.ner.evaluation import evaluate as ner_evaluate
from derek.net.evaluation import evaluate as net_evaluate
from derek.rel_ext.evaluation import evaluate as rel_evaluate

_EVALUATORS = {
    "rel_ext": lambda classifier, docs, need_stats: rel_evaluate(classifier, docs, need_stats),
    "ner": lambda classifier, docs, need_stats: ner_evaluate(classifier, docs, need_stats),
    "net": lambda classifier, docs, need_stats: net_evaluate(classifier, docs, need_stats)
}


def evaluator_for(task_name) \
        -> Callable[[Any, List[Document], bool], Tuple[float, SCORES_DICT_TYPE, Union[None, Callable]]]:
    ev = _EVALUATORS.get(task_name, None)
    if ev is None:
        raise Exception(f"{task_name} is not currently supported")
    return ev
