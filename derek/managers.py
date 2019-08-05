from derek.rel_ext import RelExtClassifier, RelExtTrainer
from derek.coref import CorefClassifier, CorefTrainer
from derek.pretraining_lms import LMSTrainer, LMSClassifier
from derek.ner import ChainedNERClassifier, NERTrainer
from derek.net import NETClassifier, NETTrainer


_TRAINERS = {
    "rel_ext": RelExtTrainer,
    "coref": CorefTrainer,
    "pretraining_lms": LMSTrainer,
    "ner": NERTrainer,
    "net": NETTrainer,
}


def trainer_for(task_name):
    trainer = _TRAINERS.get(task_name, None)
    if trainer is None:
        raise Exception(f"{task_name} is not currently supported")
    return trainer


_CLASSIFIERS = {
    "rel_ext": RelExtClassifier,
    "coref": CorefClassifier,
    "pretraining_lms": LMSClassifier,
    "ner": ChainedNERClassifier,
    "net": NETClassifier
}


def classifier_for(task_name):
    clf = _CLASSIFIERS.get(task_name, None)
    if clf is None:
        raise Exception(f"{task_name} is not currently supported")
    return clf
