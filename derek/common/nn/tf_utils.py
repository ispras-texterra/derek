import os
import random

import numpy as np
import tensorflow as tf
from abc import abstractmethod, ABCMeta


def _create_tf_session(seed=None):
    gpu_num = int(os.environ.get("GPU_NUM_PROCESSES", 1))
    if gpu_num < 1:
        raise ValueError("Invalid value for GPU_NUM_PROCESSES:" + str(gpu_num))
    config = tf.ConfigProto()
    if gpu_num == 1:
        config.gpu_options.allow_growth = True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = 1/gpu_num

    graph = tf.Graph()
    if seed is not None:
        graph.seed = seed
    return tf.Session(config=config, graph=graph)


class TFSessionContextManager:
    def __init__(self, seed=None):
        self._session = None
        self.__seed = seed

    def __enter__(self):
        self._session = _create_tf_session(self.__seed)
        self._session.__enter__()

        return self

    def __exit__(self, *exc):
        self._session.__exit__(*exc)
        self._session = None


class TFSessionAwareClassifier(TFSessionContextManager, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def __enter__(self):
        super().__enter__()
        return self._load()

    @abstractmethod
    def _load(self):
        pass


class TFSessionAwareTrainer(TFSessionContextManager):
    def __init__(self, props: dict):
        super().__init__(props["seed"])
        random.seed(props['seed'])
        np.random.seed(props['seed'])
