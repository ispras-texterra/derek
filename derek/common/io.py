import os
import pickle
from typing import Iterator

import msgpack

from derek.common.helper import FuncIterable


def save_with_pickle(obj, path, name):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_with_pickle(path, name):
    with open(os.path.join(path, name), 'rb') as f:
        return pickle.load(f)


class CacheMPManager:
    def __init__(self, iterator: Iterator, path: str):
        self.iterator = iterator
        self.path = path

    def __enter__(self):
        stop_iter = False
        with open(self.path, 'wb') as f:
            while not stop_iter:
                try:
                    msgpack.dump(next(self.iterator), f)
                except StopIteration:
                    stop_iter = True

        return FuncIterable(lambda: _mp_iterate(self.path))

    def __exit__(self, *exc):
        os.remove(self.path)


def _mp_iterate(path: str):
    with open(path, 'rb') as f:
        for obj in msgpack.Unpacker(f, raw=False):
            yield obj


def get_batch_size(default: int = 32):
    return int(os.getenv("PRED_BATCH_SIZE", str(default)))
