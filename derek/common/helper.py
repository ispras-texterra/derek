import collections
from functools import reduce
from typing import Iterator, Callable, List


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
