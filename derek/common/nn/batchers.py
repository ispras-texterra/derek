from collections import defaultdict
from logging import getLogger
from typing import Iterable, Callable, Any, Tuple

import numpy as np

from derek.common.helper import BlockIterator, FuncIterable

logger = getLogger('logger')

_DEFAULT_BATCHER_BUFFER_SIZE = 100000


def get_batcher_from_props(
        samples: Iterable, batcher_props: dict, get_padding_value: Callable[[str], Any],
        print_progress: bool = False, need_shuffling: bool = False, get_bucket_for_sample: Callable = None):

    args = {
        "samples": samples,
        "get_padding_value": get_padding_value,
        "batch_size": batcher_props["batch_size"],
        "buffer_size": batcher_props.get("buffer_size", _DEFAULT_BATCHER_BUFFER_SIZE),
        "print_progress": print_progress,
        "need_shuffling": need_shuffling
    }

    batcher_type = batcher_props.get("type", "standard")
    if batcher_type == "bucketing":
        if get_bucket_for_sample is None:
            get_bucket_for_sample = lambda s: s['seq_len'] // batcher_props["bucket_length"]

        args["get_bucket_for_sample"] = get_bucket_for_sample
        factory = get_bucketed_batcher_factory
    elif batcher_type == "standard":
        factory = get_standard_batcher_factory
    else:
        raise Exception(f"Unknown batcher: {batcher_type}")

    return factory(**args)


def get_standard_batcher_factory(
        samples: Iterable, batch_size: int, get_padding_value: Callable[[str], Any], *,
        print_progress: bool = False, need_shuffling: bool = False, buffer_size: int = _DEFAULT_BATCHER_BUFFER_SIZE):
    return lambda: _get_batches(
        samples, batch_size, get_padding_value,
        print_progress=print_progress, need_shuffling=need_shuffling, buffer_size=buffer_size)


def get_bucketed_batcher_factory(
        samples: Iterable, batch_size: int,
        get_padding_value: Callable[[str], Any], get_bucket_for_sample: Callable, *,
        print_progress: bool = False, need_shuffling: bool = False, buffer_size: int = _DEFAULT_BATCHER_BUFFER_SIZE):
    return lambda: _get_bucketed_batches(
        samples, batch_size, get_padding_value, get_bucket_for_sample,
        print_progress=print_progress, need_shuffling=need_shuffling, buffer_size=buffer_size)


def get_segment_batcher_factory(
        samples: Iterable, batch_key: str, size: int, get_padding_value: Callable[[str], Any], *,
        print_progress: bool = False, need_shuffling: bool = False, buffer_size: int = _DEFAULT_BATCHER_BUFFER_SIZE):
    return lambda: _get_segment_batches(
        samples, batch_key, size, get_padding_value,
        print_progress=print_progress, need_shuffling=need_shuffling, buffer_size=buffer_size)


def _get_batches(
        samples: Iterable, batch_size: int, get_padding_value: Callable[[str], Any], *,
        print_progress: bool = False, need_shuffling: bool = False, buffer_size: int = _DEFAULT_BATCHER_BUFFER_SIZE):

    iterator_exceeded = False
    iterator = BlockIterator(iter(samples), buffer_size)

    buffer = []
    while buffer or not iterator_exceeded:
        cur_batch_size = 0
        batch = defaultdict(list)

        while cur_batch_size < batch_size and not iterator_exceeded:
            if not buffer:
                try:
                    buffer = next(iterator)
                    if print_progress:
                        logger.info("{} samples added to buffer".format(len(buffer)))
                    if need_shuffling:
                        np.random.shuffle(buffer)
                except StopIteration:
                    iterator_exceeded = True
                    break
            sample = buffer.pop(0)
            _add_sample_to_batch(sample, batch)
            cur_batch_size += 1
        if cur_batch_size > 0:
            yield _pad_batch(batch, get_padding_value)


def _get_bucketed_batches(
        samples: Iterable, batch_size: int,
        get_padding_value: Callable[[str], Any], get_bucket_for_sample: Callable, *,
        print_progress: bool = False, need_shuffling: bool = False, buffer_size: int = _DEFAULT_BATCHER_BUFFER_SIZE):

    buffers = FuncIterable(lambda: BlockIterator(iter(samples), buffer_size))

    for buffer in buffers:
        if print_progress:
            logger.info("{} samples added to buffer".format(len(buffer)))

        if need_shuffling:
            np.random.shuffle(buffer)

        buffer_buckets = defaultdict(list)
        for s in buffer:
            buffer_buckets[get_bucket_for_sample(s)].append(s)

        if print_progress:
            logger.info("{} buckets in buffer".format(len(buffer_buckets)))

        # sorting is applied to ensure reproducibility of results
        bucketed_samples = list(buffer_buckets[key] for key in sorted(buffer_buckets.keys()))
        buffer_batches = []

        for bucket in bucketed_samples:
            cur_batch_size = 0
            batch = defaultdict(list)

            for sample in bucket:
                _add_sample_to_batch(sample, batch)
                cur_batch_size += 1

                if cur_batch_size == batch_size:
                    buffer_batches.append(_pad_batch(batch, get_padding_value))
                    batch = defaultdict(list)
                    cur_batch_size = 0

            if cur_batch_size > 0:
                buffer_batches.append(_pad_batch(batch, get_padding_value))

        if need_shuffling:
            np.random.shuffle(buffer_batches)

        for batch in buffer_batches:
            yield batch


def _get_segment_batches(
        samples: Iterable, batch_key: str, size: int, get_padding_value: Callable[[str], Any], *,
        print_progress: bool = False, need_shuffling: bool = False, buffer_size: int = _DEFAULT_BATCHER_BUFFER_SIZE):
    try:
        first_sample = next(iter(samples))
    except StopIteration:
        return []

    common_features = set(key for key in first_sample if key != batch_key)
    buffers = FuncIterable(lambda: BlockIterator(iter(samples), buffer_size))

    for buffer in buffers:
        if print_progress:
            logger.info("{} samples added to buffer".format(len(buffer)))

        if need_shuffling:
            np.random.shuffle(buffer)

        batches = []

        for sample in buffer:
            if len(sample[batch_key]) == 0:
                continue

            # get 1 element batch for token features common for all examples in segment
            common_batch = next(_get_batches(
                [{key: value for key, value in sample.items() if key in common_features}], 1, get_padding_value))

            for batch in _get_batches(
                    sample[batch_key], size, get_padding_value, print_progress=False, need_shuffling=need_shuffling):

                batch.update(common_batch)
                batches.append(batch)

        if need_shuffling:
            np.random.shuffle(batches)

        for batch in batches:
            yield batch


def _add_sample_to_batch(sample: dict, batch: defaultdict):
    for key in sample:
        batch[key].append(sample[key])


def _pad_batch(batch: dict, get_padding_value_and_rank: Callable[[str], Tuple[Any, int]]):
    padded_batch = {}

    for key, values in batch.items():
        padding_rank = get_padding_value_and_rank(key)

        if padding_rank is None:
            raise Exception(f"No padding for {key} feature")

        padding_value, rank = padding_rank
        # rank is increased with batching
        rank = rank + 1
        if padding_value is None or rank == 1:
            padded_batch[key] = values
        elif rank == 2:
            padded_batch[key] = pad_sequences2d(values, padding_value)
        elif rank == 3:
            padded_batch[key] = pad_sequences3d(values, padding_value)
        else:
            padded_batch[key] = pad_sequences(values, padding_value)

    return padded_batch


def pad_sequences3d(values, padding_val):
    max_len_dim_1 = max(map(lambda x: len(x), values))
    max_len_dim_2 = max([max([len(sub_seq) for sub_seq in seq]) for seq in values])
    ret = []
    for seq in values:
        ret_row = []
        for sub_seq in seq:
            if isinstance(sub_seq, np.ndarray):
                sub_seq = sub_seq.tolist()
            ret_row.append(sub_seq + [padding_val]*(max_len_dim_2-len(sub_seq)))
        ret.append(ret_row + [[padding_val]*max_len_dim_2]*(max_len_dim_1-len(seq)))
    return np.array(ret, dtype=type(padding_val))


def pad_sequences2d(values, padding_val):
    max_len = max(map(lambda x: len(x), values))
    ret = []
    for seq in values:
        if isinstance(seq, np.ndarray):
            seq = seq.tolist()
        ret.append(seq + [padding_val]*(max_len-len(seq)))
    return np.array(ret, dtype=type(padding_val))


def pad_sequences(values, padding_val):
    shape = _get_dim_sizes(values)
    tensor = np.full(shape, padding_val)
    _init_tensor(tensor, values, type(padding_val))
    return tensor


def _get_dim_sizes(seq):
    if not (isinstance(seq, list) or isinstance(seq, np.ndarray)):
        return []
    seq_len = len(seq)

    sub_lens = []
    for sub_seq in seq:
        sub_lens.append(_get_dim_sizes(sub_seq))

    sub_lens = np.asarray(sub_lens)
    if sub_lens.any():
        sub_lens = np.max(sub_lens, axis=0)
        return [seq_len] + list(sub_lens)
    else:
        return [seq_len]


def _init_tensor(tensor, values, padding_type):
    for i, val in enumerate(values):
        if isinstance(val, list) or isinstance(val, np.ndarray):
            _init_tensor(tensor[i], val, padding_type)
        else:
            if not isinstance(val, padding_type):
                raise Exception("Value in tensor must have the same type as padding")
            tensor[i] = val
