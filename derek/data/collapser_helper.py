from typing import Iterable, Dict, Any, Union, Tuple
from warnings import warn

from derek.data.model import TokenSpan, Entity, Sentence


def build_borders_dict(collection: Iterable[TokenSpan]) -> Dict[Any, Tuple[int, int]]:
    return {x: (x.start_token, x.end_token) for x in collection}


def create_objects_with_new_borders(
        container: Iterable[Union[Sentence, Entity]], new_borders: Dict[Any, Tuple[int, int]]) -> Dict:

    mapping = {}
    for obj in container:
        new_object = obj.relocated(*new_borders[obj])
        mapping[obj] = new_object

    return mapping


def shift_borders_after_collapse(
        borders_dict: Dict[Any, Tuple[int, int]], old_start: int, old_end: int, new_length: int = 1):

    old_length = old_end - old_start

    if old_length == new_length:
        return

    shift = old_length - new_length
    for obj, (start_token, end_token) in borders_dict.items():
        if end_token <= old_start:
            pass
        elif start_token >= old_end:
            start_token -= shift
            end_token -= shift
        elif end_token >= old_end and start_token <= old_start:
            end_token -= shift
        elif start_token >= old_start and end_token <= old_end:
            start_token = old_start
            end_token = start_token + new_length
        elif old_start <= start_token < old_end <= end_token:
            start_token = old_start
            end_token -= shift
        elif start_token <= old_start < end_token <= old_end:
            end_token = old_start + new_length

        borders_dict[obj] = (start_token, end_token)
        if start_token == end_token:
            warn(f"{obj} start == end after collapsement")
