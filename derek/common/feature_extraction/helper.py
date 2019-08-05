from enum import Enum


class Direction(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


def encode_sequence(sequence, converter):
    ret = []
    for elem in sequence:
        ret.append(converter[elem])

    return ret


def encode_sequence3d(sequence, converter: dict):
    ret = []
    for subsequence in sequence:
        ret.append(encode_sequence(subsequence, converter))

    return ret


def create_feature(name, props, converter, prop_name=''):
    if not prop_name:
        prop_name = name
    size = props.get(prop_name + '_size', -1)
    if size < 0:
        return {}
    feature = {"converter": converter}
    if size != 0:
        feature["embedding_size"] = size
    return {name: feature}
