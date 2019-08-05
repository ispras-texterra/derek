from derek.common.feature_extraction.helper import Direction
from derek.data.model import Document


def get_dt_depths_feature(doc: Document):
    head_distances = doc.token_features["dt_head_distances"]
    dependency_tree_depths = [-1] * len(head_distances)

    for i, distance in enumerate(head_distances):
        if dependency_tree_depths[i] >= 0:
            continue

        if distance == 0:
            dependency_tree_depths[i] = 0
            continue

        parents_indexes = [i]  # current token parents indexes
        cur_index = i
        while distance != 0:
            cur_index = cur_index + distance  # index of parent
            if dependency_tree_depths[cur_index] >= 0:
                depth = dependency_tree_depths[cur_index] + 1  # depth to start with
                break
            parents_indexes.append(cur_index)
            distance = head_distances[cur_index]

        else:
            #  we got in root node
            depth = 0

        for parent_idx in parents_indexes[::-1]:
            dependency_tree_depths[parent_idx] = depth
            depth += 1

    assert(-1 not in dependency_tree_depths)

    return dependency_tree_depths


def get_sentence_borders_feature(doc: Document):
    ret = []

    for sent in doc.sentences:
        sent_len = len(sent)

        if sent_len <= 0:
            continue

        ret.append("start")

        if sent_len == 1:
            continue

        for i in range(sent_len - 2):
            ret.append("in")

        ret.append("end")

    return ret


def _find_token_most_nested_entity_type_depth(doc: Document, token_idx):
    token_containing_entities = doc.entities.at_token(token_idx)

    most_nested_entity = min(token_containing_entities, key=lambda x: len(x), default=None)
    depth = len(token_containing_entities)

    if most_nested_entity is None:
        return None, depth
    else:
        return most_nested_entity.type, depth


def get_entities_types_and_depths_features(doc: Document):
    entity_types = []
    entity_depths = []

    for token_idx in range(len(doc.tokens)):
        most_nested_entity_type, depth = _find_token_most_nested_entity_type_depth(doc, token_idx)

        entity_types.append(most_nested_entity_type)
        entity_depths.append(depth)

    return entity_types, entity_depths


def get_dt_deltas_feature(doc: Document, direction, *, precomputed_depths: list = None):
    ret = []

    depths = precomputed_depths
    if depths is None:
        depths = get_dt_depths_feature(doc)

    for sent in doc.sentences:
        if direction == Direction.FORWARD:
            ret.append("$START$")
            for i in range(sent.start_token + 1, sent.end_token):
                ret.append(depths[i] - depths[i-1])
        elif direction == Direction.BACKWARD:
            for i in range(sent.start_token, sent.end_token - 1):
                ret.append(depths[i] - depths[i+1])
            ret.append("$START$")
        else:
            raise Exception("Unknown direction")

    return ret


def get_dt_breakups_feature(doc: Document, direction):
    dt_head_distances = doc.token_features["dt_head_distances"]
    ret = []

    for distance in dt_head_distances:
        if direction == Direction.FORWARD and distance > 0 or direction == Direction.BACKWARD and distance < 0:
            ret.append(True)
        else:
            ret.append(False)

    return ret


def get_morph_features(doc, morph_features):
    token_features = {}
    if 'feats' in doc.token_features:
        for feat in morph_features:
            token_features[feat] = [feats_dict.get(feat, None) for feats_dict in doc.token_features['feats']]

    return token_features
