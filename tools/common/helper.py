def _get_params_combination(base_props: dict, props_list: dict, params=None):
    if not props_list:
        yield {}
        return
    if params is None:
        params = list(props_list.keys())
        params.sort()

    param = params[0]
    if not isinstance(props_list[param], dict) and all([type(x) not in {list, dict} for x in props_list[param]]):
        props_list[param].sort()

    combination = {}

    if isinstance(props_list[param], dict):
        base_param_dict = base_props.get(param, {})
        param_vals = [
            {**base_param_dict, **comb} for comb in _get_params_combination(base_param_dict, props_list[param])
        ]
    else:
        param_vals = props_list[param]

    for val in param_vals:
        combination[param] = val
        if len(params) > 1:
            for other in _get_params_combination(base_props, props_list, params[1:]):
                combination.update(other)
                yield combination
        else:
            yield combination


def get_next_props(base_props: dict, props_list):
    if isinstance(props_list, dict):
        props_list = [props_list]
    for props_dict in props_list:
        combinations = _get_params_combination(base_props, props_dict)
        for combination in combinations:
            props = base_props.copy()
            props.update(combination)
            yield props


def params_to_str(params: dict):
    params_str = '{\n'

    for par, val in sorted(params.items(), key=lambda x: x[0]):
        params_str += par + " : " + str(val) + '\n'

    params_str += '}\n'

    return params_str


def get_fold(objects: list, n_folds, fold_idx):
    if fold_idx >= n_folds:
        raise Exception("Fold idx >= n_folds")

    fold_base_size, remain_objects_num = divmod(len(objects), n_folds)
    fold_start = fold_base_size * fold_idx
    fold_end = fold_base_size * (fold_idx + 1)

    # adding objects remainder for the first folds
    if fold_idx < remain_objects_num:
        fold_start += fold_idx
        fold_end += fold_idx + 1
    else:
        fold_start += remain_objects_num
        fold_end += remain_objects_num

    dev_part = objects[fold_start: fold_end]
    train_part = objects[:fold_start] + objects[fold_end:]
    return train_part, dev_part
