from typing import Any


def create_categorical_converter(categories_set: set, zero_padding=True, has_oov=False, oov_object: Any = "$OOV$"):
    ret = CategoricalConverter(has_oov, oov_object)

    # sorting is applied to ensure reproducibility of results
    collection = sorted(categories_set, key=lambda x: str(x))

    if zero_padding:
        ret['$PADDING$'] = len(ret)

    for elem in collection:
        if elem == '$PADDING$':
            continue
        ret[elem] = len(ret)

    if has_oov and oov_object not in ret:
        ret[oov_object] = len(ret)

    return ret


def create_signed_integers_converter(right_border, left_border=None, *, mapper=None,
                                     additional_labels: set = None, zero_padding=True):
    if left_border is None:
        left_border = -right_border

    if additional_labels is None:
        additional_labels = set()

    converter = create_categorical_converter(set(range(left_border - 1, right_border + 2)).union(additional_labels),
                                             zero_padding=zero_padding)

    if zero_padding:
        additional_labels.add("$PADDING$")

    return SignedIntegersConverter(right_border, left_border, converter, additional_labels, mapper)


def create_unsigned_integers_converter(right_border, *, mapper=None, additional_labels: set = None, zero_padding=True):
    if additional_labels is None:
        additional_labels = set()

    converter = create_categorical_converter(set(range(0, right_border + 2)).union(additional_labels),
                                             zero_padding=zero_padding)

    if zero_padding:
        additional_labels.add("$PADDING$")

    return UnsignedIntegersConverter(right_border, converter, additional_labels, mapper)


def create_signed_log_integers_converter(right_border, left_border=None, *,
                                         additional_labels: set = None, zero_padding=True):
    return create_signed_integers_converter(__signed_log(right_border), __signed_log(left_border), mapper=__signed_log,
                                            additional_labels=additional_labels, zero_padding=zero_padding)


def create_unsigned_log_integers_converter(right_border, *, additional_labels: set = None, zero_padding=True):
    return create_unsigned_integers_converter(__signed_log(right_border), mapper=__signed_log,
                                              additional_labels=additional_labels, zero_padding=zero_padding)


def __signed_log(x: int):  # 0 -> 0, if x > 0: x -> floor(log2(x)) + 1
    if x is None:
        return None
    return x.bit_length() if x >= 0 else -x.bit_length()


class CategoricalConverter(dict):
    def __init__(self, has_oov: bool, oov_object: Any):
        super().__init__()
        self._has_oov = has_oov
        self._oov_object = oov_object

    def get_reversed_converter(self):
        reversed_converter = {}

        for key, value in self.items():
            reversed_converter[value] = key

        assert (len(reversed_converter) == len(self))

        return reversed_converter

    def __getitem__(self, item):
        storage = super(CategoricalConverter, self)
        has_oov = getattr(self, "_has_oov", "$OOV$" in self)  # not to break existing models with old converter

        if item != "$PADDING$" and has_oov:
            oov_object = getattr(self, "_oov_object", "$OOV$")
            return storage.get(item, storage.__getitem__(oov_object))
        return storage.__getitem__(item)


class SignedIntegersConverter:
    def __init__(self, right_border, left_border, converter: CategoricalConverter, additional_labels: set, mapper=None):
        self.right_border = right_border
        self.left_border = left_border
        self.converter = converter
        self.additional_labels = additional_labels
        self.mapper = mapper

    def __getitem__(self, key):
        if key not in self:
            raise KeyError

        if key in self.additional_labels:
            return self.converter[key]

        if self.mapper is not None:
            key = self.mapper(key)

        if key > self.right_border:
            key = self.right_border + 1
        elif key < self.left_border:
            key = self.left_border - 1

        # Convert distance to index
        return self.converter[key]

    def __contains__(self, item):
        return item in self.additional_labels or isinstance(item, int)

    def __len__(self):
        return len(self.converter)

    def get_reversed_converter(self):
        return self.converter.get_reversed_converter()


class UnsignedIntegersConverter:
    def __init__(self, right_border, converter: CategoricalConverter, additional_labels: set, mapper=None):
        self.right_border = right_border
        self.converter = converter
        self.additional_labels = additional_labels
        self.mapper = mapper

    def __getitem__(self, key):
        if key not in self:
            raise KeyError

        if key in self.additional_labels:
            return self.converter[key]

        if self.mapper is not None:
            key = self.mapper(key)

        if key > self.right_border:
            key = self.right_border + 1

        # Convert depth to index
        return self.converter[key]

    def __contains__(self, item):
        return item in self.additional_labels or isinstance(item, int) and item >= 0

    def __len__(self):
        return len(self.converter)

    def get_reversed_converter(self):
        return self.converter.get_reversed_converter()
