import unittest
from tools.common.helper import get_next_props, get_fold


class GetNextPropsTest(unittest.TestCase):
    def test_no_lst(self):
        base_props = {"a": 1, "b": 2, "c": {1: 3, 2: 4}}
        lst = {}

        expected_combinations = [base_props]
        actual_combinations = list(get_next_props(base_props, lst))
        self.assertEqual(expected_combinations, actual_combinations)

    def test_simple_lst(self):
        base_props = {"a": 1, "b": 2, "c": {1: 3, 2: 4}}
        lst = {"a": [4, 1], "d": [7, 3]}

        expected_combinations = [
            {"a": 1, "b": 2, "c": {1: 3, 2: 4}, "d": 3},
            {"a": 1, "b": 2, "c": {1: 3, 2: 4}, "d": 7},
            {"a": 4, "b": 2, "c": {1: 3, 2: 4}, "d": 3},
            {"a": 4, "b": 2, "c": {1: 3, 2: 4}, "d": 7}
        ]
        actual_combinations = list(get_next_props(base_props, lst))
        self.assertEqual(expected_combinations, actual_combinations)

    def test_lst_with_nested_dict(self):
        base_props = {"a": 1, "b": 2, "c": {1: 3, 2: 4}}
        lst = {"a": [4, 1], "c": {1: [70, 50]}}

        expected_combinations = [
            {"a": 1, "b": 2, "c": {1: 50, 2: 4}},
            {"a": 1, "b": 2, "c": {1: 70, 2: 4}},
            {"a": 4, "b": 2, "c": {1: 50, 2: 4}},
            {"a": 4, "b": 2, "c": {1: 70, 2: 4}}
        ]
        actual_combinations = list(get_next_props(base_props, lst))
        self.assertEqual(expected_combinations, actual_combinations)

    def test_lst_with_nested_dict_in_nested_dict(self):
        base_props = {"a": 1, "b": 2, "c": {1: 3, 2: {7: 8, 10: 19}}}
        lst = {"a": [4, 1], "c": {1: [70, 50], 2: {7: [10, 12]}}}

        expected_combinations = [
            {"a": 1, "b": 2, "c": {1: 50, 2: {7: 10, 10: 19}}},
            {"a": 1, "b": 2, "c": {1: 50, 2: {7: 12, 10: 19}}},
            {"a": 1, "b": 2, "c": {1: 70, 2: {7: 10, 10: 19}}},
            {"a": 1, "b": 2, "c": {1: 70, 2: {7: 12, 10: 19}}},
            {"a": 4, "b": 2, "c": {1: 50, 2: {7: 10, 10: 19}}},
            {"a": 4, "b": 2, "c": {1: 50, 2: {7: 12, 10: 19}}},
            {"a": 4, "b": 2, "c": {1: 70, 2: {7: 10, 10: 19}}},
            {"a": 4, "b": 2, "c": {1: 70, 2: {7: 12, 10: 19}}}
        ]
        actual_combinations = list(get_next_props(base_props, lst))
        self.assertEqual(expected_combinations, actual_combinations)

    def test_lst_with_two_nested_dicts(self):
        base_props = {"a": {100: 1000, 200: 2000}, "b": 2, "c": {1: 3, 2: 20, 3: 40}}
        lst = {"a": {100: [0, 1]}, "c": {1: [1], 2: [3, 4], 3: [7, 8]}}

        expected_combinations = [
            {"a": {100: 0, 200: 2000}, "b": 2, "c": {1: 1, 2: 3, 3: 7}},
            {"a": {100: 0, 200: 2000}, "b": 2, "c": {1: 1, 2: 3, 3: 8}},
            {"a": {100: 0, 200: 2000}, "b": 2, "c": {1: 1, 2: 4, 3: 7}},
            {"a": {100: 0, 200: 2000}, "b": 2, "c": {1: 1, 2: 4, 3: 8}},

            {"a": {100: 1, 200: 2000}, "b": 2, "c": {1: 1, 2: 3, 3: 7}},
            {"a": {100: 1, 200: 2000}, "b": 2, "c": {1: 1, 2: 3, 3: 8}},
            {"a": {100: 1, 200: 2000}, "b": 2, "c": {1: 1, 2: 4, 3: 7}},
            {"a": {100: 1, 200: 2000}, "b": 2, "c": {1: 1, 2: 4, 3: 8}},
        ]
        actual_combinations = list(get_next_props(base_props, lst))
        self.assertEqual(expected_combinations, actual_combinations)


class GetFoldTest(unittest.TestCase):
    def test_no_remainder(self):
        objects = [0, 1, 2, 3, 4, 5]
        n_folds = 3

        split_0 = [2, 3, 4, 5], [0, 1]
        split_1 = [0, 1, 4, 5], [2, 3]
        split_2 = [0, 1, 2, 3], [4, 5]

        for idx, split in enumerate((split_0, split_1, split_2)):
            self.assertEqual(split, get_fold(objects, n_folds, idx))

    def test_with_remainder(self):
        objects = [0, 1, 2, 3, 4, 5, 6, 7]
        n_folds = 3

        split_0 = [3, 4, 5, 6, 7], [0, 1, 2]
        split_1 = [0, 1, 2, 6, 7], [3, 4, 5]
        split_2 = [0, 1, 2, 3, 4, 5], [6, 7]

        for idx, split in enumerate((split_0, split_1, split_2)):
            self.assertEqual(split, get_fold(objects, n_folds, idx))

    def test_single_fold(self):
        objects = [0, 1, 2, 3]
        expected = [], [0, 1, 2, 3]

        self.assertEqual(expected, get_fold(objects, 1, 0))
