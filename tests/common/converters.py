from derek.common.feature_extraction.converters import create_signed_integers_converter, \
    create_unsigned_integers_converter, create_categorical_converter
import unittest
from operator import getitem


class TestCategoricalConverterClass(unittest.TestCase):
    def test_categorical_converter_1(self):
        categories_set = {"$HABITAT$", "$BACTERIA$", "hello", "laboratory"}
        converter = create_categorical_converter(categories_set, zero_padding=False, has_oov=False)
        indexed_categories = {
            "$BACTERIA$": 0,
            "$HABITAT$": 1,
            "hello": 2,
            "laboratory": 3
        }

        converter_indexed_categories = {key: converter[key] for key in indexed_categories}
        self.assertEqual(indexed_categories, converter_indexed_categories)

        error_keys = ["$OOV$", "$PADDING$", 0, 1, "privet"]
        for key in error_keys:
            self.assertRaises(KeyError, getitem, converter, key)

    def test_categorical_converter_2(self):
        categories_set = {"$HABITAT$", "$BACTERIA$", "hello", "laboratory"}
        converter = create_categorical_converter(categories_set, zero_padding=True, has_oov=False)
        indexed_categories = {
            "$PADDING$": 0,
            "$BACTERIA$": 1,
            "$HABITAT$": 2,
            "hello": 3,
            "laboratory": 4
        }

        converter_indexed_categories = {key: converter[key] for key in indexed_categories}
        self.assertEqual(indexed_categories, converter_indexed_categories)

        error_keys = ["$OOV$", 0, 1, "privet"]
        for key in error_keys:
            self.assertRaises(KeyError, getitem, converter, key)
    
    def test_categorical_converter_3(self):
        categories_set = {"$HABITAT$", "$BACTERIA$", "hello", "laboratory"}
        converter = create_categorical_converter(categories_set, zero_padding=True, has_oov=True)
        indexed_categories = {
            "$PADDING$": 0,
            "$BACTERIA$": 1,
            "$HABITAT$": 2,
            "hello": 3,
            "laboratory": 4,
            "$OOV$": 5,
            '1': 5,
            1: 5,
            'privet': 5,
        }

        converter_indexed_categories = {key: converter[key] for key in indexed_categories}
        self.assertEqual(indexed_categories, converter_indexed_categories)
    
    def test_categorical_converter_4(self):
        categories_set = {"$HABITAT$", "$BACTERIA$", "hello", "laboratory"}
        converter = create_categorical_converter(categories_set, zero_padding=False, has_oov=True)
        indexed_categories = {
            "$BACTERIA$": 0,
            "$HABITAT$": 1,
            "hello": 2,
            "laboratory": 3,
            "$OOV$": 4,
            '1': 4,
            1: 4,
            'privet': 4,
        }

        converter_indexed_categories = {key: converter[key] for key in indexed_categories}
        self.assertEqual(indexed_categories, converter_indexed_categories)
        self.assertRaises(KeyError, getitem, converter, "$PADDING$")

    def test_custom_oov_not_in_set(self):
        categories_set = {"$HABITAT$", "$BACTERIA$", "hello", "laboratory"}
        converter = create_categorical_converter(categories_set, zero_padding=False, has_oov=True, oov_object="$CUSTOM$")
        indexed_categories = {
            "$BACTERIA$": 0,
            "$HABITAT$": 1,
            "hello": 2,
            "laboratory": 3,
            "$CUSTOM$": 4,
            '1': 4,
            1: 4,
            'privet': 4,
        }

        converter_indexed_categories = {key: converter[key] for key in indexed_categories}
        self.assertEqual(indexed_categories, converter_indexed_categories)

    def test_custom_oov_in_set(self):
        categories_set = {"$HABITAT$", "$BACTERIA$", "hello", "laboratory"}
        converter = create_categorical_converter(categories_set, zero_padding=False, has_oov=True, oov_object="hello")
        indexed_categories = {
            "$BACTERIA$": 0,
            "$HABITAT$": 1,
            "hello": 2,
            "laboratory": 3,
            '1': 2,
            1: 2,
            'privet': 2,
        }

        converter_indexed_categories = {key: converter[key] for key in indexed_categories}
        self.assertEqual(indexed_categories, converter_indexed_categories)


class TestUnsignedIntegerConverterClass(unittest.TestCase):
    def test_unsigned_integers_converter_1(self):
        right_border = 4
        additional_labels = set()
        converter = create_unsigned_integers_converter(
            right_border, additional_labels=additional_labels, zero_padding=False)
        indexed_integers = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 5,
            7: 5
        }

        converter_indexed_integers = {key: converter[key] for key in indexed_integers}
        self.assertEqual(indexed_integers, converter_indexed_integers)

        error_keys = ["$OOV$", "$PADDING$", -1, -2, -3, -1000, "habitat"]
        for key in error_keys:
            self.assertRaises(KeyError, getitem, converter, key)

    def test_unsigned_integers_converter_2(self):
        right_border = 0
        additional_labels = set()
        converter = create_unsigned_integers_converter(
            right_border, additional_labels=additional_labels, zero_padding=False)
        indexed_integers = {
            0: 0,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
            7: 1
        }

        converter_indexed_integers = {key: converter[key] for key in indexed_integers}
        self.assertEqual(indexed_integers, converter_indexed_integers)

        error_keys = ["$OOV$", "$PADDING$", -1, -2, -3, -1000, "habitat"]
        for key in error_keys:
            self.assertRaises(KeyError, getitem, converter, key)

    def test_unsigned_integers_converter_3(self):
        right_border = 4
        additional_labels = set()
        converter = create_unsigned_integers_converter(
            right_border, additional_labels=additional_labels, zero_padding=True)
        indexed_integers = {
            "$PADDING$": 0,
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: 6,
            7: 6
        }

        converter_indexed_integers = {key: converter[key] for key in indexed_integers}
        self.assertEqual(indexed_integers, converter_indexed_integers)

        error_keys = ["$OOV$", -1, -2, -3, -1000, "habitat"]
        for key in error_keys:
            self.assertRaises(KeyError, getitem, converter, key)

    def test_unsigned_integers_converter_4(self):
        right_border = 4
        additional_labels = {"$SENTEND$", "$HABITAT$"}
        converter = create_unsigned_integers_converter(
            right_border, additional_labels=additional_labels, zero_padding=True)
        indexed_integers = {
            "$PADDING$": 0,
            "$HABITAT$": 1,
            "$SENTEND$": 2,
            0: 3,
            1: 4,
            2: 5,
            3: 6,
            4: 7,
            5: 8,
            6: 8,
            7: 8
        }

        converter_indexed_integers = {key: converter[key] for key in indexed_integers}
        self.assertEqual(indexed_integers, converter_indexed_integers)

        error_keys = ["$OOV$", -1, -2, -3, -1000, "habitat"]
        for key in error_keys:
            self.assertRaises(KeyError, getitem, converter, key)

    def test_unsigned_integers_converter_5(self):
        right_border = 4
        additional_labels = {"$SENTEND$", "$HABITAT$"}
        converter = create_unsigned_integers_converter(
            right_border, additional_labels=additional_labels, zero_padding=False)
        indexed_integers = {
            "$HABITAT$": 0,
            "$SENTEND$": 1,
            0: 2,
            1: 3,
            2: 4,
            3: 5,
            4: 6,
            5: 7,
            6: 7,
            7: 7
        }

        converter_indexed_integers = {key: converter[key] for key in indexed_integers}
        self.assertEqual(indexed_integers, converter_indexed_integers)

        error_keys = ["$PADDING$", "$OOV$", -1, -2, -3, -1000, "habitat"]
        for key in error_keys:
            self.assertRaises(KeyError, getitem, converter, key)

    def test_unsigned_integers_converter_6(self):
        right_border = 0
        additional_labels = {"$SENTEND$", "$HABITAT$"}
        converter = create_unsigned_integers_converter(
            right_border, additional_labels=additional_labels, zero_padding=False)
        indexed_integers = {
            "$HABITAT$": 0,
            "$SENTEND$": 1,
            0: 2,
            1: 3,
            2: 3,
            3: 3,
            4: 3,
            5: 3,
            6: 3,
            7: 3
        }

        converter_indexed_integers = {key: converter[key] for key in indexed_integers}
        self.assertEqual(indexed_integers, converter_indexed_integers)

        error_keys = ["$PADDING$", "$OOV$", -1, -2, -3, -1000, "habitat"]
        for key in error_keys:
            self.assertRaises(KeyError, getitem, converter, key)


class TestSignedIntegersConverterClass(unittest.TestCase):
    def test_signed_integers_converter_1(self):
        absolute_offset = 4
        additional_labels = set()
        converter = create_signed_integers_converter(
            absolute_offset, additional_labels=additional_labels, zero_padding=False)
        indexed_integers = {
            -1: 0,
            -2: 1,
            -3: 2,
            -4: 3,
            -5: 4,
            -6: 4,
            -7: 4,
            -8: 4,
            0: 5,
            1: 6,
            2: 7,
            3: 8,
            4: 9,
            5: 10,
            6: 10,
            7: 10
        }

        converter_indexed_integers = {key: converter[key] for key in indexed_integers}
        self.assertEqual(indexed_integers, converter_indexed_integers)

        error_keys = ["$OOV$", "$PADDING$", "habitat"]
        for key in error_keys:
            self.assertRaises(KeyError, getitem, converter, key)

    def test_signed_integers_converter_2(self):
        absolute_offset = 0
        additional_labels = set()
        converter = create_signed_integers_converter(
            absolute_offset, additional_labels=additional_labels, zero_padding=False)
        indexed_integers = {
            -1: 0,
            -2: 0,
            -3: 0,
            -4: 0,
            -5: 0,
            -6: 0,
            -7: 0,
            -8: 0,
            0: 1,
            1: 2,
            2: 2,
            3: 2,
            4: 2,
            5: 2,
            6: 2,
            7: 2
        }

        converter_indexed_integers = {key: converter[key] for key in indexed_integers}
        self.assertEqual(indexed_integers, converter_indexed_integers)

        error_keys = ["$OOV$", "$PADDING$", "habitat"]
        for key in error_keys:
            self.assertRaises(KeyError, getitem, converter, key)

    def test_signed_integers_converter_3(self):
        absolute_offset = 4
        additional_labels = set()
        converter = create_signed_integers_converter(
            absolute_offset, additional_labels=additional_labels, zero_padding=True)
        indexed_integers = {
            "$PADDING$": 0,
            -1: 1,
            -2: 2,
            -3: 3,
            -4: 4,
            -5: 5,
            -6: 5,
            -7: 5,
            -8: 5,
            0: 6,
            1: 7,
            2: 8,
            3: 9,
            4: 10,
            5: 11,
            6: 11,
            7: 11
        }

        converter_indexed_integers = {key: converter[key] for key in indexed_integers}
        self.assertEqual(indexed_integers, converter_indexed_integers)

        error_keys = ["$OOV$", "habitat"]
        for key in error_keys:
            self.assertRaises(KeyError, getitem, converter, key)

    def test_signed_integers_converter_4(self):
        absolute_offset = 4
        additional_labels = {"privet"}
        converter = create_signed_integers_converter(
            absolute_offset, additional_labels=additional_labels, zero_padding=True)
        indexed_integers = {
            "$PADDING$": 0,
            -1: 1,
            -2: 2,
            -3: 3,
            -4: 4,
            -5: 5,
            -6: 5,
            -7: 5,
            -8: 5,
            0: 6,
            1: 7,
            2: 8,
            3: 9,
            4: 10,
            5: 11,
            6: 11,
            7: 11,
            "privet": 12
        }

        converter_indexed_integers = {key: converter[key] for key in indexed_integers}
        self.assertEqual(indexed_integers, converter_indexed_integers)

        error_keys = ["$OOV$", "$habitat$"]
        for key in error_keys:
            self.assertRaises(KeyError, getitem, converter, key)

    def test_signed_integers_converter_5(self):
        absolute_offset = 4
        additional_labels = {"privet"}
        converter = create_signed_integers_converter(
            absolute_offset, additional_labels=additional_labels, zero_padding=False)
        indexed_integers = {
            -1: 0,
            -2: 1,
            -3: 2,
            -4: 3,
            -5: 4,
            -6: 4,
            -7: 4,
            -8: 4,
            0: 5,
            1: 6,
            2: 7,
            3: 8,
            4: 9,
            5: 10,
            6: 10,
            7: 10,
            "privet": 11
        }

        converter_indexed_integers = {key: converter[key] for key in indexed_integers}
        self.assertEqual(indexed_integers, converter_indexed_integers)

        error_keys = ["$OOV$", "$habitat$", "$PADDING$"]
        for key in error_keys:
            self.assertRaises(KeyError, getitem, converter, key)


if __name__ == "__main__":
    unittest.main()
