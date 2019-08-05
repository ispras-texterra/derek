import unittest
from derek.ner.feature_extraction.labelling_strategies import IOLabellingStrategy, BIO2LabellingStrategy, \
    BIOLabellingStrategy, BILOULabellingStrategy
from derek.data.model import Sentence, Entity


class TestNERLabellingStrategies(unittest.TestCase):
    def setUp(self) -> None:
        self.ent_types = {"T1", "T2", "T3"}

    def test_io_strategy_encoding(self):
        strategy = IOLabellingStrategy()
        sent = Sentence(110, 120)
        ents = [
            Entity("_", 110, 112, "T1"),
            Entity("_", 112, 113, "T2"),
            Entity("_", 114, 115, "T3"),
            Entity("_", 115, 118, "T3"),
            Entity("_", 119, 120, "T1")
        ]

        expected_possible_categories = {"O", "I-T1", "I-T2", "I-T3"}
        actual_possible_categories = strategy.get_possible_categories(self.ent_types)
        self.assertEqual(expected_possible_categories, actual_possible_categories)

        expected_encoding = ["I-T1", "I-T1", "I-T2", "O", "I-T3", "I-T3", "I-T3", "I-T3", "O", "I-T1"]
        actual_encoding = strategy.encode_labels(sent, ents)
        self.assertEqual(expected_encoding, actual_encoding)

    def test_io_strategy_decoding(self):
        strategy = IOLabellingStrategy()
        sent = Sentence(110, 120)
        labels = ["I-T1", "I-T1", "I-T2", "O", "I-T3", "I-T3", "I-T3", "I-T3", "O", "I-T1"]

        expected = [
            Entity("generated", 110, 112, "T1"),
            Entity("generated", 112, 113, "T2"),
            Entity("generated", 114, 118, "T3"),
            Entity("generated", 119, 120, "T1")
        ]

        actual = strategy.decode_labels(sent, labels)
        self.assertEqual(expected, actual)

    def test_bio2_strategy_encoding(self):
        strategy = BIO2LabellingStrategy()
        sent = Sentence(110, 120)
        ents = [
            Entity("_", 110, 112, "T1"),
            Entity("_", 112, 113, "T2"),
            Entity("_", 114, 115, "T3"),
            Entity("_", 115, 118, "T3"),
            Entity("_", 119, 120, "T1")
        ]

        expected_possible_categories = {"O", "I-T1", "I-T2", "I-T3", "B-T1", "B-T2", "B-T3"}
        actual_possible_categories = strategy.get_possible_categories(self.ent_types)
        self.assertEqual(expected_possible_categories, actual_possible_categories)

        expected_encoding = ["I-T1", "I-T1", "I-T2", "O", "I-T3", "B-T3", "I-T3", "I-T3", "O", "I-T1"]
        actual_encoding = strategy.encode_labels(sent, ents)
        self.assertEqual(expected_encoding, actual_encoding)

    def test_bio2_strategy_decoding(self):
        strategy = BIO2LabellingStrategy()
        sent = Sentence(110, 124)
        labels = ["I-T1", "B-T1", "B-T2", "O", "I-T3", "B-T3", "I-T3", "I-T3", "O", "I-T1", "O", "B-T1", "B-T1", "I-T1"]

        expected = [
            Entity("generated", 110, 111, "T1"),
            Entity("generated", 111, 112, "T1"),
            Entity("generated", 112, 113, "T2"),
            Entity("generated", 114, 115, "T3"),
            Entity("generated", 115, 118, "T3"),
            Entity("generated", 119, 120, "T1"),
            Entity("generated", 121, 122, "T1"),
            Entity("generated", 122, 124, "T1")
        ]

        actual = strategy.decode_labels(sent, labels)
        self.assertEqual(expected, actual)

    def test_bio_strategy_encoding(self):
        strategy = BIOLabellingStrategy()
        sent = Sentence(110, 120)
        ents = [
            Entity("_", 110, 112, "T1"),
            Entity("_", 112, 113, "T2"),
            Entity("_", 114, 115, "T3"),
            Entity("_", 115, 118, "T3"),
            Entity("_", 119, 120, "T1")
        ]

        expected_possible_categories = {"O", "I-T1", "I-T2", "I-T3", "B-T1", "B-T2", "B-T3"}
        actual_possible_categories = strategy.get_possible_categories(self.ent_types)
        self.assertEqual(expected_possible_categories, actual_possible_categories)

        expected_encoding = ["B-T1", "I-T1", "B-T2", "O", "B-T3", "B-T3", "I-T3", "I-T3", "O", "B-T1"]
        actual_encoding = strategy.encode_labels(sent, ents)
        self.assertEqual(expected_encoding, actual_encoding)

    def test_bio_strategy_decoding(self):
        strategy = BIOLabellingStrategy()
        sent = Sentence(110, 123)
        labels = ["B-T1", "I-T1", "I-T2", "O", "B-T3", "B-T3", "I-T3", "I-T3", "O", "B-T1", "O", "I-T3", "O"]

        expected = [
            Entity("generated", 110, 112, "T1"),
            Entity("generated", 112, 113, "T2"),
            Entity("generated", 114, 115, "T3"),
            Entity("generated", 115, 118, "T3"),
            Entity("generated", 119, 120, "T1"),
            Entity("generated", 121, 122, "T3")
        ]

        actual = strategy.decode_labels(sent, labels)
        self.assertEqual(expected, actual)

    def test_bilou_strategy_encoding(self):
        strategy = BILOULabellingStrategy()
        sent = Sentence(110, 120)
        ents = [
            Entity("_", 110, 112, "T1"),
            Entity("_", 112, 113, "T2"),
            Entity("_", 114, 115, "T3"),
            Entity("_", 115, 118, "T3"),
            Entity("_", 119, 120, "T1")
        ]

        expected_possible_categories = {
            "O",
            "I-T1", "I-T2", "I-T3",
            "B-T1", "B-T2", "B-T3",
            "L-T1", "L-T2", "L-T3",
            "U-T1", "U-T2", "U-T3"
        }
        actual_possible_categories = strategy.get_possible_categories(self.ent_types)
        self.assertEqual(expected_possible_categories, actual_possible_categories)

        expected_encoding = ["B-T1", "L-T1", "U-T2", "O", "U-T3", "B-T3", "I-T3", "L-T3", "O", "U-T1"]
        actual_encoding = strategy.encode_labels(sent, ents)
        self.assertEqual(expected_encoding, actual_encoding)

    def test_bilou_strategy_decoding(self):
        strategy = BILOULabellingStrategy()
        sent = Sentence(110, 125)
        labels = ["B-T1", "I-T1", "U-T1", "O", "I-T3", "B-T3", "L-T3", "L-T3", "O",
                  "U-T1", "L-T1", "U-T1", "I-T1", "U-T1", "B-T1"]

        expected = [
            Entity("generated", 110, 112, "T1"),
            Entity("generated", 112, 113, "T1"),
            Entity("generated", 114, 115, "T3"),
            Entity("generated", 115, 117, "T3"),
            Entity("generated", 117, 118, "T3"),
            Entity("generated", 119, 120, "T1"),
            Entity("generated", 120, 121, "T1"),
            Entity("generated", 121, 122, "T1"),
            Entity("generated", 122, 123, "T1"),
            Entity("generated", 123, 124, "T1"),
            Entity("generated", 124, 125, "T1")
        ]

        actual = strategy.decode_labels(sent, labels)
        self.assertEqual(expected, actual)
