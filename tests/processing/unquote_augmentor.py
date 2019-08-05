import unittest

from derek.data.model import Sentence, Entity, SortedSpansSet, Document
from derek.ner.feature_extraction.augmentations import EntitiesUnquoteAugmentor


class TestEntitiesUnquoteAugmentor(unittest.TestCase):
    def setUp(self) -> None:
        tokens = [
            "Elon", "Musk", "must", "«", "donate", "»", "\'", "Tesla", "\'", "to", "our", "\"", "subscribers", "\"", ".",
            "It", "is", "\"", "important", "\"", "!"
        ]
        sentences = [
            Sentence(0, 15), Sentence(15, 21)
        ]
        entities = [
            Entity("_", 0, 2, "CEO"), Entity("_", 4, 5, "donate"),
            Entity("_", 7, 8, "Tesla"), Entity("_", 12, 13, "subscribers"),

            Entity("_", 15, 16, "It"), Entity("_", 18, 19, "important")
        ]

        nes = SortedSpansSet([
            Entity("_", 6, 9, "Tesla"), Entity("_", 17, 20, "important")
        ])

        token_features = {
            "tokens": list(tokens),
            "pos": [
                "NNP", "NNP", "VB", "QUOTE", "VB", "QUOTE", "QUOTE", "NNP", "QUOTE", "TO", "NNPS", "QUOTE", "NNS", "QUOTE", "DOT",
                "NNP", "VB", "QUOTE", "RB", "QUOTE", "DOT"
            ]
        }

        self.doc = Document("_", tokens, sentences, [], entities, token_features=token_features, extras={"ne": nes})

    def test_fully_augmented(self):
        tokens = [
            "Elon", "Musk", "must", "donate", "Tesla", "to", "our", "subscribers", ".",
            "It", "is", "important", "!"
        ]
        sentences = [
            Sentence(0, 9), Sentence(9, 13)
        ]
        entities = [
            Entity("_", 0, 2, "CEO"), Entity("_", 3, 4, "donate"),
            Entity("_", 4, 5, "Tesla"), Entity("_", 7, 8, "subscribers"),

            Entity("_", 9, 10, "It"), Entity("_", 11, 12, "important")
        ]

        nes = SortedSpansSet([
            Entity("_", 4, 5, "Tesla"), Entity("_", 11, 12, "important")
        ])

        token_features = {
            "tokens": list(tokens),
            "pos": [
                "NNP", "NNP", "VB", "VB", "NNP", "TO", "NNPS", "NNS", "DOT",
                "NNP", "VB", "RB", "DOT"
            ]
        }

        expected_doc = Document("_", tokens, sentences, [], entities, token_features=token_features, extras={"ne": nes})
        to_augment = ["CEO", "donate", "Tesla", "subscribers", "It", "important"]
        actual_doc = EntitiesUnquoteAugmentor(1.0, to_augment).transform(self.doc)

        self.assertEqual(expected_doc, actual_doc)

    def test_partly_augmented(self):
        tokens = [
            "Elon", "Musk", "must", "«", "donate", "»", "Tesla", "to", "our", "subscribers", ".",
            "It", "is", "\"", "important", "\"", "!"
        ]
        sentences = [
            Sentence(0, 11), Sentence(11, 17)
        ]
        entities = [
            Entity("_", 0, 2, "CEO"), Entity("_", 4, 5, "donate"),
            Entity("_", 6, 7, "Tesla"), Entity("_", 9, 10, "subscribers"),

            Entity("_", 11, 12, "It"), Entity("_", 14, 15, "important")
        ]

        nes = SortedSpansSet([
            Entity("_", 6, 7, "Tesla"), Entity("_", 13, 16, "important")
        ])

        token_features = {
            "tokens": list(tokens),
            "pos": [
                "NNP", "NNP", "VB", "QUOTE", "VB", "QUOTE", "NNP", "TO", "NNPS", "NNS", "DOT",
                "NNP", "VB", "QUOTE", "RB", "QUOTE", "DOT"
            ]
        }

        expected_doc = Document("_", tokens, sentences, [], entities, token_features=token_features, extras={"ne": nes})
        to_augment = ["CEO", "Tesla", "subscribers", "It"]
        actual_doc = EntitiesUnquoteAugmentor(1.0, to_augment).transform(self.doc)

        self.assertEqual(expected_doc, actual_doc)
