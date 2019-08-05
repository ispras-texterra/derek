import unittest

from derek.coref.data.model import CoreferenceChain
from derek.data.model import Sentence, Entity, Document
from derek.ner.post_processing.process_similar_entities import chain_similar_entities, compare_entities_by_tokens, \
    unify_types_of_similar_entities


class ProcessSimilarEntitiesTest(unittest.TestCase):
    def setUp(self) -> None:
        tokens = [
            "Лига", "чемпионов", "пройдет", "в", "Стамбуле", ".",
            "Стамбул", "примет", "Лигу", "чемпионов", ".",
            "Лига", "пройдет", "в", "двадцатый", "раз", ".",
            "Арсен", "Венгер", "руководит", "Арсеналом", ".",
            "Арсен", "сказал", "что-то", "."
        ]

        sentences = [
            Sentence(0, 6),
            Sentence(6, 11),
            Sentence(11, 17),
            Sentence(17, 22),
            Sentence(22, 26)
        ]

        entities = [
            Entity("_", 0, 2, "League1"), Entity("_", 4, 5, "Location1"),
            Entity("_", 6, 7, "Location2"), Entity("_", 8, 10, "League"),
            Entity("_", 11, 12, "League"),
            Entity("_", 17, 19, "Person"), Entity("_", 20, 21, "Organization"),
            Entity("_", 22, 23, "Person")
        ]

        lemmas = [
            "Лига", "чемпион", "проходить", "в", "Стамбул", "",
            "Стамбул", "принимать", "Лига", "чемпион", "",
            "Лига", "проходить", "в", "двадцать", "раз", "",
            "Арсен", "Венгер", "руководить", "Арсенал", "",
            "Арсен", "сказать", "что-то", ""
        ]

        self.doc = Document("_", tokens, sentences, [], entities, token_features={"lemmas": lemmas})

    def test_compare_entities_same_lengths(self):
        #  Стамбуле - Стамбул
        self.assertTrue(compare_entities_by_tokens(self.doc, self.doc.entities[1], self.doc.entities[2]))
        #  Стамбул - Лига
        self.assertFalse(compare_entities_by_tokens(self.doc, self.doc.entities[2], self.doc.entities[4]))
        #  Лига - Арсеналом
        self.assertFalse(compare_entities_by_tokens(self.doc, self.doc.entities[4], self.doc.entities[6]))
        #  Арсеналом - Арсен
        self.assertFalse(compare_entities_by_tokens(self.doc, self.doc.entities[6], self.doc.entities[7]))
        #  Лига чемпионов - Лигу чемпионов
        self.assertTrue(compare_entities_by_tokens(self.doc, self.doc.entities[0], self.doc.entities[3]))
        #  Лига чемпионов - Арсен Венгер
        self.assertFalse(compare_entities_by_tokens(self.doc, self.doc.entities[0], self.doc.entities[5]))

    def test_compare_entities_different_lengths(self):
        #  Лига чемпионов - Лига
        self.assertTrue(compare_entities_by_tokens(self.doc, self.doc.entities[0], self.doc.entities[4]))
        self.assertTrue(compare_entities_by_tokens(self.doc, self.doc.entities[4], self.doc.entities[0]))
        #  Лигу чемпионов - Лига
        self.assertTrue(compare_entities_by_tokens(self.doc, self.doc.entities[3], self.doc.entities[4]))
        self.assertTrue(compare_entities_by_tokens(self.doc, self.doc.entities[4], self.doc.entities[3]))
        #  Арсеналом - Арсен Венгер
        self.assertFalse(compare_entities_by_tokens(self.doc, self.doc.entities[5], self.doc.entities[6]))
        self.assertFalse(compare_entities_by_tokens(self.doc, self.doc.entities[6], self.doc.entities[5]))
        #  Арсен - Арсен Венгер
        self.assertTrue(compare_entities_by_tokens(self.doc, self.doc.entities[5], self.doc.entities[7]))
        self.assertTrue(compare_entities_by_tokens(self.doc, self.doc.entities[7], self.doc.entities[5]))

    def test_chain_similar_entities(self):
        expected = [
            # Лига чемпионов - Лигу чемпионов - лига
            CoreferenceChain([self.doc.entities[0], self.doc.entities[3], self.doc.entities[4]]),
            # Стамбуле - Стамбул
            CoreferenceChain([self.doc.entities[1], self.doc.entities[2]]),
            # Арсен Венгер - Арсен
            CoreferenceChain([self.doc.entities[5], self.doc.entities[7]]),
            # Арсенал
            CoreferenceChain([self.doc.entities[6]]),
        ]

        self.assertListEqual(expected, chain_similar_entities(self.doc, self.doc.entities))

    def test_unify_types(self):
        expected = [
            Entity("_", 0, 2, "League"), Entity("_", 4, 5, "Location1"),
            Entity("_", 6, 7, "Location1"), Entity("_", 8, 10, "League"),
            Entity("_", 11, 12, "League"),
            Entity("_", 17, 19, "Person"), Entity("_", 20, 21, "Organization"),
            Entity("_", 22, 23, "Person")
        ]
        self.assertListEqual(expected, unify_types_of_similar_entities(self.doc, self.doc.entities))
