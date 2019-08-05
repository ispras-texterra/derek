import unittest

from derek.data.helper import adjust_sentences
from derek.data.model import Sentence, Entity


class PreprocessorTest(unittest.TestCase):

    def test_no_entities(self):
        sentences = [Sentence(0, 2), Sentence(2, 3)]
        entities = []
        got_sentences = adjust_sentences(sentences, entities)
        self.assertEqual(sentences, got_sentences)

    def test_one_entity(self):
        sentences = [Sentence(0, 2), Sentence(2, 3), Sentence(3, 5)]
        entities = [Entity('', 2, 4, 'test')]

        expected_sentences = [Sentence(0, 2), Sentence(2, 5)]
        got_sentences = adjust_sentences(sentences, entities)
        self.assertEqual(expected_sentences, got_sentences)

    def test_multi_sentence(self):
        sentences = [Sentence(0, 2), Sentence(2, 3), Sentence(3, 5), Sentence(5, 8), Sentence(8, 10), Sentence(10, 15)]
        entities = [Entity('', 2, 9, 'test')]

        expected_sentences = [Sentence(0, 2), Sentence(2, 10), Sentence(10, 15)]
        got_sentences = adjust_sentences(sentences, entities)
        self.assertEqual(expected_sentences, got_sentences)

    def test_two_entities(self):
        sentences = [Sentence(0, 2), Sentence(2, 3), Sentence(3, 5), Sentence(5, 8), Sentence(8, 10), Sentence(10, 15)]
        entities = [Entity('', 2, 4, 'test'), Entity('', 4, 9, 'test')]

        expected_sentences = [Sentence(0, 2), Sentence(2, 10), Sentence(10, 15)]
        got_sentences = adjust_sentences(sentences, entities)
        self.assertEqual(expected_sentences, got_sentences)

    def test_two_entities_separated(self):
        sentences = [Sentence(0, 2), Sentence(2, 3), Sentence(3, 5), Sentence(5, 8), Sentence(8, 10), Sentence(10, 15)]
        entities = [Entity('', 2, 4, 'test'), Entity('', 9, 11, 'test')]

        expected_sentences = [Sentence(0, 2), Sentence(2, 5), Sentence(5, 8), Sentence(8, 15)]
        got_sentences = adjust_sentences(sentences, entities)
        self.assertEqual(expected_sentences, got_sentences)

    def test_contained_entities(self):
        sentences = [Sentence(0, 2), Sentence(2, 3), Sentence(3, 5), Sentence(5, 8), Sentence(8, 10), Sentence(10, 15)]
        entities = [Entity('', 2, 6, 'test'), Entity('', 6, 7, 'test'), Entity('', 7, 8, 'test')]

        expected_sentences = [Sentence(0, 2), Sentence(2, 8), Sentence(8, 10), Sentence(10, 15)]
        got_sentences = adjust_sentences(sentences, entities)
        self.assertEqual(expected_sentences, got_sentences)