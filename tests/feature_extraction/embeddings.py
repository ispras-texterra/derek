import unittest

from numpy import array

from derek.common.feature_extraction.embeddings import Word2VecReader, GloveReader


class TestEmbeddingsReaders(unittest.TestCase):
    def test_w2v(self):
        reader = Word2VecReader()
        self._test_embeddings_model(reader.read("tests/data/embeddings/w2v.txt"))

    def test_glove(self):
        reader = GloveReader()
        self._test_embeddings_model(reader.read("tests/data/embeddings/glove.txt"))

    def _test_embeddings_model(self, model):
        expected_vocab = {"some", "numbers", "test"}
        expected_size = 3
        expected_w2v = {
            "some": array([0.004003, -0.003830, 0.003021]),
            "numbers": array([-0.131167, -0.208169, 0.087276]),
            "test": array([-0.179721, -0.122177, 0.079874])
        }

        self.assertSetEqual(expected_vocab, model.vocab)
        self.assertEqual(expected_size, model.vector_size)

        for word, expected_vector in expected_w2v.items():
            # numpy vectors can't be asserted equal
            self.assertListEqual(list(expected_vector), list(model.get_vector_for_token(word)))

        self.assertIsNone(model.get_vector_for_token(""))
        self.assertIsNone(model.get_vector_for_token("\n"))
