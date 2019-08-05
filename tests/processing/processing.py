import unittest
from derek.data.model import Sentence
from derek.data.processing_helper import fix_joined_tokens, fix_raw_tokens_after_elimination, \
    eliminate_references_and_figures


class FixJoinedTokensTest(unittest.TestCase):
    def test_normal_tokens(self):
        tokens = ["Bacteria", "lives", "in", "Habitat", "-"]
        sentences = [Sentence(0, 5)]
        raw_tokens = [(0, 8), (9, 14), (15, 17), (18, 25), (26, 27)]
        expected_tokens = tokens
        expected_sentences = sentences
        expected_raw_tokens = raw_tokens

        self.assertEqual(
            fix_joined_tokens(tokens, sentences, raw_tokens, {"/", "-"}),
            (expected_tokens, expected_sentences, expected_raw_tokens))

    def test_joined_tokens_one_sent(self):
        tokens = ["/Bacteria", "lives/sleeps/works", "-in-", "---", "habitat/-/", "/", "."]
        sentences = [Sentence(0, 7)]
        raw_tokens = [(0, 9), (10, 28), (29, 33), (34, 37), (38, 48), (49, 50), (50, 51)]
        expected_tokens = [
                "/", "Bacteria", "lives", "/", "sleeps", "/", "works", "-", "in", "-",
                "-", "-", "-", "habitat", "/", "-", "/", "/", "."
        ]
        expected_sentences = [Sentence(0, 19)]
        expected_raw_tokens = [
            (0, 1), (1, 9),
            (10, 15), (15, 16), (16, 22), (22, 23), (23, 28),
            (29, 30), (30, 32), (32, 33),
            (34, 35), (35, 36), (36, 37),
            (38, 45), (45, 46), (46, 47), (47, 48),
            (49, 50),
            (50, 51)
        ]

        self.assertEqual(
            fix_joined_tokens(tokens, sentences, raw_tokens, {"/", "-"}),
            (expected_tokens, expected_sentences, expected_raw_tokens))

    def test_joined_tokens_two_sent(self):
        tokens = ["/Bacteria", "lives/sleeps/works", ".", "-In-", "---", "habitat/-/", "/", "."]
        sentences = [Sentence(0, 3), Sentence(3, 8)]
        raw_tokens = [(0, 9), (10, 28), (28, 29), (30, 34), (35, 38), (39, 49), (50, 51), (51, 52)]
        expected_tokens = [
            "/", "Bacteria", "lives", "/", "sleeps", "/", "works", ".",
            "-", "In", "-", "-", "-", "-", "habitat", "/", "-", "/", "/", "."
        ]
        expected_sentences = [Sentence(0, 8), Sentence(8, 20)]
        expected_raw_tokens = [
            (0, 1), (1, 9),
            (10, 15), (15, 16), (16, 22), (22, 23), (23, 28),
            (28, 29),
            (30, 31), (31, 33), (33, 34),
            (35, 36), (36, 37), (37, 38),
            (39, 46), (46, 47), (47, 48), (48, 49),
            (50, 51),
            (51, 52)
        ]

        self.assertEqual(
            fix_joined_tokens(tokens, sentences, raw_tokens, {"/", "-"}),
            (expected_tokens, expected_sentences, expected_raw_tokens))


class FixRawTokensAfterEliminationTest(unittest.TestCase):
    def test_no_matches(self):
        matches = []
        raw_tokens = [(0, 3), (4, 6), (7, 10)]

        self.assertEqual(fix_raw_tokens_after_elimination(raw_tokens, matches), raw_tokens)

    def test_separated_matches(self):
        """
            original string: aaa (aa) aaaaa (aaa.1) aaa aaaa (a) 1 aaaaa
            after elimination: aaa  aaaaa  aaa aaaa  1 aaaaa
        """
        matches = [(4, 8), (15, 22), (32, 35)]
        raw_tokens = [(0, 3), (5, 10), (12, 15), (16, 20), (22, 23), (24, 30)]
        expected_tokens = [(0, 3), (9, 14), (23, 26), (27, 31), (36, 37), (38, 44)]

        self.assertEqual(fix_raw_tokens_after_elimination(raw_tokens, matches), expected_tokens)

    def test_near_matches(self):
        """
                    original string: aaa (aa) (aaa) (aaa.1) aaa aaaa (a) 1 aaaaa
                    after elimination: aaa   aaa aaaa  1 aaaaa
        """
        matches = [(4, 8), (9, 14), (15, 22), (32, 35)]
        raw_tokens = [(0, 3), (7, 10), (11, 15), (17, 18), (19, 25)]
        expected_tokens = [(0, 3), (23, 26), (27, 31), (36, 37), (38, 44)]

        self.assertEqual(fix_raw_tokens_after_elimination(raw_tokens, matches), expected_tokens)

    def test_start_end_matches(self):
        """
                    original string: (a) aaa (a) aaa aaaa (a).(a)
                    after elimination: aaa  aaa asaa .
        """
        matches = [(0, 3), (8, 11), (21, 24), (25, 28)]
        raw_tokens = [(1, 4), (6, 9), (10, 14), (15, 16)]
        expected_tokens = [(4, 7), (12, 15), (16, 20), (24, 25)]

        self.assertEqual(fix_raw_tokens_after_elimination(raw_tokens, matches), expected_tokens)


class EliminateReferencesAndFiguresTest(unittest.TestCase):
    def test_no_references(self):
        text = "Pelecypod-associated bacteria in habitat."
        expected_text = text
        expected_matches = []
        actual_text, actual_matches = eliminate_references_and_figures(text)

        self.assertEqual(actual_text, expected_text)
        self.assertEqual(actual_matches, expected_matches)

    def test_references(self):
        text = "Pelecypod-associated (yum et.al 2004) bacteria(greenwood et al,2003) " \
               "in habitat(lee et.al. 2007).(wang et al. 80)"
        expected_text = "Pelecypod-associated  bacteria in habitat."
        expected_matches = [(21, 37), (46, 68), (79, 96), (97, 113)]
        actual_text, actual_matches = eliminate_references_and_figures(text)

        self.assertEqual(actual_text, expected_text)
        self.assertEqual(actual_matches, expected_matches)

    def test_figure_tables(self):
        text = "Pelecypod-associated (Fig. 1) bacteria(vegetables 3) in habitat(Table 3)(64%).(Figure 3)"
        expected_text = "Pelecypod-associated  bacteria(vegetables 3) in habitat."
        expected_matches = [(21, 29), (63, 72), (72, 77), (78, 88)]
        actual_text, actual_matches = eliminate_references_and_figures(text)

        self.assertEqual(actual_text, expected_text)
        self.assertEqual(actual_matches, expected_matches)

