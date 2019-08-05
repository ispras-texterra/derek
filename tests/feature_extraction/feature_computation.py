import unittest

from derek.common.feature_computation.computers import get_dt_breakups_feature, get_dt_deltas_feature, \
    get_sentence_borders_feature, get_dt_depths_feature
from derek.data.model import Document, Sentence, Paragraph
from derek.common.feature_extraction.helper import Direction


class TestBreakupsExtractionClass(unittest.TestCase):
    def setUp(self):
        sent_1_tokens = ["Human", "and", "tick", "spotted", "fever", "group", "Rickettsia", "isolates",
                         "from", "Israel", ":", "a", "genotypic", "analysis", "."]
        sent_1_head_distances = [3, -1, -2, 0, 2, 1, 1, 7, -1, -1, 4, 2, 1, 1, -11]

        self.doc_with_1_sent = Document("", sent_1_tokens, [Sentence(0, len(sent_1_tokens))], [Paragraph(0, 1)],
                                        token_features={"dt_head_distances": sent_1_head_distances})

        sent_2_tokens = ["The", "precise", "mechanisms", "that", "initiate", "bacterial", "uptake",
                         "have", "not", "yet", "been", "elucidated", "."]
        sent_2_head_distances = [2, 1, 9, 1, -2, 1, -2, 4, 3, 2, 1, 0, -1]

        self.doc_with_2_sent = Document("", sent_1_tokens + sent_2_tokens,
                                        [Sentence(0, len(sent_1_tokens)),
                                         Sentence(len(sent_1_tokens), len(sent_1_tokens) + len(sent_2_tokens))],
                                        [Paragraph(0, 2)],
                                        token_features={"dt_head_distances": sent_1_head_distances + sent_2_head_distances})

    def test_breakups_extraction_1(self):
        direction = Direction.FORWARD
        breakups = [True, False, False, False, True, True, True, True, False, False, True, True, True, True, False]

        self.assertEqual(get_dt_breakups_feature(self.doc_with_1_sent, direction), breakups)

    def test_breakups_extraction_2(self):
        direction = Direction.BACKWARD
        breakups = [False, True, True, False, False, False, False, False, True, True, False, False, False, False, True]

        self.assertEqual(get_dt_breakups_feature(self.doc_with_1_sent, direction), breakups)

    def test_breakups_extraction_3(self):
        direction = Direction.FORWARD
        sent_1_breakups = [True, False, False, False, True, True, True, True,
                           False, False, True, True, True, True, False]
        sent_2_breakups = [True, True, True, True, False, True, False, True, True, True, True, False, False]
        breakups = sent_1_breakups + sent_2_breakups

        self.assertEqual(get_dt_breakups_feature(self.doc_with_2_sent, direction), breakups)

    def test_breakups_extraction_4(self):
        direction = Direction.BACKWARD
        sent_1_breakups = [False, True, True, False, False, False, False, False,
                           True, True, False, False, False, False, True]

        sent_2_breakups = [False, False, False, False, True, False, True, False, False, False, False, False, True]
        breakups = sent_1_breakups + sent_2_breakups

        self.assertEqual(get_dt_breakups_feature(self.doc_with_2_sent, direction), breakups)

    def test_borders_extraction_1(self):
        borders = ["start", "in", "in", "in", "in", "in", "in", "in", "in", "in", "in", "in", "in", "in", "end"]

        self.assertEqual(get_sentence_borders_feature(self.doc_with_1_sent), borders)

    def test_borders_extraction_2(self):
        sent_1_borders = ["start", "in", "in", "in", "in", "in", "in", "in", "in", "in", "in", "in", "in", "in", "end"]
        sent_2_borders = ["start", "in", "in", "in", "in", "in", "in", "in", "in", "in", "in", "in", "end"]
        borders = sent_1_borders + sent_2_borders

        self.assertEqual(get_sentence_borders_feature(self.doc_with_2_sent), borders)

    def test_borders_extraction_3(self):
        tokens = ["bacteria", "spotted", ".", ".", "it's", "."]
        sentences = [Sentence(0, 3), Sentence(3, 4), Sentence(4, 6)]
        broken_doc = Document("", tokens, sentences, [Paragraph(0, 2)])

        borders = ["start", "in", "end", "start", "start", "end"]
        self.assertEqual(get_sentence_borders_feature(broken_doc), borders)

    def test_depths_extraction_1(self):
        depths = [1, 2, 2, 0, 4, 4, 3, 2, 3, 4, 2, 3, 3, 2, 1]

        self.assertEqual(get_dt_depths_feature(self.doc_with_1_sent), depths)

    def test_depths_extraction_2(self):
        sent_1_depths = [1, 2, 2, 0, 4, 4, 3, 2, 3, 4, 2, 3, 3, 2, 1]
        sent_2_depths = [2, 2, 1, 3, 2, 4, 3, 1, 1, 1, 1, 0, 1]
        depths = sent_1_depths + sent_2_depths

        self.assertEqual(get_dt_depths_feature(self.doc_with_2_sent), depths)

    def test_deltas_extraction_1(self):
        direction = Direction.FORWARD
        deltas = ["$START$", 1, 0, -2, 4, 0, -1, -1, 1, 1, -2, 1, 0, -1, -1]

        self.assertEqual(get_dt_deltas_feature(self.doc_with_1_sent, direction), deltas)

    def test_deltas_extraction_2(self):
        direction = Direction.BACKWARD
        deltas = [-1, 0, 2, -4, 0, 1, 1, -1, -1, 2, -1, 0, 1, 1, "$START$"]

        self.assertEqual(get_dt_deltas_feature(self.doc_with_1_sent, direction), deltas)

    def test_deltas_extraction_3(self):
        direction = Direction.FORWARD
        sent_1_deltas = ["$START$", 1, 0, -2, 4, 0, -1, -1, 1, 1, -2, 1, 0, -1, -1]
        sent_2_deltas = ["$START$", 0, -1, 2, -1, 2, -1, -2, 0, 0, 0, -1, 1]
        deltas = sent_1_deltas + sent_2_deltas

        self.assertEqual(get_dt_deltas_feature(self.doc_with_2_sent, direction), deltas)

    def test_deltas_extraction_4(self):
        direction = Direction.BACKWARD
        sent_1_deltas = [-1, 0, 2, -4, 0, 1, 1, -1, -1, 2, -1, 0, 1, 1, "$START$"]
        sent_2_deltas = [0, 1, -2, 1, -2, 1, 2, 0, 0, 0, 1, -1, "$START$"]
        deltas = sent_1_deltas + sent_2_deltas

        self.assertEqual(get_dt_deltas_feature(self.doc_with_2_sent, direction), deltas)


if __name__ == "__main__":
    unittest.main()
