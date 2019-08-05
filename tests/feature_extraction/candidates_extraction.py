import unittest

from derek.data.model import Document, Entity, Sentence, Paragraph, Relation
from derek.rel_ext.feature_extraction.sampling_strategies import DefaultPairExtractionStrategy,\
    DefaultCandidateExtractionStrategy, DifferentEntitiesCandidateFilter, InSameSentenceCandidateFilter,\
    MaxTokenDistanceCandidateFilter, RelArgTypesCandidateFilter, IntersectingCandidateFilter, AndFilter


class TestCandidatesExtraction(unittest.TestCase):
    def setUp(self) -> None:
        tokens = [
            "I", "will", "do", "my", "homework", "today", ".",
            "It", "is", "very", "hard", "but", "i", "don't", "care", "."
        ]
        sentences = [
            Sentence(0, 7),
            Sentence(7, 16)
        ]
        paragraphs = [Paragraph(0, 2)]
        entities = [
            Entity("_", 0, 1, "t1"), Entity("_", 3, 5, "t2"),
            Entity("_", 7, 8, "t1"), Entity("_", 9, 11, "t2"), Entity("_", 10, 11, "t4")
        ]

        self.doc = Document("_", tokens, sentences, paragraphs, entities)
        self.relations = {Relation(entities[2], entities[3], "t1"), Relation(entities[3], entities[4], "t2")}

    def test_DifferentEntitiesCandidateFilter(self):
        f = DifferentEntitiesCandidateFilter()
        self.assertTrue(f.apply(self.doc, self.doc.entities[0], self.doc.entities[1]))
        self.assertFalse(f.apply(self.doc, self.doc.entities[2], self.doc.entities[2]))

    def test_InSameSentenceCandidateFilter(self):
        f = InSameSentenceCandidateFilter()
        self.assertTrue(f.apply(self.doc, self.doc.entities[0], self.doc.entities[1]))
        self.assertTrue(f.apply(self.doc, self.doc.entities[2], self.doc.entities[3]))
        self.assertFalse(f.apply(self.doc, self.doc.entities[0], self.doc.entities[3]))
        self.assertFalse(f.apply(self.doc, self.doc.entities[2], self.doc.entities[1]))

    def test_MaxTokenDistanceCandidateFilter_intersecting_case(self):
        f = MaxTokenDistanceCandidateFilter(0)
        self.assertFalse(f.apply(self.doc, self.doc.entities[0], self.doc.entities[3]))
        self.assertFalse(f.apply(self.doc, self.doc.entities[2], self.doc.entities[1]))
        self.assertFalse(f.apply(self.doc, self.doc.entities[4], self.doc.entities[2]))
        self.assertTrue(f.apply(self.doc, self.doc.entities[3], self.doc.entities[4]))
        self.assertTrue(f.apply(self.doc, self.doc.entities[4], self.doc.entities[3]))

    def test_MaxTokenDistanceCandidateFilter_normal_case(self):
        f = MaxTokenDistanceCandidateFilter(3)
        self.assertFalse(f.apply(self.doc, self.doc.entities[0], self.doc.entities[3]))
        self.assertFalse(f.apply(self.doc, self.doc.entities[0], self.doc.entities[2]))
        self.assertTrue(f.apply(self.doc, self.doc.entities[1], self.doc.entities[2]))
        self.assertTrue(f.apply(self.doc, self.doc.entities[2], self.doc.entities[3]))

    def test_RelArgTypesCandidateFilter(self):
        valid_types = {("t1", "t1"), ("t2", "t4")}
        f = RelArgTypesCandidateFilter(valid_types)

        self.assertTrue(f.apply(self.doc, self.doc.entities[0], self.doc.entities[0]))
        self.assertTrue(f.apply(self.doc, self.doc.entities[0], self.doc.entities[2]))
        self.assertTrue(f.apply(self.doc, self.doc.entities[2], self.doc.entities[0]))
        self.assertTrue(f.apply(self.doc, self.doc.entities[1], self.doc.entities[4]))
        self.assertTrue(f.apply(self.doc, self.doc.entities[3], self.doc.entities[4]))
        self.assertFalse(f.apply(self.doc, self.doc.entities[4], self.doc.entities[1]))
        self.assertFalse(f.apply(self.doc, self.doc.entities[2], self.doc.entities[3]))
        self.assertFalse(f.apply(self.doc, self.doc.entities[3], self.doc.entities[3]))

    def test_IntersectingCandidateFilter(self):
        f = IntersectingCandidateFilter()
        self.assertTrue(f.apply(self.doc, self.doc.entities[0], self.doc.entities[2]))
        self.assertTrue(f.apply(self.doc, self.doc.entities[2], self.doc.entities[0]))
        self.assertTrue(f.apply(self.doc, self.doc.entities[1], self.doc.entities[4]))
        self.assertFalse(f.apply(self.doc, self.doc.entities[0], self.doc.entities[0]))
        self.assertFalse(f.apply(self.doc, self.doc.entities[3], self.doc.entities[4]))
        self.assertFalse(f.apply(self.doc, self.doc.entities[4], self.doc.entities[3]))

    def test_AndFilter(self):
        filts = [
            DifferentEntitiesCandidateFilter(),
            InSameSentenceCandidateFilter(),
            RelArgTypesCandidateFilter({("t1", "t1"), ("t2", "t4")})
        ]

        f = AndFilter(filts)

        self.assertFalse(f.apply(self.doc, self.doc.entities[0], self.doc.entities[0]))
        self.assertFalse(f.apply(self.doc, self.doc.entities[0], self.doc.entities[2]))

        self.assertFalse(f.apply(self.doc, self.doc.entities[1], self.doc.entities[4]))
        self.assertFalse(f.apply(self.doc, self.doc.entities[2], self.doc.entities[4]))

        self.assertTrue(f.apply(self.doc, self.doc.entities[3], self.doc.entities[4]))

    def test_DefaultPairExtractionStrategy_no_rels(self):
        filts = [
            DifferentEntitiesCandidateFilter(),
            InSameSentenceCandidateFilter(),
            RelArgTypesCandidateFilter({("t1", "t2"), ("t2", "t4")})
        ]

        expected_pairs = [
            (self.doc.entities[0], self.doc.entities[1]),
            (self.doc.entities[2], self.doc.entities[3]),
            (self.doc.entities[3], self.doc.entities[4])
        ]

        strategy = DefaultPairExtractionStrategy(AndFilter(filts))
        actual_pairs = strategy.apply(self.doc, include_labels=False)

        self.assertEqual(actual_pairs, expected_pairs)

    def test_DefaultPairExtractionStrategy_with_rels(self):
        filts = [
            DifferentEntitiesCandidateFilter(),
            InSameSentenceCandidateFilter(),
            RelArgTypesCandidateFilter({("t1", "t2"), ("t2", "t4")})
        ]

        expected_pairs = [
            (self.doc.entities[0], self.doc.entities[1]),
            (self.doc.entities[2], self.doc.entities[3]),
            (self.doc.entities[3], self.doc.entities[4])
        ]

        strategy = DefaultPairExtractionStrategy(AndFilter(filts))
        actual_pairs = strategy.apply(self.doc.with_relations(self.relations), include_labels=True)

        self.assertEqual(actual_pairs, expected_pairs)

    def test_DefaultCandidateExtractionStrategy_no_rels(self):
        filts = [
            DifferentEntitiesCandidateFilter(),
            InSameSentenceCandidateFilter(),
            RelArgTypesCandidateFilter({("t1", "t2"), ("t2", "t4")})
        ]

        expected_candidates = [
            (self.doc.entities[0], self.doc.entities[1], None),
            (self.doc.entities[2], self.doc.entities[3], None),
            (self.doc.entities[3], self.doc.entities[4], None)
        ]

        strategy = DefaultCandidateExtractionStrategy(DefaultPairExtractionStrategy(AndFilter(filts)))
        actual_candidates = strategy.apply(self.doc, include_labels=False)

        self.assertEqual(actual_candidates, expected_candidates)

    def test_DefaultCandidateExtractionStrategy_with_rels(self):
        filts = [
            DifferentEntitiesCandidateFilter(),
            InSameSentenceCandidateFilter(),
            RelArgTypesCandidateFilter({("t1", "t2"), ("t2", "t4")})
        ]

        expected_candidates = [
            (self.doc.entities[0], self.doc.entities[1], None),
            (self.doc.entities[2], self.doc.entities[3], "t1"),
            (self.doc.entities[3], self.doc.entities[4], "t2")
        ]

        strategy = DefaultCandidateExtractionStrategy(DefaultPairExtractionStrategy(AndFilter(filts)))
        actual_candidates = strategy.apply(self.doc.with_relations(self.relations), include_labels=True)

        self.assertEqual(actual_candidates, expected_candidates)
