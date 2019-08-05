import unittest

from derek.coref.feature_extraction.sampling_strategies import CorefPairExtractionStrategy, NounPreprocessingStrategy, \
    CorefCandidateMaker, ClusterPronPairExtractionStrategy, OneGroupSamplingStrategy
from derek.data.model import Sentence, Document, Paragraph, Entity, Relation


def get_samples(doc, max_distance, include_labels):
    return OneGroupSamplingStrategy(
        CorefPairExtractionStrategy(max_distance),
        candidate_maker=CorefCandidateMaker('1')).apply(doc, include_labels=include_labels)


def get_noun_samples(doc, max_distance, include_labels):
    return OneGroupSamplingStrategy(
        CorefPairExtractionStrategy(max_distance, NounPreprocessingStrategy()),
        candidate_maker=CorefCandidateMaker('1')).apply(doc, include_labels=include_labels)


def get_pron_samples(doc, max_distance, include_labels):
    return OneGroupSamplingStrategy(
        ClusterPronPairExtractionStrategy(max_distance),
        candidate_maker=CorefCandidateMaker('1')).apply(doc, include_labels=include_labels)


class TestCorefSamplingStrategy(unittest.TestCase):
    def test_no_entities(self):
        sentences = [
            Sentence(0, 10)
        ]
        paragraphs = [
            Paragraph(0, 1)
        ]
        entities = []
        doc = Document('test', [], sentences, paragraphs, entities)

        max_distance = 3

        actual_samples = get_samples(doc, max_distance, False)
        expected_samples = []
        
        self.assertEqual(actual_samples, expected_samples)

    def test_1_entity(self):
        sentences = [
            Sentence(0, 10)
        ]
        paragraphs = [
            Paragraph(0, 1)
        ]
        entities = [
            Entity('_', 0, 1, '1')
        ]
        doc = Document('test', [], sentences, paragraphs, entities)

        max_distance = 3

        actual_samples = get_samples(doc, max_distance, False)
        expected_samples = []

        self.assertEqual(expected_samples, actual_samples)

    def test_2_entity(self):
        sentences = [
            Sentence(0, 10)
        ]
        paragraphs = [
            Paragraph(0, 1)
        ]
        entities = [
            Entity('_', 0, 1, '1'),
            Entity('_', 1, 2, '1'),
        ]
        doc = Document('test', [], sentences, paragraphs, entities)

        max_distance = 3

        actual_samples = get_samples(doc, max_distance, False)

        expected_samples = [
            (Entity('_', 0, 1, '1'), Entity('_', 1, 2, '1'), None)
        ]

        self.assertEqual(expected_samples, actual_samples)

    def test_2_entity_rel(self):
        sentences = [
            Sentence(0, 10)
        ]
        paragraphs = [
            Paragraph(0, 1)
        ]
        entities = [
            Entity('_', 0, 1, '1'),
            Entity('_', 1, 2, '1'),
        ]
        rels = {
            Relation(Entity('_', 0, 1, '1'), Entity('_', 1, 2, '1'), '1')
        }
        doc = Document('test', [], sentences, paragraphs, entities, rels)

        max_distance = 3

        actual_samples = get_samples(doc, max_distance, True)

        expected_samples = [
            (Entity('_', 0, 1, '1'), Entity('_', 1, 2, '1'), '1')
        ]

        self.assertEqual(expected_samples, actual_samples)

    def test_2_entity_sents(self):
        sentences = [
            Sentence(0, 5),
            Sentence(5, 10),
        ]
        paragraphs = [
            Paragraph(0, 2)
        ]
        entities = [
            Entity('_', 0, 1, '1'),
            Entity('_', 5, 6, '1'),
        ]
        rels = {
            Relation(Entity('_', 0, 1, '1'), Entity('_', 5, 6, '1'), '1')
        }
        doc = Document('test', [], sentences, paragraphs, entities, rels)

        max_distance = 3

        actual_samples = get_samples(doc, max_distance, True)

        expected_samples = [
            (Entity('_', 0, 1, '1'), Entity('_', 5, 6, '1'), '1')
        ]

        self.assertEqual(expected_samples, actual_samples)

    def test_2_entity_long(self):
        sentences = [
            Sentence(0, 3),
            Sentence(3, 5),
            Sentence(5, 10),
        ]
        paragraphs = [
            Paragraph(0, 3)
        ]
        entities = [
            Entity('_', 0, 1, '1'),
            Entity('_', 3, 4, '1'),
            Entity('_', 5, 6, '1'),
        ]
        doc = Document('test', [], sentences, paragraphs, entities)

        max_distance = 1

        actual_samples = get_samples(doc, max_distance, False)

        expected_samples = [
            (Entity('_', 0, 1, '1'), Entity('_', 3, 4, '1'), None),
            (Entity('_', 3, 4, '1'), Entity('_', 5, 6, '1'), None)
        ]

        self.assertEqual(expected_samples, actual_samples)

    def test_3_entity_paragraphs(self):
        sentences = [
            Sentence(0, 5),
            Sentence(5, 10),
        ]
        paragraphs = [
            Paragraph(0, 1),
            Paragraph(1, 2),
        ]
        entities = [
            Entity('_', 0, 1, '1'),
            Entity('_', 1, 2, '1'),
            Entity('_', 5, 6, '2'),
        ]
        doc = Document('test', [], sentences, paragraphs, entities)

        max_distance = 3

        actual_samples = get_samples(doc, max_distance, False)

        expected_samples = [
            (Entity('_', 0, 1, '1'), Entity('_', 1, 2, '1'), None),
            (Entity('_', 0, 1, '1'), Entity('_', 5, 6, '2'), None),
            (Entity('_', 1, 2, '1'), Entity('_', 5, 6, '2'), None),
        ]

        self.assertEqual(expected_samples, actual_samples)


class TestNounCorefSamplingStrategy(unittest.TestCase):

    def test_no_entities(self):
        sentences = [
            Sentence(0, 10)
        ]
        paragraphs = [
            Paragraph(0, 1)
        ]
        entities = []
        doc = Document('test', [], sentences, paragraphs, entities)

        max_distance = 3

        actual_samples = get_noun_samples(doc, max_distance, False)
        expected_samples = []

        self.assertEqual(actual_samples, expected_samples)

    def test_1_entity(self):
        sentences = [
            Sentence(0, 10)
        ]
        paragraphs = [
            Paragraph(0, 1)
        ]
        entities = [
            Entity('_', 0, 1, 'noun')
        ]
        doc = Document('test', [], sentences, paragraphs, entities)

        max_distance = 3

        actual_samples = get_noun_samples(doc, max_distance, False)
        expected_samples = []

        self.assertEqual(expected_samples, actual_samples)

    def test_2_entity(self):
        sentences = [
            Sentence(0, 10)
        ]
        paragraphs = [
            Paragraph(0, 1)
        ]
        entities = [
            Entity('_', 0, 1, 'noun'),
            Entity('_', 1, 2, 'noun'),
        ]
        doc = Document('test', [], sentences, paragraphs, entities)

        max_distance = 3

        actual_samples = get_noun_samples(doc, max_distance, False)

        expected_samples = [
            (Entity('_', 0, 1, 'noun'), Entity('_', 1, 2, 'noun'), None)
        ]

        self.assertEqual(expected_samples, actual_samples)

    def test_2_entity_pron(self):
        sentences = [
            Sentence(0, 10)
        ]
        paragraphs = [
            Paragraph(0, 1)
        ]
        entities = [
            Entity('_', 0, 1, 'noun'),
            Entity('_', 1, 2, 'noun'),
            Entity('_', 2, 3, 'pron'),
        ]
        doc = Document('test', [], sentences, paragraphs, entities)

        max_distance = 3

        actual_samples = get_noun_samples(doc, max_distance, False)

        expected_samples = [
            (Entity('_', 0, 1, 'noun'), Entity('_', 1, 2, 'noun'), None)
        ]

        self.assertEqual(expected_samples, actual_samples)

    def test_2_entity_2_pron(self):
        sentences = [
            Sentence(0, 10)
        ]
        paragraphs = [
            Paragraph(0, 1)
        ]
        entities = [
            Entity('_', 0, 1, 'noun'),
            Entity('_', 1, 2, 'pron'),
            Entity('_', 2, 3, 'noun'),
            Entity('_', 3, 4, 'pron'),
        ]
        doc = Document('test', [], sentences, paragraphs, entities)

        max_distance = 3

        actual_samples = get_noun_samples(doc, max_distance, False)

        expected_samples = [
            (Entity('_', 0, 1, 'noun'), Entity('_', 2, 3, 'noun'), None)
        ]

        self.assertEqual(expected_samples, actual_samples)

    def test_2_entity_2_pron_rels(self):
        sentences = [
            Sentence(0, 10)
        ]
        paragraphs = [
            Paragraph(0, 1)
        ]
        entities = [
            Entity('_', 0, 1, 'noun'),
            Entity('_', 1, 2, 'pron'),
            Entity('_', 2, 3, 'noun'),
            Entity('_', 3, 4, 'pron'),
        ]
        rels = {
            Relation(Entity('_', 0, 1, 'noun'), Entity('_', 1, 2, 'pron'), '1'),
            Relation(Entity('_', 0, 1, 'noun'), Entity('_', 2, 3, 'noun'), '1'),
        }
        doc = Document('test', [], sentences, paragraphs, entities, rels)

        max_distance = 3

        actual_samples = get_noun_samples(doc, max_distance, True)

        expected_samples = [
            (Entity('_', 0, 1, 'noun'), Entity('_', 2, 3, 'noun'), '1')
        ]

        self.assertEqual(expected_samples, actual_samples)


class TestPronounCorefSamplingStrategy(unittest.TestCase):
    def test_no_entities(self):
        sentences = [
            Sentence(0, 10)
        ]
        paragraphs = [
            Paragraph(0, 1)
        ]
        entities = []
        rels = set()
        doc = Document('test', [], sentences, paragraphs, entities, rels)

        max_distance = 3

        actual_samples = get_pron_samples(doc, max_distance, False)
        expected_samples = []

        self.assertEqual(actual_samples, expected_samples)

    def test_nouns(self):
        sentences = [
            Sentence(0, 10)
        ]
        paragraphs = [
            Paragraph(0, 1)
        ]
        entities = [
            Entity('_', 0, 1, 'noun'),
            Entity('_', 1, 2, 'noun'),
            Entity('_', 2, 3, 'noun'),
            Entity('_', 3, 4, 'noun'),
        ]
        rels = {
            Relation(Entity('_', 0, 1, 'noun'), Entity('_', 1, 2, 'noun'), '1'),
            Relation(Entity('_', 0, 1, 'noun'), Entity('_', 2, 3, 'noun'), '1'),
            Relation(Entity('_', 2, 3, 'noun'), Entity('_', 3, 4, 'noun'), '1'),
        }
        doc = Document('test', [], sentences, paragraphs, entities, rels)

        max_distance = 3

        actual_samples = get_pron_samples(doc, max_distance, False)
        expected_samples = []

        self.assertEqual(actual_samples, expected_samples)

    def test_pron(self):
        sentences = [
            Sentence(0, 10)
        ]
        paragraphs = [
            Paragraph(0, 1)
        ]
        entities = [
            Entity('_', 0, 1, 'noun'),
            Entity('_', 1, 2, 'pron'),
            Entity('_', 3, 4, 'noun'),
            Entity('_', 5, 6, 'noun'),
        ]
        rels = {
            Relation(Entity('_', 0, 1, 'noun'), Entity('_', 1, 2, 'pron'), '1'),
            Relation(Entity('_', 0, 1, 'noun'), Entity('_', 3, 4, 'noun'), '1'),
            Relation(Entity('_', 3, 4, 'noun'), Entity('_', 5, 6, 'noun'), '1'),
        }
        doc = Document('test', [], sentences, paragraphs, entities, rels)

        max_distance = 3

        actual_samples = get_pron_samples(doc, max_distance, True)
        expected_samples = [
            (Entity('_', 0, 1, 'noun'), Entity('_', 1, 2, 'pron'), '1')
        ]

        self.assertEqual(actual_samples, expected_samples)

    def test_2_pron(self):
        sentences = [
            Sentence(0, 10)
        ]
        paragraphs = [
            Paragraph(0, 1)
        ]
        entities = [
            Entity('_', 0, 1, 'noun'),
            Entity('_', 1, 2, 'pron'),
            Entity('_', 2, 3, 'pron'),
            Entity('_', 3, 4, 'noun'),
            Entity('_', 5, 6, 'noun'),
        ]
        rels = {
            Relation(Entity('_', 0, 1, 'noun'), Entity('_', 1, 2, 'pron'), '1'),
            Relation(Entity('_', 0, 1, 'noun'), Entity('_', 3, 4, 'noun'), '1'),
            Relation(Entity('_', 2, 3, 'pron'), Entity('_', 0, 1, 'noun'), '1'),
            Relation(Entity('_', 3, 4, 'noun'), Entity('_', 5, 6, 'noun'), '1'),
        }
        doc = Document('test', [], sentences, paragraphs, entities, rels)

        max_distance = 3

        actual_samples = get_pron_samples(doc, max_distance, True)
        expected_samples = [
            (Entity('_', 0, 1, 'noun'), Entity('_', 1, 2, 'pron'), '1'),
            (Entity('_', 0, 1, 'noun'), Entity('_', 2, 3, 'pron'), '1'),
        ]

        self.assertEqual(actual_samples, expected_samples)

    def test_2_chains_2_pron(self):
        sentences = [
            Sentence(0, 10)
        ]
        paragraphs = [
            Paragraph(0, 1)
        ]
        entities = [
            Entity('_', 0, 1, 'noun'),
            Entity('_', 1, 2, 'pron'),
            Entity('_', 2, 3, 'pron'),
            Entity('_', 3, 4, 'noun'),
            Entity('_', 5, 6, 'noun'),
        ]
        rels = {
            Relation(Entity('_', 0, 1, 'noun'), Entity('_', 2, 3, 'pron'), '1'),
            Relation(Entity('_', 1, 2, 'pron'), Entity('_', 3, 4, 'noun'), '1'),
            Relation(Entity('_', 3, 4, 'noun'), Entity('_', 5, 6, 'noun'), '1'),
        }
        doc = Document('test', [], sentences, paragraphs, entities, rels)

        max_distance = 3

        actual_samples = get_pron_samples(doc, max_distance, True)
        expected_samples = [
            (Entity('_', 0, 1, 'noun'), Entity('_', 1, 2, 'pron'), None),
            (Entity('_', 1, 2, 'pron'), Entity('_', 3, 4, 'noun'), '1'),
            (Entity('_', 0, 1, 'noun'), Entity('_', 2, 3, 'pron'), '1'),
            (Entity('_', 2, 3, 'pron'), Entity('_', 3, 4, 'noun'), None),
        ]
        self.assertEqual(actual_samples, expected_samples)
