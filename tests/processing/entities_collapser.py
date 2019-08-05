import unittest
import numpy as np

from derek.data.model import Sentence, Entity, Paragraph, Document, Relation, SortedSpansSet
from derek.data.entities_collapser import EntitiesCollapser


class TestEntitiesCollapser(unittest.TestCase):
    def setUp(self):
        tokens = [
            "Recurrence", "of", "Pelecypod-associated", "cholera", "in", "Sardinia", ".",
            
            "From", "Oct.", "30", "to", "Nov.", "7", ",", "1979", ",", "10", "people", "in", "the", "Sardinian",
            "province", "of", "Cagliari", "had", "onset", "of", "bacteriologically", "confirmed", "cholera", "."
        ]
        sentences = [Sentence(0, 7), Sentence(7, 31)]
        
        entities = [
            Entity("T1", 2, 3, "Habitat"), Entity("T2", 2, 4, "Bacteria"), Entity("T3", 3, 4, "Bacteria"),
            Entity("T4", 5, 6, "Geographical"),

            Entity("T5", 17, 18, "Habitat"), Entity("T6", 17, 24, "Habitat"),
            Entity("T7", 20, 22, "Geographical"), Entity("T8", 23, 24, "Geographical"), Entity("T9", 29, 30, "Bacteria")
        ]

        paragraphs = [Paragraph(0, 1), Paragraph(1, 2)]
        relations = [Relation(entities[0], entities[1], "Lives_in"), Relation(entities[8], entities[6], "Lives_in")]

        self.doc = Document("_", tokens, sentences, paragraphs, entities, relations)

    def test_inner_entities_collapse(self):
        expected_tokens = [
            "Recurrence", "of", "Pelecypod-associated", "cholera", "in", "$Geographical$", ".",

            "From", "Oct.", "30", "to", "Nov.", "7", ",", "1979", ",", "10", "people", "in", "the", "$Geographical$",
            "of", "$Geographical$", "had", "onset", "of", "bacteriologically", "confirmed", "cholera", "."
        ]
        expected_sentences = [Sentence(0, 7), Sentence(7, 30)]

        expected_entities = [
            Entity("T1", 2, 3, "Habitat"), Entity("T2", 2, 4, "Bacteria"), Entity("T3", 3, 4, "Bacteria"),
            Entity("T4", 5, 6, "Geographical"),

            Entity("T5", 17, 18, "Habitat"), Entity("T6", 17, 23, "Habitat"),
            Entity("T7", 20, 21, "Geographical"), Entity("T8", 22, 23, "Geographical"), Entity("T9", 28, 29, "Bacteria")
        ]

        expected_paragraphs = [Paragraph(0, 1), Paragraph(1, 2)]
        expected_relations = [
            Relation(expected_entities[0], expected_entities[1], "Lives_in"),
            Relation(expected_entities[8], expected_entities[6], "Lives_in")
        ]

        expected_doc = Document(
            "_", expected_tokens, expected_sentences, expected_paragraphs, expected_entities, expected_relations)

        actual_doc = EntitiesCollapser({"Geographical"}).transform(self.doc)
        self.assertEqual(expected_doc, actual_doc)

    def test_entities_with_nesting_collapse(self):
        expected_tokens = [
            "Recurrence", "of", "$Bacteria$", "in", "Sardinia", ".",

            "From", "Oct.", "30", "to", "Nov.", "7", ",", "1979", ",", "10", "people", "in", "the", "Sardinian",
            "province", "of", "Cagliari", "had", "onset", "of", "bacteriologically", "confirmed", "$Bacteria$", "."
        ]
        expected_sentences = [Sentence(0, 6), Sentence(6, 30)]

        expected_entities = [
            Entity("T1", 2, 3, "Habitat"), Entity("T2", 2, 3, "Bacteria"), Entity("T3", 2, 3, "Bacteria"),
            Entity("T4", 4, 5, "Geographical"),

            Entity("T5", 16, 17, "Habitat"), Entity("T6", 16, 23, "Habitat"),
            Entity("T7", 19, 21, "Geographical"), Entity("T8", 22, 23, "Geographical"), Entity("T9", 28, 29, "Bacteria")
        ]

        expected_paragraphs = [Paragraph(0, 1), Paragraph(1, 2)]
        expected_relations = [
            Relation(expected_entities[0], expected_entities[1], "Lives_in"),
            Relation(expected_entities[8], expected_entities[6], "Lives_in")
        ]

        expected_doc = Document(
            "_", expected_tokens, expected_sentences, expected_paragraphs, expected_entities, expected_relations)

        actual_doc = EntitiesCollapser({"Bacteria"}).transform(self.doc)
        self.assertEqual(expected_doc, actual_doc)

    def test_different_entities_types_collapse(self):
        expected_tokens = [
            "Recurrence", "of", "$Bacteria$", "in", "$Geographical$", ".",

            "From", "Oct.", "30", "to", "Nov.", "7", ",", "1979", ",", "10", "$Habitat$",
            "had", "onset", "of", "bacteriologically", "confirmed", "$Bacteria$", "."
        ]
        expected_sentences = [Sentence(0, 6), Sentence(6, 24)]

        expected_entities = [
            Entity("T1", 2, 3, "Habitat"), Entity("T2", 2, 3, "Bacteria"), Entity("T3", 2, 3, "Bacteria"),
            Entity("T4", 4, 5, "Geographical"),

            Entity("T5", 16, 17, "Habitat"), Entity("T6", 16, 17, "Habitat"),
            Entity("T7", 16, 17, "Geographical"), Entity("T8", 16, 17, "Geographical"), Entity("T9", 22, 23, "Bacteria")
        ]

        expected_paragraphs = [Paragraph(0, 1), Paragraph(1, 2)]
        expected_relations = [
            Relation(expected_entities[0], expected_entities[1], "Lives_in"),
            Relation(expected_entities[8], expected_entities[6], "Lives_in")
        ]

        expected_doc = Document(
            "_", expected_tokens, expected_sentences, expected_paragraphs, expected_entities, expected_relations)

        actual_doc = EntitiesCollapser({"Habitat", "Bacteria", "Geographical"}).transform(self.doc)
        self.assertEqual(expected_doc, actual_doc)

    def test_dt_features_collapse(self):
        dt_head_distances = [
            6, -1, 1, -2, -1, -1, 0,

            2, -1, 0, -1, 1, -2, 11, 10, 9, 1, 7, -1, 2, 1,
            -3, -1, -1, -12, -1, -1, 1, 1, -3, -6
        ]

        dt_labels = [
            "nsubj", "prep", "nn", "pobj", "prep", "pobj", "ROOT",

            "prep", "pobj", "ROOT", "dep", "nn", "dep", "punct", "dep", "punct", "amod", "nsubj", "prep", "det", "amod",
            "pobj", "prep", "pobj", "null", "advmod", "dep", "advmod", "amod", "pobj", "punct"
        ]

        expected_dt_head_distances = [
            6, -1, 0, -1, 0, 0,

            2, -1, 0, -1, 1, -2, 11, 10, 9, 1, 0, -12, -1, -1, 1, 1, 0, -6
        ]
        expected_dt_labels = [
            "nsubj", "prep", "$Bacteria$", "prep", "$Geographical$", "ROOT",

            "prep", "pobj", "ROOT", "dep", "nn", "dep", "punct", "dep", "punct", "amod", "$Habitat$",
            "null", "advmod", "dep", "advmod", "amod", "$Bacteria$", "punct"
        ]

        expected_tf = {"dt_labels": expected_dt_labels, "dt_head_distances": expected_dt_head_distances}
        input_doc = self.doc.with_additional_token_features(
            {"dt_head_distances": dt_head_distances, "dt_labels": dt_labels})
        actual_doc = EntitiesCollapser({"Habitat", "Bacteria", "Geographical"}).transform(input_doc)
        actual_tf = actual_doc.token_features
        self.assertDictEqual(actual_tf, expected_tf)

    def test_pos_features_collapse(self):
        pos = [
            "NNP", "IN", "NNP", "NN", "IN", "NNP", "DOT",

            "IN", "NNP", "OTHER", "OTHER", "NNP", "OTHER", "COMMA", "OTHER", "COMMA", "OTHER", "NNS", "IN", "DT", "JJ",
            "NN", "IN", "NNP", "VBD", "RB", "IN", "RB", "VBN", "NN", "DOT"
        ]

        expected_pos = [
            "NNP", "IN", "$Bacteria$", "IN", "$Geographical$", "DOT",

            "IN", "NNP", "OTHER", "OTHER", "NNP", "OTHER", "COMMA", "OTHER", "COMMA", "OTHER", "$Habitat$",
            "VBD", "RB", "IN", "RB", "VBN", "$Bacteria$", "DOT",
        ]

        expected_tf = {"pos": expected_pos}
        input_doc = self.doc.with_additional_token_features({"pos": pos})
        actual_doc = EntitiesCollapser({"Habitat", "Bacteria", "Geographical"}).transform(input_doc)
        actual_tf = actual_doc.token_features
        self.assertDictEqual(actual_tf, expected_tf)

    def test_ne_extras_collapse(self):
        nes = SortedSpansSet([
            Entity("_", 0, 1, "left"), Entity("_", 2, 4, "same"), Entity("_", 3, 4, "include"),
            Entity("_", 5, 6, "same"), Entity("_", 15, 19, "intersect"), Entity("_", 17, 20, "include"),
            Entity("_", 22, 25, "intersect")])

        expected_nes = SortedSpansSet([
                Entity("_", 0, 1, "left"), Entity("_", 2, 3, "same"), Entity("_", 2, 3, "include"),
                Entity("_", 4, 5, "same"), Entity("_", 14, 17, "intersect"), Entity("_", 16, 17, "include"),
                Entity("_", 16, 18, "intersect")])

        input_doc = self.doc.with_additional_extras({"ne": nes})
        actual_doc = EntitiesCollapser({"Habitat", "Bacteria", "Geographical"}).transform(input_doc)
        actual_extras = actual_doc.extras
        self.assertDictEqual(actual_extras, {"ne": expected_nes})

    def test_feats_features_collapse(self):
        feats = [
            {"test_feat": "true"}, {}, {"yet": "1"}, {"another": "false"}, {}, {"test": "3"}, {"test": 4},

            {}, {"yet": "3"}, {}, {}, {}, {"another": "true"}, {}, {}, {}, {"test": "4"}, {}, {}, {}, {},
            {}, {}, {}, {}, {}, {}, {}, {}, {"bacteria": "true"}, {"bacteria": "false"}
        ]

        expected_feats = [
            {"test_feat": "true"}, {}, {}, {}, {}, {"test": 4},

            {}, {"yet": "3"}, {}, {}, {}, {"another": "true"}, {}, {}, {}, {"test": "4"}, {},
            {}, {}, {}, {}, {}, {}, {"bacteria": "false"}
        ]

        expected_tf = {"feats": expected_feats}
        input_doc = self.doc.with_additional_token_features({"feats": feats})
        actual_doc = EntitiesCollapser({"Habitat", "Bacteria", "Geographical"}).transform(input_doc)
        actual_tf = actual_doc.token_features
        self.assertDictEqual(actual_tf, expected_tf)

    def test_vectors_features_collapse(self):
        vectors = [np.array([1, 2])] * 7 + [np.array([3, 4])] * 24

        expected_vectors = \
            [np.array([1, 2])] * 2 + [np.array([0, 0]), np.array([1, 2]), np.array([0, 0]), np.array([1, 2])] + \
            [np.array([3, 4])] * 10 + [np.array([0, 0])] + [np.array([3, 4])] * 5 + [np.array([0, 0]), np.array([3, 4])]

        expected_tf = {"vectors": expected_vectors}
        input_doc = self.doc.with_additional_token_features({"vectors": vectors})
        actual_doc = EntitiesCollapser({"Habitat", "Bacteria", "Geographical"}).transform(input_doc)
        actual_tf = actual_doc.token_features
        self.assertSetEqual(set(actual_tf.keys()), set(expected_tf.keys()))
        self.assertSequenceEqual([x.tolist() for x in actual_tf["vectors"]],
                                 [x.tolist() for x in expected_tf["vectors"]])

    def test_collapsing_with_ne(self):
        input_doc = self.doc.with_additional_extras({"ne": self.doc.entities})
        input_doc = input_doc.without_relations().without_entities()

        entities = SortedSpansSet([
            Entity("_", 0, 1, "left"), Entity("_", 2, 4, "same"), Entity("_", 3, 4, "include"),
            Entity("_", 5, 6, "same"), Entity("_", 15, 19, "intersect"), Entity("_", 17, 20, "include"),
            Entity("_", 22, 25, "intersect")])

        input_doc = input_doc.with_entities(entities)

        expected_tokens = [
            "Recurrence", "of", "$Bacteria$", "in", "$Geographical$", ".",

            "From", "Oct.", "30", "to", "Nov.", "7", ",", "1979", ",", "10", "$Habitat$",
            "had", "onset", "of", "bacteriologically", "confirmed", "$Bacteria$", "."
        ]
        expected_sentences = [Sentence(0, 6), Sentence(6, 24)]
        expected_paragraphs = [Paragraph(0, 1), Paragraph(1, 2)]

        expected_nes = SortedSpansSet([
            Entity("T1", 2, 3, "Habitat"), Entity("T2", 2, 3, "Bacteria"), Entity("T3", 2, 3, "Bacteria"),
            Entity("T4", 4, 5, "Geographical"),

            Entity("T5", 16, 17, "Habitat"), Entity("T6", 16, 17, "Habitat"),
            Entity("T7", 16, 17, "Geographical"), Entity("T8", 16, 17, "Geographical"), Entity("T9", 22, 23, "Bacteria")
        ])

        expected_entities = SortedSpansSet([
            Entity("_", 0, 1, "left"), Entity("_", 2, 3, "same"), Entity("_", 2, 3, "include"),
            Entity("_", 4, 5, "same"), Entity("_", 14, 17, "intersect"), Entity("_", 16, 17, "include"),
            Entity("_", 16, 18, "intersect")])

        expected_doc = Document(
            "_", expected_tokens, expected_sentences, expected_paragraphs, expected_entities,
            extras={"ne": expected_nes})

        actual_doc = EntitiesCollapser({"Habitat", "Bacteria", "Geographical"}, True).transform(input_doc)
        self.assertEqual(expected_doc, actual_doc)

    def test_collapsement_of_same_spans(self):
        tokens = ["Elon", "Musk", "is", "CEO", "of", "Tesla", "."]
        sentences = [Sentence(0, 7)]
        entities = [
            Entity("_", 0, 2, "ELON"), Entity("_", 0, 2, "MUSK"), Entity("_", 5, 6, "COMP"), Entity("_", 5, 6, "ORG")]

        input_doc = Document("_", tokens, sentences, [], entities)

        expected_tokens = ["$ELON$", "is", "CEO", "of", "$COMP$", "."]
        expected_sentences = [Sentence(0, 6)]
        expected_entities = [
            Entity("_", 0, 1, "ELON"), Entity("_", 0, 1, "MUSK"), Entity("_", 4, 5, "COMP"), Entity("_", 4, 5, "ORG")]

        expected_doc = Document("_", expected_tokens, expected_sentences, [], expected_entities)

        actual_doc = EntitiesCollapser({"ELON", "COMP"}).transform(input_doc)
        self.assertEqual(expected_doc, actual_doc)
