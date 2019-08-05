import unittest

import numpy as np

from derek.common.feature_extraction.token_position_feature_extractor import generate_token_position_feature_extractor
from derek.data.model import Document, Sentence, Entity, SortedSpansSet
from derek.rel_ext.feature_extraction.factory import generate_feature_extractor as rel_ext_fe_factory
from derek.rel_ext.syntactic.parser.feature_extraction.factory import generate_feature_extractor as parser_fe_factory
from derek.common.feature_extraction.factory import generate_token_feature_extractor
from derek.rel_ext.feature_extraction.spans_feature_extractor import generate_spans_common_feature_extractor
from tests.test_helper import make_document_from_json_file, load_json_file_as_dict
from derek.ner.feature_extraction.feature_extractor import generate_feature_extractor as ner_fe_factory
from derek.common.feature_extraction.ne_feature_extractor import generate_ne_feature_extractor as ne_fe_factory
from derek.net.feature_extraction.feature_extractor import generate_feature_extractor as net_fe_factory


class TestTokenFeatureExtractorClass(unittest.TestCase):
    def test_no_features(self):
        doc = Document('', ['Go', 'to', 'shop'], [], [])
        token_fe, token_meta = generate_token_feature_extractor([doc], {})
        features = token_fe.extract_features_from_doc(doc, 0, 2)

        self.assertEqual(token_meta.get_precomputed_features(), [])
        self.assertEqual(token_meta.get_embedded_features(), [])
        self.assertEqual(token_meta.get_one_hot_features(), [])
        self.assertEqual(token_meta.get_vectorized_features(), [])
        self.assertEqual(token_meta.get_char_features(), [])

        self.assertEqual(features, {'seq_len': 2})

    def test_we_features(self):
        doc = Document('', ['Planning', 'of', 'work', 'of', 'Elon'], [], [])
        token_fe, token_meta = generate_token_feature_extractor([doc], {"internal_emb_size": 10})
        features = token_fe.extract_features_from_doc(doc, 1, 4)
        words = features['words_0']

        self.assertEqual(len(token_meta.get_precomputed_features()), 1)
        self.assertEqual(token_meta.get_embedded_features(), [])
        self.assertEqual(token_meta.get_one_hot_features(), [])
        self.assertEqual(token_meta.get_vectorized_features(), [])
        self.assertEqual(token_meta.get_char_features(), [])

        self.assertEqual(features['seq_len'], 3)
        self.assertEqual(len(words), 3)

        self.assertNotEqual(words[0], words[1]) # of work
        self.assertEqual(words[0], words[2]) # of of

    def test_pos_emb_features(self):
        doc = Document('', ['Planning', 'of', 'work', 'by', 'Elon'], [], [],
                       token_features={"pos": ['NN', 'IN', 'NN', 'IN', 'NNP']})
        token_fe, token_meta = generate_token_feature_extractor([doc], {"pos_emb_size": 10})
        features = token_fe.extract_features_from_doc(doc, 1, 4)
        pos = features['pos']

        self.assertEqual(token_meta.get_precomputed_features(), [])
        self.assertEqual(len(token_meta.get_embedded_features()), 1)
        self.assertEqual(token_meta.get_one_hot_features(), [])
        self.assertEqual(token_meta.get_vectorized_features(), [])
        self.assertEqual(token_meta.get_char_features(), [])

        self.assertEqual(features['seq_len'], 3)
        self.assertEqual(len(pos), 3)

        self.assertNotEqual(pos[0], pos[1])  # of work
        self.assertEqual(pos[0], pos[2])  # of by

    def test_char_features(self):
        doc = Document('', ['Go', 'into'], [], [])
        token_fe, token_meta = generate_token_feature_extractor([doc], {"char_embedding_size": 10})
        features = token_fe.extract_features_from_doc(doc, 0, 2)
        chars = features['chars']

        self.assertEqual(token_meta.get_precomputed_features(), [])
        self.assertEqual(token_meta.get_embedded_features(), [])
        self.assertEqual(token_meta.get_one_hot_features(), [])
        self.assertEqual(token_meta.get_vectorized_features(), [])
        self.assertEqual(len(token_meta.get_char_features()), 1)

        self.assertEqual(features['seq_len'], 2)
        self.assertEqual(len(chars), 2)
        self.assertEqual(len(chars[0]), 2)
        self.assertEqual(len(chars[1]), 4)

        self.assertNotEqual(chars[0][0], chars[0][1])  # G o
        self.assertEqual(chars[0][1], chars[1][3])  # o o

    def test_vectorized_features(self):
        doc = Document(
            '', ['Planning', 'of', 'work', 'by', 'Elon'], [], [],
            token_features={
              "vectors": [np.array([1, 2]), np.array([1, 1]), np.array([0, 0]), np.array([0, 1]), np.array([9, 10])]
            })
        token_fe, token_meta = generate_token_feature_extractor([doc], {"vectors_keys": ["vectors"]})
        features = token_fe.extract_features_from_doc(doc, 1, 4)
        vectors = features['vectors']

        self.assertEqual(token_meta.get_precomputed_features(), [])
        self.assertEqual(len(token_meta.get_embedded_features()), 0)
        self.assertEqual(token_meta.get_one_hot_features(), [])
        self.assertEqual(token_meta.get_vectorized_features(), [{"name": "vectors", "size": 2}])
        self.assertEqual(token_meta.get_char_features(), [])

        self.assertEqual(features['seq_len'], 3)
        self.assertEqual(len(vectors), 3)

        self.assertEqual(doc.token_features["vectors"][1: 4], vectors)


class TestTokenPositionFeatureExtractorClass(unittest.TestCase):
    def test_no_features(self):
        doc = Document('', ['Go', 'to', 'shop'], [], [])
        tp_fe, tp_meta = generate_token_position_feature_extractor({})
        features = tp_fe.extract_features_from_doc(doc, 0, 3, (2, 3, 0))

        self.assertEqual(tp_meta.get_embedded_features(), [])
        self.assertEqual(tp_meta.get_one_hot_features(), [])
        self.assertEqual(tp_meta.get_vectorized_features(), [])
        self.assertEqual(features, {})

    def test_token_position_emb_features(self):
        doc = Document('', ['Go', 'to', 'shop'], [], [])
        tp_fe, tp_meta = generate_token_position_feature_extractor(
            {"token_position_size": 10, "max_word_distance": 0})
        features = tp_fe.extract_features_from_doc(doc, 0, 3, (0, 1, 0))
        token_position = features['token_position']

        self.assertEqual(len(tp_meta.get_embedded_features()), 1)
        self.assertEqual(tp_meta.get_one_hot_features(), [])
        self.assertEqual(tp_meta.get_vectorized_features(), [])
        self.assertEqual(len(token_position), 3)
        self.assertNotEqual(token_position[0], token_position[1])
        self.assertEqual(token_position[1], token_position[2])

    def test_sent_position_one_hot_features(self):
        doc = Document('', ['Go', 'to', 'shop'], [Sentence(0, 3)], [])
        tp_fe, tp_meta = generate_token_position_feature_extractor(
            {"sent_position_size": 0, "max_sent_distance": 5})
        features = tp_fe.extract_features_from_doc(doc, 0, 3, (1, 2, 0))
        sent_position = features['sent_position']

        self.assertEqual(tp_meta.get_embedded_features(), [])
        self.assertEqual(len(tp_meta.get_one_hot_features()), 1)
        self.assertEqual(tp_meta.get_vectorized_features(), [])
        self.assertEqual(len(sent_position), 3)
        self.assertEqual(sent_position[0], sent_position[1])
        self.assertEqual(sent_position[1], sent_position[2])

    def test_at_root_dt_path_one_hot_features(self):
        doc = Document('', ['Go', 'to', 'shop'], [Sentence(0, 3)], [],
                       token_features={'dt_head_distances': [0, -1, -2]})
        tp_fe, tp_meta = generate_token_position_feature_extractor(
            {"at_root_dt_path_size": 0})
        features = tp_fe.extract_features_from_doc(doc, 0, 3, (2, 3, 0))
        at_root_dt_path = features['at_root_dt_path']

        self.assertEqual(tp_meta.get_embedded_features(), [])
        self.assertEqual(len(tp_meta.get_one_hot_features()), 1)
        self.assertEqual(tp_meta.get_vectorized_features(), [])
        self.assertEqual(len(at_root_dt_path), 3)
        self.assertNotEqual(at_root_dt_path[0], at_root_dt_path[1])
        self.assertEqual(at_root_dt_path[0], at_root_dt_path[2])


class TestNEFeatureExtractorClass(unittest.TestCase):
    def test_ne_features(self):
        ents = [Entity("_", 4, 5, "PER"), Entity("_", 6, 7, "PER")]
        doc = Document('', ['Planning', 'of', 'work', 'of', 'Elon', "by", "Elon"], [Sentence(0, 7)], [],
                       extras={'ne': SortedSpansSet(ents)})
        fe, meta = ne_fe_factory([doc], {"ne_emb_size": 10})
        features = fe.extract_features_from_doc(doc, 3, 7)['ne']
        self.assertEqual(len(meta.get_embedded_features()), 1)
        self.assertEqual(len(features), 4)
        self.assertEqual(features[0], features[2])  # O O
        self.assertEqual(features[1], features[3])  # I-PER I-PER
        self.assertNotEqual(features[0], features[1])  # O I-PER


class TestNERFeatureExtractorClass(unittest.TestCase):
    def test_without_labels(self):
        ents = [Entity("_", 4, 5, "PER"), Entity("_", 6, 7, "PER")]
        doc = Document('', ['Planning', 'of', 'work', 'of', 'Elon', "by", "Elon"], [Sentence(0, 7)], [], ents)

        ner_fe, token_meta = ner_fe_factory([doc], {"internal_emb_size": 10})
        doc = doc.without_entities()
        features, = ner_fe.extract_features_from_doc(doc)

        words = features['words_0']
        self.assertEqual(features['seq_len'], 7)
        self.assertEqual(len(words), 7)

        self.assertNotEqual(words[0], words[1])  # Planning of
        self.assertEqual(words[1], words[3])  # of of

        self.assertRaises(KeyError, lambda: features['labels'])

    def test_with_labels(self):
        ents = [Entity("_", 4, 5, "PER"), Entity("_", 6, 7, "PER")]
        doc = Document('', ['Planning', 'of', 'work', 'of', 'Elon', "by", "Elon"], [Sentence(0, 7)], [], ents)

        ner_fe, token_meta = ner_fe_factory([doc], {"internal_emb_size": 10})
        # one sentence in doc -> one sample
        features, = ner_fe.extract_features_from_doc(doc, include_labels=True)

        words = features['words_0']
        self.assertEqual(features['seq_len'], 7)
        self.assertEqual(len(words), 7)

        self.assertNotEqual(words[0], words[1])  # Planning of
        self.assertEqual(words[1], words[3])  # of of

        labels = features["labels"]
        self.assertEqual(len(labels), 7)
        self.assertEqual(labels[4], labels[6])  # B-PER, B-PER
        self.assertNotEqual(labels[3], labels[4])  # O, B-PER
        self.assertEqual(labels[0], labels[1])  # O, O


class TestNETFeatureExtractorClass(unittest.TestCase):
    def setUp(self) -> None:
        ents = [Entity("_", 4, 5, "Master"), Entity("_", 6, 7, "CEO"), Entity("_", 8, 9, "CITY")]
        self.doc_ne = [
                Entity("gen", 0, 1, "STUFF"), Entity("gen", 4, 5, "PER"), Entity("gen", 4, 5, "ELON"),
                Entity("gen", 6, 7, "PER"), Entity("gen", 6, 7, "RESTR"), Entity("gen", 8, 9, "LOC"),
                Entity("gen", 8, 9, "RESTR"), Entity("gen", 10, 11, "LOC"),
                Entity("gen", 12, 13, "PER"), Entity("gen", 18, 19, "LOC")
        ]

        extras = {
            "ne": self.doc_ne
        }

        self.doc = Document(
            '', [
                'Planning', 'of', 'work', 'of', 'Elon', "by", "Elon", "in", "LA", "in", "USA", "."
                'Elon', 'is', 'going', 'to', 'land', 'on', 'Mars', "."
            ],
            [Sentence(0, 12), Sentence(12, 20)], [], ents, extras=extras)

        self.net_fe, _, _ = net_fe_factory(
            [self.doc],
            {
                "internal_emb_size": 10,  "ne_type_in_classifier_size": 10,
                "token_position_size": 10, "max_word_distance": 5, "restricted_ne_types": ["RESTR"]
            })

    def test_without_labels(self):
        actual_entities, actual_samples = zip(*self.net_fe.extract_features_from_doc(self.doc.without_entities()))
        self._test_samples(actual_entities, actual_samples)

        for sample in actual_samples:
            if isinstance(sample, dict):
                self.assertNotIn("labels", sample)

    def test_with_labels(self):
        actual_entities, actual_samples = zip(*self.net_fe.extract_features_from_doc(self.doc, include_labels=True))
        self._test_samples(actual_entities, actual_samples)

        for sample in actual_samples:
            if isinstance(sample, dict):
                self.assertIn("labels", sample)

        self.assertNotEqual(actual_samples[1]["labels"], actual_samples[3]["labels"])  # MASTER CEO
        self.assertNotEqual(actual_samples[5]["labels"], actual_samples[7]["labels"])  # CITY None

    def _test_samples(self, actual_entities, actual_samples):
        self.assertListEqual(list(actual_entities), self.doc_ne)

        # STUFF entities are not classified, RESTR entities are restricted
        for idx in [0, 4, 6]:
            self.assertIsNone(actual_samples[idx])

        # ELON entities always classified as Master
        self.assertEqual("Master", actual_samples[2])

        # Need to classify PER and LOC entities
        first_sent_samples_to_classify = [actual_samples[i] for i in [1, 3, 5, 7]]

        for ent_sample in first_sent_samples_to_classify:
            self.assertIsInstance(ent_sample, dict)
            words = ent_sample['words_0']
            self.assertEqual(ent_sample['seq_len'], 12)
            self.assertEqual(len(words), 12)

            self.assertNotEqual(words[0], words[1])  # Planning of
            self.assertEqual(words[1], words[3])  # of of

        ent_1_sample, ent_3_sample, ent_5_sample, ent_7_sample = first_sent_samples_to_classify

        self.assertEqual(ent_5_sample["ne_type_in_classifier"], ent_7_sample["ne_type_in_classifier"])  # LOC LOC
        self.assertEqual(ent_1_sample["ne_type_in_classifier"], ent_3_sample["ne_type_in_classifier"])  # PER PER
        self.assertNotEqual(ent_1_sample["ne_type_in_classifier"], ent_5_sample["ne_type_in_classifier"])  # LOC PER

        self.assertEqual(ent_5_sample["token_position"][7], ent_7_sample["token_position"][9])  # -1 -1
        self.assertEqual(ent_5_sample["token_position"][9], ent_7_sample["token_position"][11])  # 1 1

        self.assertNotEqual(ent_3_sample["labels_mask"], ent_5_sample["labels_mask"])  # MASTER, CEO | CITY, None
        self.assertEqual(ent_5_sample["labels_mask"], ent_7_sample["labels_mask"])
        self.assertEqual(ent_1_sample["labels_mask"], ent_3_sample["labels_mask"])

        self.assertEqual(ent_1_sample["indices"], [[4, 5]])
        self.assertEqual(ent_3_sample["indices"], [[6, 7]])
        self.assertEqual(ent_5_sample["indices"], [[8, 9]])
        self.assertEqual(ent_7_sample["indices"], [[10, 11]])

        ent_8_sample, ent_9_sample = actual_samples[8], actual_samples[9]
        self.assertEqual(ent_8_sample["indices"], [[0, 1]])
        self.assertEqual(ent_9_sample["indices"], [[6, 7]])


class TestRelExtFeatureExtractorClass(unittest.TestCase):
    def test_rel_ext_feature_extractor(self):
        doc = make_document_from_json_file("tests/data/feature_extractor/geo_bact_replacement.json")
        props = load_json_file_as_dict("tests/data/feature_extractor/props.json")
        expected_features = load_json_file_as_dict("tests/data/feature_extractor/feature_extraction_result.json")

        token_fe, _ = generate_spans_common_feature_extractor([doc], props)
        extractor, _ = rel_ext_fe_factory([doc], props, token_fe)

        extracted_features, _ = extractor.extract_features_from_doc(doc, include_labels=True)
        for sample in extracted_features:
            for key in {
                "seq_len", "words_0", "borders", "dt_head_distances",
                "e1_token_position", "e2_token_position", "span1_token_position", "span2_token_position", "indices"
            }:
                self.assertTrue(key in sample)
                del sample[key]

        self.assertEqual(extracted_features, expected_features)


class TestParserFeatureExtractorClass(unittest.TestCase):
    def test_parser_feature_extractor_default_strategy(self):
        doc_file = "tests/data/parser_feature_extractor/geo_bact_replacement.json"
        props_file = "tests/data/parser_feature_extractor/props.json"
        expected_features_file = "tests/data/parser_feature_extractor/feature_extraction_result.json"

        self._test_parser_fe(doc_file, props_file, expected_features_file)

    def test_parser_feature_extractor_pos_filtering_strategy(self):
        doc_file = "tests/data/parser_feature_extractor/geo_bact_replacement.json"
        props_file = "tests/data/parser_feature_extractor/props_pos_sampling.json"
        expected_features_file = "tests/data/parser_feature_extractor/pos_sampling_feature_extraction_result.json"

        self._test_parser_fe(doc_file, props_file, expected_features_file)

    def _test_parser_fe(self, doc_file, props_file, expected_features_file):
        doc = make_document_from_json_file(doc_file)
        props = load_json_file_as_dict(props_file)
        expected_features = load_json_file_as_dict(expected_features_file)

        token_fe, _ = generate_spans_common_feature_extractor([doc], props)
        extractor, _ = parser_fe_factory([doc], props, token_fe)

        extracted_features = extractor.extract_features_from_doc(doc)
        for sample in extracted_features:
            for key in {
                "seq_len", "words_0", "span1_token_position", "span2_token_position",
                "head_token_position", "dep_token_position", "indices"
            }:
                self.assertTrue(key in sample)
                del sample[key]

        self.assertEqual(extracted_features, expected_features)


if __name__ == "__main__":
    unittest.main()
