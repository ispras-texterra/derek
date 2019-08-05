import unittest

from derek.data.model import Sentence, Entity, SortedSpansSet, Document
from derek.net.feature_extraction.feature_extractor import generate_feature_extractor as net_fe_factory, \
    GroupingFeatureExtractor


class TestGroupingFeatureExtractorClass(unittest.TestCase):
    def setUp(self) -> None:
        tokens = ['Planning', 'of', 'work', 'of', 'Elon', "by", "Elon", "in", "LA", "in", "USA", "."]
        sents = [Sentence(0, 12)]
        ents = [Entity("_", 4, 5, "PER"), Entity("_", 6, 7, "PER"), Entity("_", 8, 9, "ORG"), Entity("_", 10, 11, "ORG")]
        nes = SortedSpansSet([
                Entity("gen", 0, 1, "STUFF"),
                Entity("gen", 4, 5, "PERORG"), Entity("gen", 6, 7, "PERORG"),
                Entity("gen", 8, 9, "PERORG"), Entity("gen", 10, 11, "PERORG")
        ])

        self.doc = Document('', tokens, sents, [], ents, extras={"ne": nes})

    def test_with_NET(self):
        net_fe, _, _ = net_fe_factory(
            [self.doc],
            {
                "internal_emb_size": 10, "ne_type_in_classifier_size": 10,
                "token_position_size": 10, "max_word_distance": 5
            })

        groups = [
            (self.doc.extras["ne"][0],),
            (self.doc.extras["ne"][1], self.doc.extras["ne"][2]),
            (self.doc.extras["ne"][3], self.doc.extras["ne"][4])
        ]

        fe = GroupingFeatureExtractor(net_fe, group_level_features=("labels_mask",))
        extr_groups, samples = zip(*fe.extract_features_from_doc(self.doc, groups, include_labels=True))

        self.assertEqual(tuple(groups), extr_groups)
        self.assertEqual(len(groups), len(samples))

        # STUFF NE's were not mapped to any entity
        self.assertIsNone(samples[0])

        for group, sample in zip(groups[1:], samples[1:]):
            self.assertEqual(sample["chain_len"], len(group))
            self.assertIsInstance(sample, dict)
            self.assertIsInstance(sample["labels"], int)

            self.assertIsInstance(sample["labels_mask"], list)
            self.assertTrue(all(isinstance(a, int) for a in sample["labels_mask"]))

            for other_feature in set(sample.keys()).difference({"labels", "labels_mask", "chain_len"}):
                feature_val = sample[other_feature]
                self.assertIsInstance(feature_val, list)
                self.assertEqual(len(feature_val), len(group))

        # STUFF NE's sample is None, other is dict
        bad_groups = [
            (self.doc.extras["ne"][0], self.doc.extras["ne"][1])
        ]
        self.assertRaises(Exception, fe.extract_features_from_doc, self.doc, bad_groups)

        # different labels in dict samples
        bad_groups = [
            (self.doc.extras["ne"][1], self.doc.extras["ne"][3])
        ]
        self.assertRaises(Exception, fe.extract_features_from_doc, self.doc, bad_groups, include_labels=True)
