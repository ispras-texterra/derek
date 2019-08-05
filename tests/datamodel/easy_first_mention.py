import unittest

from derek.coref.data.chains_collection import collect_easy_first_mention_chains
from tests.datamodel.chain_collection import _create_entity, _create_chain


class TestEasyFirstMention(unittest.TestCase):
    def test_1_pair(self):
        pairs = {
            (_create_entity(0), _create_entity(1)): {None: 0.1, "coref": 0.9},
        }

        actual_chains = collect_easy_first_mention_chains(pairs)

        expected_chains = [
            _create_chain([0, 1])
        ]
        self.assertEqual(expected_chains, actual_chains)

    def test_2_pair(self):
        pairs = {
            (_create_entity(0), _create_entity(1)): {None: 0.1, "coref": 0.9},
            (_create_entity(0), _create_entity(2)): {None: 0.1, "coref": 0.9},
        }

        actual_chains = collect_easy_first_mention_chains(pairs)

        expected_chains = [
            _create_chain([0, 1, 2])
        ]
        self.assertEqual(expected_chains, actual_chains)

    def test_2_pair_no_rel(self):
        pairs = {
            (_create_entity(0), _create_entity(1)): {None: 0.1, "coref": 0.9},
            (_create_entity(0), _create_entity(2)): {None: 0.6, "coref": 0.4},
        }

        actual_chains = collect_easy_first_mention_chains(pairs)

        expected_chains = [
            _create_chain([0, 1])
        ]
        self.assertEqual(expected_chains, actual_chains)

    def test_2_pair_unlinked(self):
        pairs = {
            (_create_entity(0), _create_entity(1)): {None: 0.9, "coref": 0.1},
            (_create_entity(0), _create_entity(2)): {None: 0.35, "coref": 0.65},
            (_create_entity(1), _create_entity(2)): {None: 0.4, "coref": 0.6},
        }

        actual_chains = collect_easy_first_mention_chains(pairs)

        expected_chains = [
            _create_chain([0, 2])
        ]
        self.assertEqual(expected_chains, actual_chains)

    def test_2_clusters(self):
        pairs = {
            (_create_entity(0), _create_entity(1)): {None: 0.9, "coref": 0.1},
            (_create_entity(0), _create_entity(2)): {None: 0.35, "coref": 0.65},
            (_create_entity(1), _create_entity(2)): {None: 0.4, "coref": 0.6},

            (_create_entity(1), _create_entity(3)): {None: 0.4, "coref": 0.6},
            (_create_entity(1), _create_entity(0)): {None: 0.4, "coref": 0.6},
        }

        actual_chains = collect_easy_first_mention_chains(pairs)

        expected_chains = [
            _create_chain([0, 2]),
            _create_chain([1, 3]),
        ]
        self.assertEqual(expected_chains, actual_chains)

    def test_3_clusters(self):
        pairs = {
            (_create_entity(0), _create_entity(1)): {None: 0.9, "coref": 0.1},
            (_create_entity(0), _create_entity(2)): {None: 0.35, "coref": 0.65},
            (_create_entity(1), _create_entity(2)): {None: 0.4, "coref": 0.6},

            (_create_entity(1), _create_entity(3)): {None: 0.3, "coref": 0.7},
            (_create_entity(1), _create_entity(0)): {None: 0.4, "coref": 0.6},
            (_create_entity(3), _create_entity(0)): {None: 0.3, "coref": 0.7},

            (_create_entity(4), _create_entity(1)): {None: 0.4, "coref": 0.6},
            (_create_entity(6), _create_entity(5)): {None: 0.4, "coref": 0.6},
        }

        actual_chains = collect_easy_first_mention_chains(pairs)

        expected_chains = [
            _create_chain([0, 2]),
            _create_chain([1, 3, 4]),
            _create_chain([5, 6]),
        ]
        self.assertEqual(expected_chains, actual_chains)
