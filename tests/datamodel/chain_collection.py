import unittest

from derek.coref.data.model import CoreferenceChain
from derek.coref.data.chains_collection import collect_chains
from derek.data.model import Relation, Entity


def _create_chain(idxs):
    entities = []
    for i in idxs:
        entities.append(_create_entity(i))
    return CoreferenceChain(entities)


def _create_rel(idx1, idx2):
    return Relation(_create_entity(idx1), _create_entity(idx2), '_')


def _create_entity(idx):
    return Entity(str(idx), idx, idx + 1, '_')


class TestChainCollection(unittest.TestCase):
    def test_1_chain(self):
        entites = [
            _create_entity(0),
            _create_entity(1),
            _create_entity(2),
            _create_entity(3)
        ]
        rels = {
            _create_rel(0, 1),
            _create_rel(1, 2),
        }
        got_chains = collect_chains(rels, entites)
        expected_chains = [_create_chain([0, 1, 2]), _create_chain([3])]
        self.assertEqual(got_chains, expected_chains)

    def test_2_chains(self):
        rels = {
            _create_rel(0, 1),
            _create_rel(1, 2),
            _create_rel(3, 4),
        }
        got_chains = collect_chains(rels)
        expected_chains = [
            _create_chain([0, 1, 2]),
            _create_chain([3, 4]),
        ]
        self.assertEqual(got_chains, expected_chains)

    def test_2_complex_chains(self):
        rels = {
            _create_rel(0, 1),
            _create_rel(1, 2),
            _create_rel(3, 4),
            _create_rel(2, 5),
            _create_rel(6, 3),
        }
        got_chains = collect_chains(rels)
        expected_chains = [
            _create_chain([0, 1, 2, 5]),
            _create_chain([3, 4, 6]),
        ]
        self.assertEqual(got_chains, expected_chains)

    def test_3_chains(self):
        rels = {
            _create_rel(0, 1),
            _create_rel(1, 2),
            _create_rel(3, 4),
            _create_rel(2, 5),
            _create_rel(6, 3),
            _create_rel(10, 11),
            _create_rel(20, 10),
        }
        got_chains = collect_chains(rels)
        expected_chains = [
            _create_chain([0, 1, 2, 5]),
            _create_chain([3, 4, 6]),
            _create_chain([10, 11, 20]),
        ]
        self.assertEqual(got_chains, expected_chains)

    def test_no_chains(self):
        rels = set()
        got_chains = collect_chains(rels)
        expected_chains = []
        self.assertEqual(got_chains, expected_chains)

    def test_many_entities(self):
        rels = {
            _create_rel(0, 1),
            _create_rel(1, 2),
            _create_rel(2, 1),
            _create_rel(2, 0),
            _create_rel(6, 2),
        }
        got_chains = collect_chains(rels)
        expected_chains = [
            _create_chain([0, 1, 2, 6]),
        ]
        self.assertEqual(got_chains, expected_chains)

    def test_chain_common_mention(self):
        rels = {
            _create_rel(0, 1),
            _create_rel(1, 2),
            _create_rel(3, 4),
            _create_rel(4, 5),
            _create_rel(2, 5),
        }
        got_chains = collect_chains(rels)
        expected_chains = [
            _create_chain([0, 1, 2, 3, 4, 5]),
        ]
        self.assertEqual(got_chains, expected_chains)

    def test_chain_multiple_common_mention(self):
        rels = {
            _create_rel(0, 1),
            _create_rel(1, 2),
            _create_rel(3, 4),
            _create_rel(4, 5),
            _create_rel(2, 5),

            _create_rel(10, 20),
            _create_rel(20, 30),

            _create_rel(100, 200),

            _create_rel(40, 50),
            _create_rel(50, 60),
            _create_rel(60, 70),
            _create_rel(70, 30),
        }
        got_chains = collect_chains(rels)
        expected_chains = [
            _create_chain([0, 1, 2, 3, 4, 5]),
            _create_chain([10, 20, 30, 40, 50, 60, 70]),
            _create_chain([100, 200]),
        ]
        self.assertEqual(got_chains, expected_chains)
