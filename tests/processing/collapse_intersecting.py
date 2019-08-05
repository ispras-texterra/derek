import unittest

from derek.data.model import Entity, Relation
from derek.data.helper import collapse_intersecting_entities


def _create_ents(start_ends):
    return [Entity(str(i), start, end, "T1") for i, (start, end) in enumerate(start_ends)]


def _create_rel(e1, e2):
    return Relation(e1, e2, "T1")


class CollapseIntersectingTest(unittest.TestCase):
    def test_no_intersecting(self):
        ents = _create_ents([(0, 3), (3, 4), (5, 10), (11, 12)])
        rels = {
            _create_rel(ents[0], ents[0]), _create_rel(ents[2], ents[1]), _create_rel(ents[3], ents[2])
        }

        self.assertEqual((ents, rels), collapse_intersecting_entities(ents, rels))

    def test_intersecting(self):
        ents = _create_ents([
            (0, 3), (0, 4), (1, 2),
            (5, 7), (5, 7), (6, 7),
            (8, 9),
            (10, 13), (12, 14), (13, 16)])

        rels = {
            _create_rel(ents[1], ents[2]), _create_rel(ents[0], ents[1]),
            _create_rel(ents[5], ents[2]), _create_rel(ents[3], ents[4]),
            _create_rel(ents[6], ents[7]), _create_rel(ents[8], ents[6])
        }

        expected_ents = [
            Entity("0", 0, 4, "T1"),
            Entity("3", 5, 7, "T1"),
            Entity("6", 8, 9, "T1"),
            Entity("7", 10, 16, "T1")
        ]

        expected_rels = {
            _create_rel(expected_ents[0], expected_ents[0]), _create_rel(expected_ents[0], expected_ents[0]),
            _create_rel(expected_ents[1], expected_ents[0]), _create_rel(expected_ents[1], expected_ents[1]),
            _create_rel(expected_ents[2], expected_ents[3]), _create_rel(expected_ents[3], expected_ents[2])
        }

        self.assertEqual((expected_ents, expected_rels), collapse_intersecting_entities(ents, rels))

    def test_different_types(self):
        ents = [
            # ent2 is smaller: starts are equal, end2 < end1
            Entity("0", 0, 4, "T1"), Entity("1", 0, 1, "T2"),

            # ent2 is smaller: start2 > start1, ends are equal
            Entity("4", 100, 103, "T1"), Entity("5", 102, 103, "T2"),

            # ent2 inside ent1, ent2 is smaller
            Entity("6", 110, 114, "T1"), Entity("7", 112, 113, "T2"),

            # ent2 is larger: starts are equal, end2 > end1
            Entity("8", 200, 201, "T1"), Entity("9", 200, 204, "T2"),
        ]

        expected_ents = [Entity("0", 0, 4, "T1"), Entity("4", 100, 103, "T1"),
                         Entity("6", 110, 114, "T1"), Entity("8", 200, 204, "T2")]
        rels = set()
        self.assertEqual((expected_ents, rels), collapse_intersecting_entities(ents, rels))

    def test_assert_equal_with_different_types(self):
        ents = [
            # equal lenghts, different_types
            Entity("8", 200, 204, "T1"), Entity("9", 200, 204, "T2"),
            ]
        rels = set()
        self.assertRaises(Exception, collapse_intersecting_entities, ents, rels)


    def test_assert_intersecting_entities_with_different_types(self):
        ents = [
            # intersecting with equals lenghts=2
            Entity("2", 5, 7, "T1"), Entity("3", 6, 8, "T2"),

            # intersecting, second is larger
            Entity("8", 200, 202, "T1"), Entity("9", 201, 205, "T2"),

            # intersecting, first is larger
            Entity("10", 210, 214, "T1"), Entity("11", 213, 215, "T2")
        ]

        rels = set()
        self.assertRaises(Exception, collapse_intersecting_entities, ents, rels)
