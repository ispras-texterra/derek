import unittest

from derek.data.model import SortedSpansSet, Sentence, Entity, TokenSpan


class TestSpansStorage(unittest.TestCase):
    def setUp(self):
        self.sents = SortedSpansSet([Sentence(6, 9), Sentence(0, 10), Sentence(4, 12), Sentence(6, 9), Sentence(6, 12)])
        self.ents = SortedSpansSet([Entity('', 0, 5, ''), Entity('', 0, 5, ''), Entity('1', 2, 6, ''),
                                    Entity('2', 2, 7, ''), Entity('', 7, 9, ''), Entity('', 7, 9, '')])

    def test_sent_contains(self):
        self.assertTrue(Sentence(0, 10) in self.sents)

    def test_sent_at_token_no_values(self):
        self.assertEqual([], self.sents.at_token(12))

    def test_sent_at_token_single_value(self):
        self.assertEqual([Sentence(0, 10)], self.sents.at_token(0))

    def test_sent_at_token_multiple_values(self):
        self.assertEqual([Sentence(0, 10), Sentence(4, 12)], self.sents.at_token(4))

    def test_ent_indexed_at_token_middle(self):
        self.assertEqual([(1, Entity('1', 2, 6, '')), (2, Entity('2', 2, 7, ''))],
                         self.ents.indexed_at_token(5))

    def test_ent_indexed_at_token_end(self):
        self.assertEqual([(3, Entity('', 7, 9, ''))], self.ents.indexed_at_token(8))

    def test_sent_contained_in_exact(self):
        self.assertEqual([Sentence(6, 9)], self.sents.contained_in(Sentence(6, 9)))

    def test_sent_contained_in_intersect(self):
        self.assertEqual([Sentence(0, 10), Sentence(6, 9)],
                         self.sents.contained_in(Sentence(0, 11)))

    def test_ent_indexed_contained_inexact(self):
        self.assertEqual([(1, Entity('1', 2, 6, '')), (2, Entity('2', 2, 7, ''))],
                         self.ents.indexed_contained_in(Sentence(1, 8)))

    def test_ent_indexed_contained_exact(self):
        self.assertEqual([(1, Entity('1', 2, 6, '')), (2, Entity('2', 2, 7, ''))],
                         self.ents.indexed_contained_in(Sentence(2, 7)))


class TestTokenSpan(unittest.TestCase):
    def test_contains(self):
        self.assertTrue(TokenSpan(4, 7).contains(TokenSpan(5, 6)))
        self.assertTrue(TokenSpan(4, 7).contains(TokenSpan(6, 7)))

        self.assertFalse(TokenSpan(4, 7).contains(TokenSpan(7, 9)))
        self.assertFalse(TokenSpan(4, 7).contains(TokenSpan(3, 4)))
        self.assertFalse(TokenSpan(4, 7).contains(TokenSpan(0, 10)))
        self.assertFalse(TokenSpan(4, 7).contains(TokenSpan(5, 11)))
        self.assertFalse(TokenSpan(4, 7).contains(TokenSpan(4, 8)))

    def test_intersects(self):
        self.assertTrue(TokenSpan(4, 7).intersects(TokenSpan(5, 6)))
        self.assertTrue(TokenSpan(4, 7).intersects(TokenSpan(6, 7)))
        self.assertTrue(TokenSpan(4, 7).intersects(TokenSpan(0, 10)))
        self.assertTrue(TokenSpan(4, 7).intersects(TokenSpan(5, 11)))
        self.assertTrue(TokenSpan(4, 7).intersects(TokenSpan(4, 8)))

        self.assertFalse(TokenSpan(4, 7).intersects(TokenSpan(7, 9)))
        self.assertFalse(TokenSpan(4, 7).intersects(TokenSpan(3, 4)))

    def test_coincides(self):
        self.assertTrue(TokenSpan(4, 7).coincides(TokenSpan(4, 7)))
        self.assertFalse(TokenSpan(4, 7).coincides(TokenSpan(4, 5)))
        self.assertFalse(TokenSpan(4, 7).coincides(TokenSpan(7, 8)))

    def test_token_distance_to(self):
        self.assertEqual(TokenSpan(4, 7).token_distance_to(TokenSpan(5, 6)), 0)
        self.assertEqual(TokenSpan(4, 7).token_distance_to(TokenSpan(6, 7)), 0)
        self.assertEqual(TokenSpan(4, 7).token_distance_to(TokenSpan(0, 10)), 0)
        self.assertEqual(TokenSpan(4, 7).token_distance_to(TokenSpan(5, 11)), 0)
        self.assertEqual(TokenSpan(4, 7).token_distance_to(TokenSpan(4, 8)), 0)

        self.assertEqual(TokenSpan(4, 7).token_distance_to(TokenSpan(7, 9)), 1)
        self.assertEqual(TokenSpan(4, 7).token_distance_to(TokenSpan(10, 12)), 4)
        self.assertEqual(TokenSpan(4, 7).token_distance_to(TokenSpan(3, 4)), 1)
        self.assertEqual(TokenSpan(4, 7).token_distance_to(TokenSpan(0, 1)), 4)

    def test_eq(self):
        self.assertEqual(TokenSpan(10, 15), TokenSpan(10, 15))
        self.assertNotEqual(TokenSpan(10, 15), TokenSpan(10, 11))
        self.assertNotEqual(TokenSpan(10, 15), Sentence(10, 15))

    def test_hash(self):
        self.assertEqual(hash(TokenSpan(10, 15)), hash(TokenSpan(10, 15)))
        self.assertNotEqual(hash(TokenSpan(10, 15)), hash(TokenSpan(100, 150)))

    def test_leq(self):
        self.assertTrue(TokenSpan(1, 2) < TokenSpan(2, 3))
        self.assertTrue(TokenSpan(1, 2) < TokenSpan(2, 3))
        self.assertFalse(TokenSpan(2, 3) < TokenSpan(0, 1))

        self.assertRaises(Exception, lambda: TokenSpan(0, 1) < Sentence(2, 3))
        self.assertRaises(Exception, lambda: Sentence(0, 1) < TokenSpan(2, 3))


class TestEntity(unittest.TestCase):
    def test_eq(self):
        self.assertEqual(Entity("_", 1, 2, "T"), Entity("_", 1, 2, "T"))
        self.assertNotEqual(Entity("_", 1, 2, "T"), Entity("_", 1, 2, "P"))
        self.assertNotEqual(Entity("__", 1, 2, "T"), Entity("_", 1, 2, "T"))
        self.assertNotEqual(Entity("_", 1, 2, "T"), Entity("_", 1, 3, "T"))

        self.assertNotEqual(Entity("_", 1, 2, "T"), Sentence(1, 2))

    def test_hash(self):
        self.assertEqual(hash(Entity("_", 1, 2, "T")), hash(Entity("_", 1, 2, "T")))
        self.assertNotEqual(hash(Entity("_", 1, 2, "T")), hash(Sentence(1, 2)))

    def test_leq(self):
        self.assertTrue(Entity("_", 2, 3, "A") < Entity("_", 2, 4, "A"))
        self.assertTrue(Entity("_", 2, 3, "A") < Entity("_", 2, 3, "B"))
        self.assertTrue(Entity("1", 2, 3, "B") < Entity("11", 2, 3, "B"))
        self.assertTrue(Entity("1", 0, 1, "B") < Entity("1", 2, 3, "B"))

        self.assertRaises(Exception, lambda: Entity("_", 0, 1, "A") < TokenSpan(2, 3))
