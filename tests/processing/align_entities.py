import unittest

from derek.data.helper import align_raw_entities
from derek.data.model import Entity


class AlignRawEntitiesTest(unittest.TestCase):
    @staticmethod
    def _r_e(start, end):
        return {"id": "test", "start": start, "end": end, "type": "test"}

    @staticmethod
    def _ent(start, end):
        return Entity("test", start, end, "test")

    def test_normal_alignment(self):
        raw_tokens = [(0, 4), (5, 7), (7, 12), (14, 17), (18, 20)]
        raw_entities = [self._r_e(5, 7), self._r_e(14, 17)]
        expected = [self._ent(1, 2), self._ent(3, 4)]

        self.assertEqual(expected, align_raw_entities(raw_entities, raw_tokens))

    def test_inner_entities(self):
        raw_tokens = [(0, 4), (5, 7), (7, 12), (14, 17)]
        raw_entities = [self._r_e(6, 7), self._r_e(10, 16)]
        expected = [self._ent(1, 2), self._ent(2, 4)]

        self.assertEqual(expected, align_raw_entities(raw_entities, raw_tokens))

    def test_shifted_entities(self):
        raw_tokens = [(0, 4), (5, 7), (7, 12), (14, 17)]
        raw_entities = [self._r_e(4, 7), self._r_e(5, 7), self._r_e(10, 13), self._r_e(10, 17)]
        expected = [self._ent(1, 2), self._ent(1, 2), self._ent(2, 3), self._ent(2, 4)]

        self.assertEqual(expected, align_raw_entities(raw_entities, raw_tokens))

    def test_different_entities(self):
        raw_tokens = [(0, 4), (5, 7), (7, 12), (14, 17), (18, 20)]
        raw_entities = [
            self._r_e(0, 4), self._r_e(0, 5), self._r_e(0, 6), self._r_e(0, 7), self._r_e(0, 8),
            self._r_e(4, 7), self._r_e(4, 8), self._r_e(4, 12), self._r_e(14, 20)
        ]
        expected = [
            self._ent(0, 1), self._ent(0, 1), self._ent(0, 2), self._ent(0, 2), self._ent(0, 3),
            self._ent(1, 2), self._ent(1, 3), self._ent(1, 3), self._ent(3, 5)
        ]

        self.assertEqual(expected, align_raw_entities(raw_entities, raw_tokens))
