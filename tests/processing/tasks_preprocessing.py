import unittest

from derek.data.model import Sentence, Paragraph, Entity, Document, SortedSpansSet
from derek.preprocessors import NERPreprocessor, NETPreprocessor


class NERCPreprocessorsTest(unittest.TestCase):
    def setUp(self) -> None:
        tokens = [
            "Главный", "тренер", "римского", "«", "Лацио", "»", "Симоне", "Индзаги", "продолжит", "работу", "с",
            "командой", ",", "сообщает", "пресс-служба", "клуба", ".", "Ранее", "сообщалось", ",", "что", "в",
            "услугах", "Индзаги", "заинтересованы", "«", "Милан", "»", "и", "«", "Ювентус", "»", ",", "которые",
            "пребывают", "без", "наставников", "после", "ухода", "Дженнаро", "Гаттузо", "и", "Массимилиано", "Аллегри",
            "."
        ]

        sentences = [Sentence(0, 17), Sentence(17, 45)]
        paragraphs = [Paragraph(0, 1)]
        entities = [
            Entity("T1", 4, 5, "Team"),
            Entity("T2", 6, 8, "PlayerCoach1"),
            Entity("T3", 23, 24, "PlayerCoach2"),
            Entity("T4", 26, 27, "TeamFilter"),
            Entity("T5", 30, 31, "Team"),
            Entity("T6", 39, 41, "Coach"),
            Entity("T7", 42, 44, "Coach")
        ]

        self.doc = Document("_", tokens, sentences, paragraphs, entities)

    def test_ner_preprocessor(self):
        filter_types = {"TeamFilter"}
        replacements = {"PlayerCoach1": "Coach", "PlayerCoach2": "Coach"}
        preprocessor = NERPreprocessor(filter_types, replacements)

        expected_entities = [
            Entity("T1", 4, 5, "Team"),
            Entity("T2", 6, 8, "Coach"),
            Entity("T3", 23, 24, "Coach"),
            Entity("T5", 30, 31, "Team"),
            Entity("T6", 39, 41, "Coach"),
            Entity("T7", 42, 44, "Coach")
        ]
        expected_doc = self.doc.without_entities().with_entities(expected_entities)
        self.assertEqual(expected_doc, preprocessor.process_doc(self.doc))

        props = {
            "ent_types_to_filter": ["TeamFilter"],
            "ent_types_merge_pattern": {"Coach": ["PlayerCoach1", "PlayerCoach2"]}
        }

        preprocessor = NERPreprocessor.from_props(props)
        self.assertEqual(expected_doc, preprocessor.process_doc(self.doc))

    def test_net_preprocessor(self):
        filter_types = {"TeamFilter"}
        ne_replacements = {"PlayerCoach1": "Coach", "PlayerCoach2": "Coach"}
        ent_replacements = {"PlayerCoach1": "PlayerCoach", "PlayerCoach2": "PlayerCoach"}
        preprocessor = NETPreprocessor(filter_types, ne_replacements, ent_replacements)

        expected_entities = [
            Entity("T1", 4, 5, "Team"),
            Entity("T2", 6, 8, "PlayerCoach"),
            Entity("T3", 23, 24, "PlayerCoach"),
            Entity("T5", 30, 31, "Team"),
            Entity("T6", 39, 41, "Coach"),
            Entity("T7", 42, 44, "Coach")
        ]
        expected_nes = SortedSpansSet([
            Entity("T1", 4, 5, "Team"),
            Entity("T2", 6, 8, "Coach"),
            Entity("T3", 23, 24, "Coach"),
            Entity("T5", 30, 31, "Team"),
            Entity("T6", 39, 41, "Coach"),
            Entity("T7", 42, 44, "Coach")
        ])

        expected_doc = self.doc.without_entities().with_entities(expected_entities).\
            with_additional_extras({"ne": expected_nes})
        self.assertEqual(expected_doc, preprocessor.process_doc(self.doc))

        props = {
            "ent_types_to_filter": ["TeamFilter"],
            "ne_types_merge_pattern": {"Coach": ["PlayerCoach1", "PlayerCoach2"]},
            "ent_types_merge_pattern": {"PlayerCoach": ["PlayerCoach1", "PlayerCoach2"]}
        }

        preprocessor = NETPreprocessor.from_props(props)
        self.assertEqual(expected_doc, preprocessor.process_doc(self.doc))
