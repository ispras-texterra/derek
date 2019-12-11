import unittest

from derek import trainer_for
from derek.data.model import Sentence, Paragraph, Entity, Document, SortedSpansSet
from tests.test_helper import get_training_hook


class TestNERCManagers(unittest.TestCase):
    def setUp(self) -> None:
        self.docs = []

        tokens = [
            "Главный", "тренер", "римского", "«", "Лацио", "»", "Симоне", "Индзаги", "продолжит", "работу", "с",
            "командой", ",", "сообщает", "пресс-служба", "клуба", ".", "Ранее", "сообщалось", ",",  "что", "в",
            "услугах", "Индзаги", "заинтересованы", "«", "Милан", "»", "и", "«", "Ювентус", "»", ",", "которые",
            "пребывают", "без", "наставников", "после", "ухода", "Дженнаро", "Гаттузо", "и", "Массимилиано", "Аллегри",
            "."
        ]

        sentences = [Sentence(0, 17), Sentence(17, 45)]
        paragraphs = [Paragraph(0, 1)]
        entities = [
            Entity("T1", 4, 5, "Team"),
            Entity("T2", 6, 8, "Coach"),
            Entity("T3", 23, 24, "Coach"),
            Entity("T4", 26, 27, "Team"),
            Entity("T5", 30, 31, "Team"),
            Entity("T6", 39, 41, "Coach"),
            Entity("T7", 42, 44, "Coach")
        ]
        named_entities = [
            Entity("generated", 3, 6, "ORG"),
            Entity("generated", 6, 8, "PER"),
            Entity("generated", 23, 24, "PER"),
            Entity("generated", 25, 28, "ORG"),
            Entity("generated", 29, 32, "ORG"),
            Entity("generated", 39, 41, "PER"),
            Entity("generated", 42, 44, "PER")
        ]

        doc = Document("_", tokens, sentences, paragraphs, entities, extras={"ne": SortedSpansSet(named_entities)})
        self.docs.append(doc)

        tokens = [
            "Врачи", "сборной", "Бразилии", "подтвердили", "травму", "нападающего", "«", "Пари", "Сен-Жермен", "»",
            "Неймара", ",", "полученную", "во", "время", "товарищеского", "матча", "с", "Катаром", "."
        ]

        sentences = [Sentence(0, 20)]
        paragraphs = [Paragraph(0, 1)]
        entities = [
            Entity("T1", 1, 3, "Team"),
            Entity("T2", 7, 9, "Team"),
            Entity("T3", 10, 11, "Player"),
            Entity("T4", 18, 19, "Team")
        ]
        named_entities = [
            Entity("generated", 1, 3, "ORG"),
            Entity("generated", 6, 10, "ORG"),
            Entity("generated", 10, 11, "PER"),
            Entity("generated", 18, 19, "ORG")
        ]

        doc = Document("_", tokens, sentences, paragraphs, entities, extras={"ne": SortedSpansSet(named_entities)})
        self.docs.append(doc)

        self.common_props = {
            "seed": 1,
            "internal_emb_size": 10,
            "learning_rate": 0.005,
            "batch_size": 4,
            "encoding_size": 1,
            "dropout": 0.5,
            "optimizer": "adam",
            "epoch": 2,
            "clip_norm": 5
        }

        self.docs_no_entities = [d.without_entities() for d in self.docs]

    def test_ner_manager(self):
        props = {
            **self.common_props,
            "labelling_strategy": "BIO2",
            "loss": "crf",
            "ne_emb_size": 0
        }

        hook, lst = get_training_hook(self.docs_no_entities)

        with trainer_for("ner")(props) as trainer:
            trainer.train(self.docs, early_stopping_callback=hook)

        # validate that hook was called after each epoch
        self.assertEqual(lst, [True] * props["epoch"])

    def test_net_manager(self):
        props = {
            **self.common_props,
            "loss": "cross_entropy",
            "token_position_size": 1,
            "max_word_distance": 5,
            "aggregation": {"attention": {}}
        }

        hook, lst = get_training_hook(self.docs_no_entities)

        with trainer_for("net")(props) as trainer:
            trainer.train(self.docs, early_stopping_callback=hook)

        # validate that hook was called after each epoch
        self.assertEqual(lst, [True] * props["epoch"])
