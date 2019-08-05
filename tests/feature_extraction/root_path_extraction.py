import unittest

from derek.data.helper import get_marked_tokens_on_root_path_for_span
from derek.data.model import Document, Entity, Sentence, Paragraph
from derek.data.sdp_extraction import compute_sdp, compute_sdp_subtree


class DTPathsExtractionTest(unittest.TestCase):
    def setUp(self):
        doc_tokens = [
            "Human", "and", "tick", "spotted", "fever", "group", "Rickettsia", "isolates", "from", "Israel", ":", "a",
            "genotypic", "analysis", "."] + [

            "The", "precise", "mechanisms", "that", "initiate", "bacterial", "uptake", "have", "not", "yet", "been",
            "elucidated", "."]

        doc_sentences = [Sentence(0, 15), Sentence(15, 28)]
        doc_paragraphs = [Paragraph(0, 2)]

        doc_head_distances = [
            3, -1, -2, 0, 2, 1, 1, 7, -1, -1, 4, 2, 1, 1, -11] + [

            2, 1, 9, 1, -2, 1, -2, 4, 3, 2, 1, 0, -1]

        doc_dt_labels = ["test"] * len(doc_tokens)
        doc_token_features = {"dt_head_distances": doc_head_distances, "dt_labels": doc_dt_labels}

        self.entity_with_one_token_no_root = (6, 7, 0)
        self.entity_with_several_tokens_no_root = (12, 14, 0)
        self.entity_with_one_token_root = (3, 4, 0)
        self.entity_with_several_tokens_root = (22, 27, 1)

        doc_entities = [self.entity_with_one_token_no_root, self.entity_with_several_tokens_no_root,
                        self.entity_with_one_token_root, self.entity_with_several_tokens_root]
        doc_entities = [Entity("", start, end, "") for start, end, _ in doc_entities]

        self.doc = Document("", doc_tokens, doc_sentences, doc_paragraphs, doc_entities,
                            token_features=doc_token_features)

    def test_root_path_entity_with_one_token_no_root(self):
        expected = [
            False, False, False, True, False, False, True, True, False, False, False, False, False, False, True] + [

            False, False, False, False, False, False, False, False, False, False, False, False, False]

        self.assertEqual(
            expected, get_marked_tokens_on_root_path_for_span(self.doc, self.entity_with_one_token_no_root))

    def test_root_path_entity_with_one_token_no_root_with_distance(self):
        expected = [
            False, False, False, 3, False, False, 0, 1, False, False, False, False, False, False, 2] + [

            False, False, False, False, False, False, False, False, False, False, False, False, False]

        self.assertEqual(
            expected, get_marked_tokens_on_root_path_for_span(
                self.doc, self.entity_with_one_token_no_root, add_distance=True))

    def test_root_path_entity_with_several_tokens_no_root(self):
        expected = [
            False, False, False, True, False, False, False, False, False, False, False, False, False, True, True] + [

            False, False, False, False, False, False, False, False, False, False, False, False, False]

        self.assertEqual(
            expected, get_marked_tokens_on_root_path_for_span(self.doc, self.entity_with_several_tokens_no_root))

    def test_root_path_entity_with_several_tokens_no_root_with_distance(self):
        expected = [
            False, False, False, 2, False, False, False, False, False, False, False, False, False, 0, 1] + [

            False, False, False, False, False, False, False, False, False, False, False, False, False]

        self.assertEqual(
            expected, get_marked_tokens_on_root_path_for_span(
                self.doc, self.entity_with_several_tokens_no_root, add_distance=True))

    def test_root_path_entity_with_one_token_root(self):
        expected = [
            False, False, False, True, False, False, False, False, False, False, False, False, False, False, False] + [

            False, False, False, False, False, False, False, False, False, False, False, False, False]

        self.assertEqual(
            expected, get_marked_tokens_on_root_path_for_span(self.doc, self.entity_with_one_token_root))

    def test_root_path_entity_with_one_token_root_with_distance(self):
        expected = [
            False, False, False, 0, False, False, False, False, False, False, False, False, False, False, False] + [

            False, False, False, False, False, False, False, False, False, False, False, False, False]

        self.assertEqual(
            expected, get_marked_tokens_on_root_path_for_span(
                self.doc, self.entity_with_one_token_root, add_distance=True))

    def test_root_path_entity_with_several_tokens_root(self):
        expected = [
            False, False, False, False, False, False, False, False, False, False, False, False, False, False, False] + [

            False, False, False, False, False, False, False, False, False, False, False, True, False]

        self.assertEqual(
            expected, get_marked_tokens_on_root_path_for_span(self.doc, self.entity_with_several_tokens_root))

    def test_root_path_entity_with_several_tokens_root_with_distance(self):
        expected = [
            False, False, False, False, False, False, False, False, False, False, False, False, False, False, False] + [

            False, False, False, False, False, False, False, False, False, False, False, 0, False]

        self.assertEqual(
            expected, get_marked_tokens_on_root_path_for_span(
                self.doc, self.entity_with_several_tokens_root, add_distance=True))

    def test_sdp_root_child(self):
        sent = self.doc.sentences[0]
        source_idx = 3
        target_idx = 4
        expected = [3, 14, 7, 6, 4]

        self.assertEqual(expected, compute_sdp(self.doc, sent, source_idx, target_idx))

    def test_sdp_child_root_child(self):
        sent = self.doc.sentences[0]
        source_idx = 1
        target_idx = 8
        expected = [1, 0, 3, 14, 7, 8]

        self.assertEqual(expected, compute_sdp(self.doc, sent, source_idx, target_idx))

    def test_sdp_child_child(self):
        sent = self.doc.sentences[1]
        source_idx = 15
        target_idx = 16
        expected = [15, 17, 16]

        self.assertEqual(expected, compute_sdp(self.doc, sent, source_idx, target_idx))

    def test_sdp_linked_nodes(self):
        sent = self.doc.sentences[1]
        source_idx = 27
        target_idx = 26
        expected = [27, 26]

        self.assertEqual(expected, compute_sdp(self.doc, sent, source_idx, target_idx))

    def test_sdp_same_node(self):
        sent = self.doc.sentences[1]
        source_idx = 18
        target_idx = 18
        expected = [18]

        self.assertEqual(expected, compute_sdp(self.doc, sent, source_idx, target_idx))

    def test_sdp_subtree_with_root(self):
        sent = self.doc.sentences[0]
        sdp = [3, 14, 7, 6, 4]
        expected = {i for i in range(len(sent))}

        self.assertEqual(expected, compute_sdp_subtree(self.doc, sent, sdp))

    def test_sdp_subtree_without_root(self):
        sent = self.doc.sentences[1]
        sdp = [15, 17, 16]
        expected = {i for i in range(15, 22)}

        self.assertEqual(expected, compute_sdp_subtree(self.doc, sent, sdp))
