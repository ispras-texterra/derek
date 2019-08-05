import unittest
import derek.common.evaluation.metrics as evaluation


class EvaluationTests(unittest.TestCase):
    def test_binary_precision(self):
        predicted = [True, False, True, True, True, True, True, False, False]
        gold = [False, True, True, False, True, True, False, False, False]

        expected_precision = 3 / 6
        self.assertAlmostEqual(
            expected_precision,
            evaluation.binary_precision_score(predicted, gold), delta=1e-6)

    def test_binary_precision_zeros(self):
        predicted = [False, False, False, False]
        gold = [True, False, True, True]

        expected_precision = 0
        self.assertAlmostEqual(
            expected_precision,
            evaluation.binary_precision_score(predicted, gold), delta=1e-6)

    def test_binary_recall(self):
        predicted = [True, False, True, True, True, True, True, False, False]
        gold = [False, True, True, False, True, True, False, False, False]

        expected_recall = 3 / 4
        self.assertAlmostEqual(
            expected_recall,
            evaluation.binary_recall_score(predicted, gold), delta=1e-6)

    def test_binary_recall_zeros(self):
        predicted = [True, True, True, True]
        gold = [False, False, False, False]

        expected_recall = 0
        self.assertAlmostEqual(
            expected_recall,
            evaluation.binary_recall_score(predicted, gold), delta=1e-6)

    def test_binary_f1(self):
        predicted = [True, False, True, True, True, True, True, False, False]
        gold = [False, True, True, False, True, True, False, False, False]

        expected_f1 = 3 / 5
        self.assertAlmostEqual(
            expected_f1,
            evaluation.binary_f1_score(predicted, gold), delta=1e-6)

    def test_binary_f1_zeros(self):
        predicted = [False]
        gold = [True]

        expected_f1 = 0
        self.assertAlmostEqual(
            expected_f1,
            evaluation.binary_f1_score(predicted, gold), delta=1e-6)

    def test_binary_micro_avg_precision(self):
        segments_predicted = [[False, True, False, True], [False, False, False, True]]
        segments_gold = [[False, True, True, True], [True, False, False, True]]

        expected_micro_precision = 3 / 3
        self.assertAlmostEqual(
            expected_micro_precision,
            evaluation.binary_micro_avg_score(evaluation.binary_precision_score, segments_predicted, segments_gold),
            delta=1e-6)

    def test_binary_micro_avg_recall(self):
        segments_predicted = [[False, True, False, True], [False, False, False, True]]
        segments_gold = [[False, True, True, True], [True, False, False, True]]

        expected_micro_recall = 3 / 5
        self.assertAlmostEqual(
            expected_micro_recall,
            evaluation.binary_micro_avg_score(evaluation.binary_recall_score, segments_predicted, segments_gold),
            delta=1e-6)

    def test_binary_micro_avg_f1(self):
        segments_predicted = [[False, True, False, True], [False, False, False, True]]
        segments_gold = [[False, True, True, True], [True, False, False, True]]

        expected_micro_f1 = 3 / 4
        self.assertAlmostEqual(
            expected_micro_f1,
            evaluation.binary_micro_avg_score(evaluation.binary_f1_score, segments_predicted, segments_gold),
            delta=1e-6)

    def test_binary_macro_avg_precision(self):
        segments_predicted = [[False, True, False, True], [False, False, False, True]]
        segments_gold = [[False, True, True, True], [True, False, False, True]]

        precision_1 = 2 / 2
        precision_2 = 1 / 1

        expected_macro_precision = (precision_1 + precision_2) / 2
        self.assertAlmostEqual(
            expected_macro_precision,
            evaluation.binary_macro_avg_score(evaluation.binary_precision_score, segments_predicted, segments_gold),
            delta=1e-6)

    def test_binary_macro_avg_recall(self):
        segments_predicted = [[False, True, False, True], [False, False, False, True]]
        segments_gold = [[False, True, True, True], [True, False, False, True]]

        recall_1 = 2 / 3
        recall_2 = 1 / 2

        expected_macro_recall = (recall_1 + recall_2) / 2
        self.assertAlmostEqual(
            expected_macro_recall,
            evaluation.binary_macro_avg_score(evaluation.binary_recall_score, segments_predicted, segments_gold),
            delta=1e-6)

    def test_binary_macro_avg_f1(self):
        segments_predicted = [[False, True, False, True], [False, False, False, True]]
        segments_gold = [[False, True, True, True], [True, False, False, True]]

        f1_1 = 4 / 5
        f1_2 = 2 / 3

        expected_macro_f1 = (f1_1 + f1_2) / 2
        self.assertAlmostEqual(
            expected_macro_f1,
            evaluation.binary_macro_avg_score(evaluation.binary_f1_score, segments_predicted, segments_gold),
            delta=1e-6)

    def test_ir_precision(self):
        predicted = {3, 4, 5}
        gold = {1, 2, 3}

        expected_precision = 1 / 3

        self.assertAlmostEqual(
            expected_precision,
            evaluation.ir_precision_score(predicted, gold), delta=1e-6)

    def test_ir_precision_zero(self):
        predicted = set()
        gold = {1, 2, 3}

        expected_precision = 0

        self.assertAlmostEqual(
            expected_precision,
            evaluation.ir_precision_score(predicted, gold), delta=1e-6)

    def test_ir_recall(self):
        predicted = {3, 4, 5}
        gold = {1, 2, 3}

        expected_recall = 1 / 3

        self.assertAlmostEqual(
            expected_recall,
            evaluation.ir_recall_score(predicted, gold), delta=1e-6)

    def test_ir_recall_zero(self):
        predicted = {3, 4, 5}
        gold = set()

        expected_recall = 0

        self.assertAlmostEqual(
            expected_recall,
            evaluation.ir_recall_score(predicted, gold), delta=1e-6)

    def test_ir_f1(self):
        predicted = {3, 4, 5}
        gold = {1, 2, 3}

        expected_f1 = 1 / 3

        self.assertAlmostEqual(
            expected_f1,
            evaluation.ir_f1_score(predicted, gold), delta=1e-6)

    def test_ir_f1_zero(self):
        predicted = set()
        gold = set()

        expected_f1 = 0

        self.assertAlmostEqual(
            expected_f1,
            evaluation.ir_f1_score(predicted, gold), delta=1e-6)

    def test_ir_micro_avg_precision(self):
        predicted = [{3, 4, 5}, {1, 2, 0}]
        gold = [{4, 3}, {1, 2, 4, 6}]

        expected_precision = 2 / 3

        self.assertAlmostEqual(
            expected_precision,
            evaluation.ir_micro_avg_score(evaluation.ir_precision_score, predicted, gold), delta=1e-6)

    def test_ir_micro_avg_recall(self):
        predicted = [{3, 4, 5}, {1, 2, 0}]
        gold = [{4, 3}, {1, 2, 4, 6}]

        expected_recall = 2 / 3

        self.assertAlmostEqual(
            expected_recall,
            evaluation.ir_micro_avg_score(evaluation.ir_recall_score, predicted, gold), delta=1e-6)

    def test_ir_micro_avg_f1(self):
        predicted = [{3, 4, 5}, {1, 2, 0}]
        gold = [{4, 3}, {1, 2, 4, 6}]

        expected_f1 = 2 / 3

        self.assertAlmostEqual(
            expected_f1,
            evaluation.ir_micro_avg_score(evaluation.ir_f1_score, predicted, gold), delta=1e-6)

    def test_ir_macro_avg_precision(self):
        predicted = [{3, 4, 5}, {1, 2, 0}]
        gold = [{4, 3}, {1, 2, 0}]

        expected_precision = 5 / 6

        self.assertAlmostEqual(
            expected_precision,
            evaluation.ir_macro_avg_score(evaluation.ir_precision_score, predicted, gold), delta=1e-6)

    def test_ir_macro_avg_recall(self):
        predicted = [{3, 4, 5}, {1, 2, 0}]
        gold = [{4, 3}, {1, 2, 0}]

        expected_recall = 1

        self.assertAlmostEqual(
            expected_recall,
            evaluation.ir_macro_avg_score(evaluation.ir_recall_score, predicted, gold), delta=1e-6)

    def test_ir_macro_avg_f1(self):
        predicted = [{3, 4, 5}, {1, 2, 0}]
        gold = [{4, 3}, {1, 2, 0}]

        expected_f1 = 9 / 10

        self.assertAlmostEqual(
            expected_f1,
            evaluation.ir_macro_avg_score(evaluation.ir_f1_score, predicted, gold), delta=1e-6)


if __name__ == "__main__":
    unittest.main()
