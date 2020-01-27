import unittest

from derek.common.evaluation.results_controllers import history_improvement_controller


class HistoryImprovementControllerTests(unittest.TestCase):
    def test_zero_length(self):
        self._test_cntrl(history_improvement_controller(0), [False])
        self._test_cntrl(history_improvement_controller(0), [True, False])
        self._test_cntrl(history_improvement_controller(0), [True, True, False])

    def test_nonzero_length(self):
        self._test_cntrl(history_improvement_controller(1), [False, False])
        self._test_cntrl(history_improvement_controller(1), [True, True, False, True, False, True, False, False])

        self._test_cntrl(history_improvement_controller(2), [False, False, False])
        self._test_cntrl(history_improvement_controller(2), [False, False, True, False, False, False])
        self._test_cntrl(history_improvement_controller(2), [True, True, True, False, False, False])

    def _test_cntrl(self, cntrl, inputs):
        next(cntrl)
        generator_length = 0
        for value in inputs:
            try:
                cntrl.send(value)
                generator_length += 1
            except StopIteration:
                break

        expected_generator_length = len(inputs) - 1
        self.assertEqual(generator_length, expected_generator_length)
