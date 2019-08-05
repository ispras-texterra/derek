import unittest

import numpy as np

from derek.common.nn.batchers import get_standard_batcher_factory, get_bucketed_batcher_factory, \
    get_segment_batcher_factory, pad_sequences, pad_sequences2d, pad_sequences3d


def batch_to_list(batch):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in batch.items()}


def get_batches(*args, **kwargs):
    return list(map(batch_to_list, get_standard_batcher_factory(*args, **kwargs)()))


def get_bucketed_batches(*args, **kwargs):
    return list(map(batch_to_list, get_bucketed_batcher_factory(*args, **kwargs)()))


def get_segment_batches(*args, **kwargs):
    return list(map(batch_to_list, get_segment_batcher_factory(*args, **kwargs)()))


class GetbatchesTest(unittest.TestCase):
    def setUp(self):
        # in most tests we have no padding
        self.get_padding = lambda k: (None, 0)

    def test_one_example(self):
        samples = [{
            "f1": [3, 4, 5, 6],
            "f2": 3,
            "f3": [1, 1]
        }]

        true_batches = [{
            "f1": [[3, 4, 5, 6]],
            "f2": [3],
            "f3": [[1, 1]]
        }]

        batches = get_batches(samples, 100, self.get_padding)

        self.assertEqual(true_batches, batches)

    def test_two_examples(self):
        samples = [{
            "f1": [3, 4, 5, 6],
            "f2": 3,
            "f3": [1, 1]
        }, {
            "f1": [1, 1, 2],
            "f2": 0,
            "f3": [0, 1, 1]
        }
        ]

        true_batches = [{
            "f1": [[3, 4, 5, 6], [1, 1, 2]],
            "f2": [3, 0],
            "f3": [[1, 1], [0, 1, 1]]
        }]

        batches = get_batches(samples, 100, self.get_padding, buffer_size=1)

        self.assertEqual(true_batches, batches)

    def test_full_batch(self):
        samples = [{
            "f1": [3, 4, 5, 6],
            "f2": 3,
            "f3": [1, 1]
        }, {
            "f1": [1, 1, 2],
            "f2": 0,
            "f3": [0, 1, 1]
        }
        ]

        true_batches = [{
            "f1": [[3, 4, 5, 6], [1, 1, 2]],
            "f2": [3, 0],
            "f3": [[1, 1], [0, 1, 1]]
        }]

        batches = get_batches(samples, 2, self.get_padding, buffer_size=1)

        self.assertEqual(true_batches, batches)

    def test_several_batches(self):
        samples = [{
            "f1": [3, 4, 5, 6],
            "f2": 3,
            "f3": [1, 1]
        }, {
            "f1": [1, 1, 2],
            "f2": 0,
            "f3": [0, 1, 1]
        }
        ]

        true_batches = [{
            "f1": [[3, 4, 5, 6]],
            "f2": [3],
            "f3": [[1, 1]]
        }, {
            "f1": [[1, 1, 2]],
            "f2": [0],
            "f3": [[0, 1, 1]]
        }]

        batches = get_batches(samples, 1, self.get_padding, buffer_size=1)

        self.assertEqual(true_batches, batches)

    def test_parted_several_batches(self):
        samples = [{
            "f1": [3, 4, 5, 6],
            "f2": 3,
            "f3": [1, 1]
        }, {
            "f1": [1, 1, 2],
            "f2": 0,
            "f3": [0, 1, 1]
        }, {
            "f1": [6, 5],
            "f2": 7,
            "f3": [0]
        }
        ]

        true_batches = [{
            "f1": [[3, 4, 5, 6], [1, 1, 2]],
            "f2": [3, 0],
            "f3": [[1, 1], [0, 1, 1]]
        }, {
            "f1": [[6, 5]],
            "f2": [7],
            "f3": [[0]]
        }
        ]

        batches = get_batches(samples, 2, self.get_padding, buffer_size=1)

        self.assertEqual(true_batches, batches)

    def test_full_batch_with_padding(self):
        samples = [{
            "f1": [3, 4, 5, 6],
            "f2": 3,
            "f3": [1, 1],
            "f4": [[2, 1], [2, 3]]
        }, {
            "f1": [1, 1, 2],
            "f2": 0,
            "f3": [0, 1, 1],
            "f4": [[2, 1], [2, 3], [4, 5]]
        }
        ]

        padding_values = {
            "f1": (0, 1),
            "f2": (None, 0),
            "f3": (-1, 1),
            "f4": (42, 2)
        }

        true_batches = [{
            "f1": [[3, 4, 5, 6], [1, 1, 2, 0]],
            "f2": [3, 0],
            "f3": [[1, 1, -1], [0, 1, 1]],
            "f4": [[[2, 1], [2, 3], [42, 42]], [[2, 1], [2, 3], [4, 5]]]
        }]

        batches = get_batches(samples, 2, padding_values.get, buffer_size=1)

        self.assertEqual(true_batches, batches)


class GetBucketedBatchesTest(unittest.TestCase):
    def setUp(self):
        # in most tests we have no padding
        self.get_padding = lambda k: (None, 0)
        self.get_bucket = lambda s: s['b']

    def test_one_example(self):
        samples = [{
            "b": 1,
            "f1": [3, 4, 5, 6],
            "f2": 3,
            "f3": [1, 1]
        }]

        true_batches = [{
            "b": [1],
            "f1": [[3, 4, 5, 6]],
            "f2": [3],
            "f3": [[1, 1]]
        }]

        batches = get_bucketed_batches(samples, 100, self.get_padding, self.get_bucket)

        self.assertEqual(true_batches, batches)

    def test_two_examples_same_bucket(self):
        samples = [{
            "b": 1,
            "f1": [3, 4, 5, 6],
            "f2": 3,
            "f3": [1, 1]
        }, {
            "b": 1,
            "f1": [1, 1, 2],
            "f2": 0,
            "f3": [0, 1, 1]
        }
        ]

        true_batches = [{
            "b": [1, 1],
            "f1": [[3, 4, 5, 6], [1, 1, 2]],
            "f2": [3, 0],
            "f3": [[1, 1], [0, 1, 1]]
        }]

        batches = get_bucketed_batches(samples, 100, self.get_padding, self.get_bucket)

        self.assertEqual(true_batches, batches)

    def test_full_batch(self):
        samples = [{
            "b": 1,
            "f1": [3, 4, 5, 6],
            "f2": 3,
            "f3": [1, 1]
        }, {
            "b": 1,
            "f1": [1, 1, 2],
            "f2": 0,
            "f3": [0, 1, 1]
        }
        ]

        true_batches = [{
            "b": [1, 1],
            "f1": [[3, 4, 5, 6], [1, 1, 2]],
            "f2": [3, 0],
            "f3": [[1, 1], [0, 1, 1]]
        }]

        batches = get_bucketed_batches(samples, 2, self.get_padding, self.get_bucket)

        self.assertEqual(true_batches, batches)

    def test_several_batches(self):
        samples = [{
            "b": 1,
            "f1": [3, 4, 5, 6],
            "f2": 3,
            "f3": [1, 1]
        }, {
            "b": 1,
            "f1": [1, 1, 2],
            "f2": 0,
            "f3": [0, 1, 1]
        }
        ]

        true_batches = [{
            "b": [1],
            "f1": [[3, 4, 5, 6]],
            "f2": [3],
            "f3": [[1, 1]]
        }, {
            "b": [1],
            "f1": [[1, 1, 2]],
            "f2": [0],
            "f3": [[0, 1, 1]]
        }]

        batches = get_bucketed_batches(samples, 1, self.get_padding, self.get_bucket)

        self.assertEqual(true_batches, batches)

    def test_parted_several_batches(self):
        samples = [{
            "b": 1,
            "f1": [3, 4, 5, 6],
            "f2": 3,
            "f3": [1, 1]
        }, {
            "b": 2,
            "f1": [1, 1, 2],
            "f2": 0,
            "f3": [0, 1, 1]
        }, {
            "b": 1,
            "f1": [6, 5],
            "f2": 7,
            "f3": [0]
        }
        ]

        true_batches = [{
            "b": [1, 1],
            "f1": [[3, 4, 5, 6], [6, 5]],
            "f2": [3, 7],
            "f3": [[1, 1], [0]]
        }, {
            "b": [2],
            "f1": [[1, 1, 2]],
            "f2": [0],
            "f3": [[0, 1, 1]]
        }
        ]

        batches = get_bucketed_batches(samples, 5, self.get_padding, self.get_bucket)

        self.assertEqual(true_batches, batches)

    def test_full_batch_with_padding(self):
        samples = [{
            "b": 1,
            "f1": [3, 4, 5, 6],
            "f2": 3,
            "f3": [1, 1],
            "f4": [[2, 1], [2, 3]]
        }, {
            "b": 1,
            "f1": [1, 1, 2],
            "f2": 0,
            "f3": [0, 1, 1],
            "f4": [[2, 1], [2, 3], [4, 5]]
        }
        ]

        padding_values = {
            "b": (None, 0),
            "f1": (0, 1),
            "f2": (None, 0),
            "f3": (-1, 1),
            "f4": (42, 2)
        }

        true_batches = [{
            "b": [1, 1],
            "f1": [[3, 4, 5, 6], [1, 1, 2, 0]],
            "f2": [3, 0],
            "f3": [[1, 1, -1], [0, 1, 1]],
            "f4": [[[2, 1], [2, 3], [42, 42]], [[2, 1], [2, 3], [4, 5]]]
        }]

        batches = get_bucketed_batches(samples, 2, padding_values.get, self.get_bucket)

        self.assertEqual(true_batches, batches)


class GetSegmentBatchesTest(unittest.TestCase):
    def setUp(self):
        # in most tests we have no padding
        self.get_padding = lambda k: (None, 0)

    def test_one_example(self):
        samples = [{
            "f1": [3, 4, 5, 6],
            "f2": 3,
            "key": [
                {"f3": [1, 1]}
            ]

        }]

        true_batches = [{
            "f1": [[3, 4, 5, 6]],
            "f2": [3],
            "f3": [[1, 1]]
        }]

        batches = get_segment_batches(samples, "key", 100, self.get_padding)
        self.assertEqual(true_batches, batches)

    def test_3_examples(self):
        samples = [{
            "f1": [3, 4, 5, 6],
            "f2": 3,
            "key": [
                {"f3": 1},
                {"f3": 2},
                {"f3": 3},
            ]

        }]

        true_batches = [{
            "f1": [[3, 4, 5, 6]],
            "f2": [3],
            "f3": [1]
        }, {
            "f1": [[3, 4, 5, 6]],
            "f2": [3],
            "f3": [2]
        }, {
            "f1": [[3, 4, 5, 6]],
            "f2": [3],
            "f3": [3]
        },
        ]

        batches = get_segment_batches(samples, "key", 1, self.get_padding)
        self.assertEqual(true_batches, batches)

    def test_big_batch(self):
        samples = [{
            "f1": [3, 4, 5, 6],
            "f2": 3,
            "key": [
                {"f3": 1},
                {"f3": 2},
                {"f3": 3},
            ]

        }]

        true_batches = [{
            "f1": [[3, 4, 5, 6]],
            "f2": [3],
            "f3": [1, 2, 3]
        }
        ]

        batches = get_segment_batches(samples, "key", 100, self.get_padding)
        self.assertEqual(true_batches, batches)

    def test_2_features(self):
        samples = [{
            "f1": [3, 4, 5, 6],
            "f2": 3,
            "key": [
                {"f3": 1,
                 "f4": [10, 11]
                 },
                {"f3": 2,
                 "f4": [20, 22]
                 },
                {"f3": 3,
                 "f4": [30, 33]
                 },
            ]

        }]

        true_batches = [{
            "f1": [[3, 4, 5, 6]],
            "f2": [3],
            "f3": [1, 2],
            "f4": [[10, 11], [20, 22]],
        }, {
            "f1": [[3, 4, 5, 6]],
            "f2": [3],
            "f3": [3],
            "f4": [[30, 33]],
        },
        ]

        batches = get_segment_batches(samples, "key", 2, self.get_padding)
        self.assertEqual(true_batches, batches)

    def test_2_features_with_padding(self):
        samples = [{
            "f1": [3, 4, 5, 6],
            "f2": 3,
            "key": [
                {"f3": 1,
                 "f4": [10, 11, 12]
                 },
                {"f3": 2,
                 "f4": [20, 22]
                 },
                {"f3": 3,
                 "f4": [30, 33]
                 },
            ]

        }]

        true_batches = [{
            "f1": [[3, 4, 5, 6]],
            "f2": [3],
            "f3": [1, 2],
            "f4": [[10, 11, 12], [20, 22, 10]],
        }, {
            "f1": [[3, 4, 5, 6]],
            "f2": [3],
            "f3": [3],
            "f4": [[30, 33]],
        },
        ]

        padding_values = {
            "f1": (0, 1),
            "f2": (None, 0),
            "f3": (-1, 0),
            "f4": (10, 1)
        }

        batches = get_segment_batches(samples, "key", 2, padding_values.get)
        self.assertEqual(true_batches, batches)


class PadSequencesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.padding_val = 0

    def test_1D(self):
        sequence = [3, 4, 5, 6]
        expected = sequence

        self.assertEqual(expected, pad_sequences(sequence, self.padding_val).tolist())

    def test_2D(self):
        sequence = [[3, 4, 5, 6], [7], [], [10, 0, 0, 1]]
        expected = [[3, 4, 5, 6], [7, 0, 0, 0], [0, 0, 0, 0], [10, 0, 0, 1]]

        self.assertEqual(expected, pad_sequences(sequence, self.padding_val).tolist())

    def test_2D_numpy(self):
        sequence = [np.array([3, 4, 5, 6]), np.array([7]), np.array([]), np.array([10, 0, 0, 1])]
        expected = [[3, 4, 5, 6], [7, 0, 0, 0], [0, 0, 0, 0], [10, 0, 0, 1]]

        self.assertEqual(expected, pad_sequences(sequence, np.int64(0)).tolist())

    def test_3D(self):
        sequence = [
            [[3, 4, 5, 6], [8, 8]],
            [[7]],
            [[], [10, 0, 0, 1]]
        ]
        expected = [
            [[3, 4, 5, 6], [8, 8, 0, 0]],
            [[7, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [10, 0, 0, 1]]
        ]

        self.assertEqual(expected, pad_sequences(sequence, self.padding_val).tolist())

    def test_3D_numpy(self):
        sequence = [
            [np.array([10.1, 15.0]), np.array([0, 1.2])],
            [np.array([7.0, 100.0])],
            [np.array([5.0, 5.0]), np.array([20.2, 20.0]), np.array([100.1, 100.0])]
        ]
        expected = [
            [[10.1, 15], [0, 1.2], [0, 0]],
            [[7, 100], [0, 0], [0, 0]],
            [[5, 5], [20.2, 20], [100.1, 100]]
        ]

        # padding value must have the same type as sequence elements
        self.assertEqual(expected, pad_sequences(sequence, 0.0).tolist())

    def test_3D_numpy_different_shapes(self):
        sequence = [
            [np.array([3, 4, 5, 6]), np.array([8, 8])],
            [np.array([7])],
            [np.array([]), np.array([10, 0, 0, 1])]
        ]
        expected = [
            [[3, 4, 5, 6], [8, 8, 0, 0]],
            [[7, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [10, 0, 0, 1]]
        ]

        # padding value must have the same type as sequence elements
        self.assertEqual(expected, pad_sequences(sequence, np.int64(0)).tolist())

    def test_3D_numpy_bad_padding(self):
        sequence = [
            [np.array([10.1, 15.0]), np.array([0, 1.2])],
            [np.array([7.0, 100.0])],
            [np.array([5.0, 5.0]), np.array([20.2, 20.0]), np.array([100.1, 100.0])]
        ]

        # padding value must have the same type as sequence elements - float int
        self.assertRaises(Exception, pad_sequences, sequence, self.padding_val)


class PadSequences2DTest(unittest.TestCase):
    def setUp(self) -> None:
        self.padding_val = 0

    def test_2D(self):
        sequence = [[3, 4, 5, 6], [7], [], [10, 0, 0, 1]]
        expected = [[3, 4, 5, 6], [7, 0, 0, 0], [0, 0, 0, 0], [10, 0, 0, 1]]

        self.assertEqual(expected, pad_sequences2d(sequence, self.padding_val).tolist())

    def test_2D_numpy(self):
        sequence = [np.array([3, 4, 5, 6]), np.array([7]), np.array([]), np.array([10, 0, 0, 1])]
        expected = [[3, 4, 5, 6], [7, 0, 0, 0], [0, 0, 0, 0], [10, 0, 0, 1]]

        self.assertEqual(expected, pad_sequences2d(sequence, np.int64(0)).tolist())


class PadSequences3DTest(unittest.TestCase):
    def setUp(self) -> None:
        self.padding_val = 0

    def test_3D(self):
        sequence = [
            [[3, 4, 5, 6], [8, 8]],
            [[7]],
            [[], [10, 0, 0, 1]]
        ]
        expected = [
            [[3, 4, 5, 6], [8, 8, 0, 0]],
            [[7, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [10, 0, 0, 1]]
        ]

        self.assertEqual(expected, pad_sequences3d(sequence, self.padding_val).tolist())

    def test_3D_numpy(self):
        sequence = [
            [np.array([10.1, 15.0]), np.array([0, 1.2])],
            [np.array([7.0, 100.0])],
            [np.array([5.0, 5.0]), np.array([20.2, 20.0]), np.array([100.1, 100.0])]
        ]
        expected = [
            [[10.1, 15], [0, 1.2], [0, 0]],
            [[7, 100], [0, 0], [0, 0]],
            [[5, 5], [20.2, 20], [100.1, 100]]
        ]

        self.assertEqual(expected, pad_sequences3d(sequence, 0.0).tolist())

    def test_3D_numpy_different_shapes(self):
        sequence = [
            [np.array([3, 4, 5, 6]), np.array([8, 8])],
            [np.array([7])],
            [np.array([]), np.array([10, 0, 0, 1])]
        ]
        expected = [
            [[3, 4, 5, 6], [8, 8, 0, 0]],
            [[7, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [10, 0, 0, 1]]
        ]

        self.assertEqual(expected, pad_sequences3d(sequence, np.int64(0)).tolist())


if __name__ == "__main__":
    unittest.main()
