import os
import unittest

import numpy as np
from sklearn.svm import LinearSVC

from stree import Splitter
from .utils import load_dataset


class Splitter_test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._random_state = 1
        super().__init__(*args, **kwargs)

    @staticmethod
    def build(
        clf=LinearSVC(),
        min_samples_split=0,
        splitter_type="random",
        criterion="gini",
        criteria="min_distance",
        random_state=None,
    ):
        return Splitter(
            clf=clf,
            min_samples_split=min_samples_split,
            splitter_type=splitter_type,
            criterion=criterion,
            criteria=criteria,
            random_state=random_state,
        )

    @classmethod
    def setUp(cls):
        os.environ["TESTING"] = "1"

    def test_init(self):
        with self.assertRaises(ValueError):
            self.build(criterion="duck")
        with self.assertRaises(ValueError):
            self.build(splitter_type="duck")
        with self.assertRaises(ValueError):
            self.build(criteria="duck")
        with self.assertRaises(ValueError):
            self.build(clf=None)
        for splitter_type in ["best", "random"]:
            for criterion in ["gini", "entropy"]:
                for criteria in [
                    "min_distance",
                    "max_samples",
                    "max_distance",
                ]:
                    tcl = self.build(
                        splitter_type=splitter_type,
                        criterion=criterion,
                        criteria=criteria,
                    )
                    self.assertEqual(splitter_type, tcl._splitter_type)
                    self.assertEqual(criterion, tcl._criterion)
                    self.assertEqual(criteria, tcl._criteria)

    def test_gini(self):
        expected_values = [
            ([0, 1, 1, 1, 1, 1, 0, 0, 0, 1], 0.48),
            ([0, 1, 1, 2, 2, 3, 4, 5, 3, 2, 1, 1], 0.7777777777777778),
            ([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2], 0.520408163265306),
            ([0, 0, 1, 1, 1, 1, 0, 0], 0.5),
            ([0, 0, 1, 1, 2, 2, 3, 3], 0.75),
            ([0, 0, 1, 1, 1, 1, 1, 1], 0.375),
            ([0], 0),
            ([1, 1, 1, 1], 0),
        ]
        for labels, expected in expected_values:
            self.assertAlmostEqual(expected, Splitter._gini(labels))
            tcl = self.build(criterion="gini")
            self.assertAlmostEqual(expected, tcl.criterion_function(labels))

    def test_entropy(self):
        expected_values = [
            ([0, 1, 1, 1, 1, 1, 0, 0, 0, 1], 0.9709505944546686),
            ([0, 1, 1, 2, 2, 3, 4, 5, 3, 2, 1, 1], 0.9111886696810589),
            ([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2], 0.8120406807940999),
            ([0, 0, 1, 1, 1, 1, 0, 0], 1),
            ([0, 0, 1, 1, 2, 2, 3, 3], 1),
            ([0, 0, 1, 1, 1, 1, 1, 1], 0.8112781244591328),
            ([1], 0),
            ([0, 0, 0, 0], 0),
        ]
        for labels, expected in expected_values:
            self.assertAlmostEqual(expected, Splitter._entropy(labels))
            tcl = self.build(criterion="entropy")
            self.assertAlmostEqual(expected, tcl.criterion_function(labels))

    def test_information_gain(self):
        expected_values = [
            (
                [0, 1, 1, 1, 1, 1],
                [0, 0, 0, 1],
                0.16333333333333333,
                0.25642589168200297,
            ),
            (
                [0, 1, 1, 2, 2, 3, 4, 5, 3, 2, 1, 1],
                [5, 3, 2, 1, 1],
                0.007381776239907684,
                -0.03328610916207225,
            ),
            ([], [], 0.0, 0.0),
            ([1], [], 0.0, 0.0),
            ([], [1], 0.0, 0.0),
            ([0, 0, 0, 0], [0, 0], 0.0, 0.0),
            ([], [1, 1, 1, 2], 0.0, 0.0),
            (None, [1, 2, 3], 0.0, 0.0),
            ([1, 2, 3], None, 0.0, 0.0),
        ]
        for yu, yd, expected_gini, expected_entropy in expected_values:
            yu = np.array(yu, dtype=np.int32) if yu is not None else None
            yd = np.array(yd, dtype=np.int32) if yd is not None else None
            if yu is not None and yd is not None:
                complete = np.append(yu, yd)
            elif yd is not None:
                complete = yd
            else:
                complete = yu
            tcl = self.build(criterion="gini")
            computed = tcl.information_gain(complete, yu, yd)
            self.assertAlmostEqual(expected_gini, computed)
            tcl = self.build(criterion="entropy")
            computed = tcl.information_gain(complete, yu, yd)
            self.assertAlmostEqual(expected_entropy, computed)

    def test_max_samples(self):
        tcl = self.build(criteria="max_samples")
        data = np.array(
            [
                [-0.1, 0.2, -0.3],
                [0.7, 0.01, -0.1],
                [0.7, -0.9, 0.5],
                [0.1, 0.2, 0.3],
            ]
        )
        expected = np.array([0.2, 0.01, -0.9, 0.2])
        y = [1, 2, 1, 0]
        computed = tcl._max_samples(data, y)
        self.assertEqual((4,), computed.shape)
        self.assertListEqual(expected.tolist(), computed.tolist())

    def test_min_distance(self):
        tcl = self.build()
        data = np.array(
            [
                [-0.1, 0.2, -0.3],
                [0.7, 0.01, -0.1],
                [0.7, -0.9, 0.5],
                [0.1, 0.2, 0.3],
            ]
        )
        expected = np.array([2, 2, 1, 0])
        computed = tcl._min_distance(data, None)
        self.assertEqual((4,), computed.shape)
        self.assertListEqual(expected.tolist(), computed.tolist())

    def test_max_distance(self):
        tcl = self.build(criteria="max_distance")
        data = np.array(
            [
                [-0.1, 0.2, -0.3],
                [0.7, 0.01, -0.1],
                [0.7, -0.9, 0.5],
                [0.1, 0.2, 0.3],
            ]
        )
        expected = np.array([1, 0, 0, 2])
        computed = tcl._max_distance(data, None)
        self.assertEqual((4,), computed.shape)
        self.assertListEqual(expected.tolist(), computed.tolist())

    def test_splitter_parameter(self):
        expected_values = [
            [1, 2],  # random gini    min_distance
            [0, 2],  # random gini    max_samples
            [1, 3],  # random gini    max_distance
            [1, 2],  # random entropy min_distance
            [1, 2],  # random entropy max_samples
            [0, 2],  # random entropy max_distance
            [1, 2],  # best   gini    min_distance
            [0, 2],  # best   gini    max_samples
            [0, 2],  # best   gini    max_distance
            [0, 1],  # best   entropy min_distance
            [0, 1],  # best   entropy max_samples
            [0, 1],  # best   entropy max_distance
        ]
        X, y = load_dataset(self._random_state, n_features=6, n_classes=3)
        from sklearn.datasets import load_iris

        X, y = load_iris(return_X_y=True)
        rn = 0
        for splitter_type in ["random", "best"]:
            for criterion in ["gini", "entropy"]:
                for criteria in [
                    "min_distance",
                    "max_samples",
                    "max_distance",
                ]:
                    tcl = self.build(
                        splitter_type=splitter_type,
                        criterion=criterion,
                        criteria=criteria,
                        random_state=rn,
                    )
                    rn += 3
                    expected = expected_values.pop(0)
                    dataset, computed = tcl.get_subspace(X, y, max_features=2)
                    # print(
                    #     "{},  # {:7s}{:8s}{:15s}".format(
                    #         list(computed), splitter_type, criterion,
                    #           criteria,
                    #     )
                    # )
                    self.assertListEqual(expected, list(computed))
                    self.assertListEqual(
                        X[:, computed].tolist(), dataset.tolist()
                    )
