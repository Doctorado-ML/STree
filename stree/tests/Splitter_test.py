import os
import unittest
import random

import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_wine, load_iris
from stree import Splitter


class Splitter_test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._random_state = 1
        super().__init__(*args, **kwargs)

    @staticmethod
    def build(
        clf=SVC,
        min_samples_split=0,
        splitter_type="random",
        criterion="gini",
        criteria="max_samples",
        random_state=None,
    ):
        return Splitter(
            clf=clf(random_state=random_state, kernel="rbf"),
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
            _ = Splitter(clf=None)
        for splitter_type in ["best", "random"]:
            for criterion in ["gini", "entropy"]:
                for criteria in ["max_samples", "impurity"]:
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
                [-0.1, 0.2, 0.3],
                [-0.1, 0.2, 0.3],
            ]
        )
        expected = data[:, 0]
        y = [1, 2, 1, 0, 0, 0]
        computed = tcl._max_samples(data, y)
        self.assertEqual(0, computed)
        computed_data = data[:, computed]
        self.assertEqual((6,), computed_data.shape)
        self.assertListEqual(expected.tolist(), computed_data.tolist())

    def test_impurity(self):
        tcl = self.build(criteria="impurity")
        data = np.array(
            [
                [-0.1, 0.2, -0.3],
                [0.7, 0.01, -0.1],
                [0.7, -0.9, 0.5],
                [0.1, 0.2, 0.3],
                [-0.1, 0.2, 0.3],
                [-0.1, 0.2, 0.3],
            ]
        )
        expected = data[:, 2]
        y = np.array([1, 2, 1, 0, 0, 0])
        computed = tcl._impurity(data, y)
        self.assertEqual(2, computed)
        computed_data = data[:, computed]
        self.assertEqual((6,), computed_data.shape)
        self.assertListEqual(expected.tolist(), computed_data.tolist())

    def test_best_splitter_few_sets(self):
        X, y = load_iris(return_X_y=True)
        X = np.delete(X, 3, 1)
        tcl = self.build(splitter_type="best", random_state=self._random_state)
        dataset, computed = tcl.get_subspace(X, y, max_features=2)
        self.assertListEqual([0, 2], list(computed))
        self.assertListEqual(X[:, computed].tolist(), dataset.tolist())

    def test_splitter_parameter(self):
        expected_values = [
            [0, 1, 7, 9],  # best   entropy max_samples
            [3, 8, 10, 11],  # best   entropy impurity
            [0, 2, 8, 12],  # best   gini    max_samples
            [1, 2, 5, 12],  # best   gini    impurity
            [1, 2, 5, 10],  # random entropy max_samples
            [4, 8, 9, 12],  # random entropy impurity
            [3, 9, 11, 12],  # random gini    max_samples
            [1, 5, 6, 9],  # random gini    impurity
        ]
        X, y = load_wine(return_X_y=True)
        rn = 0
        for splitter_type in ["best", "random"]:
            for criterion in ["entropy", "gini"]:
                for criteria in [
                    "max_samples",
                    "impurity",
                ]:
                    tcl = self.build(
                        splitter_type=splitter_type,
                        criterion=criterion,
                        criteria=criteria,
                    )
                    expected = expected_values.pop(0)
                    random.seed(rn)
                    rn += 1
                    dataset, computed = tcl.get_subspace(X, y, max_features=4)
                    # print(
                    #     "{},  # {:7s}{:8s}{:15s}".format(
                    #         list(computed),
                    #         splitter_type,
                    #         criterion,
                    #         criteria,
                    #     )
                    # )
                    self.assertListEqual(expected, list(computed))
                    self.assertListEqual(
                        X[:, computed].tolist(), dataset.tolist()
                    )
