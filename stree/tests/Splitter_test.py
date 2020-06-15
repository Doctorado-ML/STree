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
                for criteria in ["min_distance", "max_samples"]:
                    tcl = self.build(
                        splitter_type=splitter_type,
                        criterion=criterion,
                        criteria=criteria,
                    )
                    self.assertEqual(splitter_type, tcl._splitter_type)
                    self.assertEqual(criterion, tcl._criterion)
                    self.assertEqual(criteria, tcl._criteria)

    def test_gini(self):
        y = [0, 1, 1, 1, 1, 1, 0, 0, 0, 1]
        expected = 0.48
        self.assertEqual(expected, Splitter._gini(y))
        tcl = self.build(criterion="gini")
        self.assertEqual(expected, tcl.criterion_function(y))

    def test_entropy(self):
        y = [0, 1, 1, 1, 1, 1, 0, 0, 0, 1]
        expected = 0.9709505944546686
        self.assertAlmostEqual(expected, Splitter._entropy(y))
        tcl = self.build(criterion="entropy")
        self.assertEqual(expected, tcl.criterion_function(y))

    def test_information_gain(self):
        yu = np.array([0, 1, 1, 1, 1, 1])
        yd = np.array([0, 0, 0, 1])
        values_expected = [
            ("gini", 0.31666666666666665),
            ("entropy", 0.7145247027726656),
        ]
        for criterion, expected in values_expected:
            tcl = self.build(criterion=criterion)
            computed = tcl.information_gain(yu, yd)
            self.assertAlmostEqual(expected, computed)

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
        expected = np.array([-0.1, 0.01, 0.5, 0.1])
        computed = tcl._min_distance(data, None)
        self.assertEqual((4,), computed.shape)
        self.assertListEqual(expected.tolist(), computed.tolist())

    def test_splitter_parameter(self):
        expected_values = [
            [1, 7, 9],
            [1, 7, 9],
            [1, 7, 9],
            [1, 7, 9],
            [0, 5, 6],
            [0, 5, 6],
            [0, 5, 6],
            [0, 5, 6],
        ]
        X, y = load_dataset(self._random_state, n_features=12)
        for splitter_type in ["best", "random"]:
            for criterion in ["gini", "entropy"]:
                for criteria in ["min_distance", "max_samples"]:
                    tcl = self.build(
                        splitter_type=splitter_type,
                        criterion=criterion,
                        criteria=criteria,
                        random_state=self._random_state,
                    )
                    expected = expected_values.pop(0)
                    dataset, computed = tcl.get_subspace(X, y, max_features=3)
                    self.assertListEqual(expected, list(computed))
                    self.assertListEqual(
                        X[:, computed].tolist(), dataset.tolist()
                    )
