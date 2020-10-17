import os
import unittest
import warnings

import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.exceptions import ConvergenceWarning

from stree import Stree, Snode
from .utils import load_dataset


class Stree_test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._random_state = 1
        self._kernels = ["linear", "rbf", "poly"]
        super().__init__(*args, **kwargs)

    @classmethod
    def setUp(cls):
        os.environ["TESTING"] = "1"

    def _check_tree(self, node: Snode):
        """Check recursively that the nodes that are not leaves have the
        correct number of labels and its sons have the right number of elements
        in their dataset

        Arguments:
            node {Snode} -- node to check
        """
        if node.is_leaf():
            return
        y_prediction = node._clf.predict(node._X)
        y_down = node.get_down()._y
        y_up = node.get_up()._y
        # Is a correct partition in terms of cadinality?
        # i.e. The partition algorithm didn't forget any sample
        self.assertEqual(node._y.shape[0], y_down.shape[0] + y_up.shape[0])
        unique_y, count_y = np.unique(node._y, return_counts=True)
        _, count_d = np.unique(y_down, return_counts=True)
        _, count_u = np.unique(y_up, return_counts=True)
        #
        for i in unique_y:
            number_down = count_d[i]
            try:
                number_up = count_u[i]
            except IndexError:
                number_up = 0
            self.assertEqual(count_y[i], number_down + number_up)
        # Is the partition made the same as the prediction?
        # as the node is not a leaf...
        _, count_yp = np.unique(y_prediction, return_counts=True)
        self.assertEqual(count_yp[0], y_up.shape[0])
        self.assertEqual(count_yp[1], y_down.shape[0])
        self._check_tree(node.get_down())
        self._check_tree(node.get_up())

    def test_build_tree(self):
        """Check if the tree is built the same way as predictions of models"""
        warnings.filterwarnings("ignore")
        for kernel in self._kernels:
            clf = Stree(kernel=kernel, random_state=self._random_state)
            clf.fit(*load_dataset(self._random_state))
            self._check_tree(clf.tree_)

    def test_single_prediction(self):
        X, y = load_dataset(self._random_state)
        for kernel in self._kernels:
            clf = Stree(kernel=kernel, random_state=self._random_state)
            yp = clf.fit(X, y).predict((X[0, :].reshape(-1, X.shape[1])))
            self.assertEqual(yp[0], y[0])

    def test_multiple_prediction(self):
        # First 27 elements the predictions are the same as the truth
        num = 27
        X, y = load_dataset(self._random_state)
        for kernel in self._kernels:
            clf = Stree(kernel=kernel, random_state=self._random_state)
            yp = clf.fit(X, y).predict(X[:num, :])
            self.assertListEqual(y[:num].tolist(), yp.tolist())

    def test_single_vs_multiple_prediction(self):
        """Check if predicting sample by sample gives the same result as
        predicting all samples at once
        """
        X, y = load_dataset(self._random_state)
        for kernel in self._kernels:
            clf = Stree(kernel=kernel, random_state=self._random_state)
            clf.fit(X, y)
            # Compute prediction line by line
            yp_line = np.array([], dtype=int)
            for xp in X:
                yp_line = np.append(
                    yp_line, clf.predict(xp.reshape(-1, X.shape[1]))
                )
            # Compute prediction at once
            yp_once = clf.predict(X)
            self.assertListEqual(yp_line.tolist(), yp_once.tolist())

    def test_iterator_and_str(self):
        """Check preorder iterator"""
        expected = [
            "root feaures=(0, 1, 2) impurity=0.5000",
            "root - Down feaures=(0, 1, 2) impurity=0.0671",
            "root - Down - Down, <cgaf> - Leaf class=1 belief= 0.975989 "
            "impurity=0.0469 counts=(array([0, 1]), array([ 17, 691]))",
            "root - Down - Up feaures=(0, 1, 2) impurity=0.3967",
            "root - Down - Up - Down, <cgaf> - Leaf class=1 belief= 0.750000 "
            "impurity=0.3750 counts=(array([0, 1]), array([1, 3]))",
            "root - Down - Up - Up, <pure> - Leaf class=0 belief= 1.000000 "
            "impurity=0.0000 counts=(array([0]), array([7]))",
            "root - Up, <cgaf> - Leaf class=0 belief= 0.928297 impurity=0.1331"
            " counts=(array([0, 1]), array([725,  56]))",
        ]
        computed = []
        expected_string = ""
        clf = Stree(kernel="linear", random_state=self._random_state)
        clf.fit(*load_dataset(self._random_state))
        for node in clf:
            computed.append(str(node))
            expected_string += str(node) + "\n"
        self.assertListEqual(expected, computed)
        self.assertEqual(expected_string, str(clf))

    @staticmethod
    def test_is_a_sklearn_classifier():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        from sklearn.utils.estimator_checks import check_estimator

        check_estimator(Stree())

    def test_exception_if_C_is_negative(self):
        tclf = Stree(C=-1)
        with self.assertRaises(ValueError):
            tclf.fit(*load_dataset(self._random_state))

    def test_exception_if_bogus_split_criteria(self):
        tclf = Stree(split_criteria="duck")
        with self.assertRaises(ValueError):
            tclf.fit(*load_dataset(self._random_state))

    def test_check_max_depth_is_positive_or_None(self):
        tcl = Stree()
        self.assertIsNone(tcl.max_depth)
        tcl = Stree(max_depth=1)
        self.assertGreaterEqual(1, tcl.max_depth)
        with self.assertRaises(ValueError):
            tcl = Stree(max_depth=-1)
            tcl.fit(*load_dataset(self._random_state))

    def test_check_max_depth(self):
        depths = (3, 4)
        for depth in depths:
            tcl = Stree(random_state=self._random_state, max_depth=depth)
            tcl.fit(*load_dataset(self._random_state))
            self.assertEqual(depth, tcl.depth_)

    def test_unfitted_tree_is_iterable(self):
        tcl = Stree()
        self.assertEqual(0, len(list(tcl)))

    def test_min_samples_split(self):
        dataset = [[1], [2], [3]], [1, 1, 0]
        tcl_split = Stree(min_samples_split=3).fit(*dataset)
        self.assertIsNotNone(tcl_split.tree_.get_down())
        self.assertIsNotNone(tcl_split.tree_.get_up())
        tcl_nosplit = Stree(min_samples_split=4).fit(*dataset)
        self.assertIsNone(tcl_nosplit.tree_.get_down())
        self.assertIsNone(tcl_nosplit.tree_.get_up())

    def test_simple_muticlass_dataset(self):
        for kernel in self._kernels:
            clf = Stree(
                kernel=kernel,
                split_criteria="max_samples",
                random_state=self._random_state,
            )
            px = [[1, 2], [5, 6], [9, 10]]
            py = [0, 1, 2]
            clf.fit(px, py)
            self.assertEqual(1.0, clf.score(px, py))
            self.assertListEqual(py, clf.predict(px).tolist())
            self.assertListEqual(py, clf.classes_.tolist())

    def test_muticlass_dataset(self):
        datasets = {
            "Synt": load_dataset(random_state=self._random_state, n_classes=3),
            "Iris": load_iris(return_X_y=True),
        }
        outcomes = {
            "Synt": {
                "max_samples linear": 0.9533333333333334,
                "max_samples rbf": 0.836,
                "max_samples poly": 0.9473333333333334,
                "impurity linear": 0.9533333333333334,
                "impurity rbf": 0.836,
                "impurity poly": 0.9473333333333334,
            },
            "Iris": {
                "max_samples linear": 0.98,
                "max_samples rbf": 1.0,
                "max_samples poly": 1.0,
                "impurity linear": 0.98,
                "impurity rbf": 1,
                "impurity poly": 1,
            },
        }
        for name, dataset in datasets.items():
            px, py = dataset
            for criteria in ["max_samples", "impurity"]:
                for kernel in self._kernels:
                    clf = Stree(
                        C=1e4,
                        max_iter=1e4,
                        kernel=kernel,
                        random_state=self._random_state,
                    )
                    clf.fit(px, py)
                    # print(f"{name} {criteria} {kernel}")
                    outcome = outcomes[name][f"{criteria} {kernel}"]
                    self.assertAlmostEqual(outcome, clf.score(px, py))

    def test_max_features(self):
        n_features = 16
        expected_values = [
            ("auto", 4),
            ("log2", 4),
            ("sqrt", 4),
            (0.5, 8),
            (3, 3),
            (None, 16),
        ]
        clf = Stree()
        clf.n_features_ = n_features
        for max_features, expected in expected_values:
            clf.set_params(**dict(max_features=max_features))
            computed = clf._initialize_max_features()
            self.assertEqual(expected, computed)
        # Check bogus max_features
        values = ["duck", -0.1, 0.0]
        for max_features in values:
            clf.set_params(**dict(max_features=max_features))
            with self.assertRaises(ValueError):
                _ = clf._initialize_max_features()

    def test_get_subspaces(self):
        dataset = np.random.random((10, 16))
        y = np.random.randint(0, 2, 10)
        expected_values = [
            ("auto", 4),
            ("log2", 4),
            ("sqrt", 4),
            (0.5, 8),
            (3, 3),
            (None, 16),
        ]
        clf = Stree()
        for max_features, expected in expected_values:
            clf.set_params(**dict(max_features=max_features))
            clf.fit(dataset, y)
            computed, indices = clf.splitter_.get_subspace(
                dataset, y, clf.max_features_
            )
            self.assertListEqual(
                dataset[:, indices].tolist(), computed.tolist()
            )
            self.assertEqual(expected, len(indices))

    def test_bogus_criterion(self):
        clf = Stree(criterion="duck")
        with self.assertRaises(ValueError):
            clf.fit(*load_dataset())

    def test_predict_feature_dimensions(self):
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        clf = Stree()
        clf.fit(X, y)
        with self.assertRaises(ValueError):
            clf.predict(X[:, :3])

    # Tests of score

    def test_score_binary(self):
        X, y = load_dataset(self._random_state)
        accuracies = [
            0.9506666666666667,
            0.9606666666666667,
            0.9433333333333334,
        ]
        for kernel, accuracy_expected in zip(self._kernels, accuracies):
            clf = Stree(
                random_state=self._random_state,
                kernel=kernel,
            )
            clf.fit(X, y)
            accuracy_score = clf.score(X, y)
            yp = clf.predict(X)
            accuracy_computed = np.mean(yp == y)
            self.assertEqual(accuracy_score, accuracy_computed)
            self.assertAlmostEqual(accuracy_expected, accuracy_score)

    def test_score_max_features(self):
        X, y = load_dataset(self._random_state)
        clf = Stree(random_state=self._random_state, max_features=2)
        clf.fit(X, y)
        self.assertAlmostEqual(0.9426666666666667, clf.score(X, y))

    def test_score_multi_class(self):
        warnings.filterwarnings("ignore")
        accuracies = [
            0.7022472,  # Wine    linear impurity
            0.8314607,  # Wine    linear max_samples
            0.4044944,  # Wine    rbf   impurity
            0.4044944,  # Wine    rbf   max_samples
            0.3988764,  # Wine    poly  impurity
            0.7640449,  # Wine    poly  max_samples
            0.6600000,  # Iris    linear impurity
            0.9666667,  # Iris    linear max_samples
            0.3333333,  # Iris    rbf   impurity
            0.9800000,  # Iris    rbf   max_samples
            0.3333333,  # Iris    poly  impurity
            1.0000000,  # Iris    poly  max_samples
            0.7153333,  # Synthetic linear impurity
            0.9313333,  # Synthetic linear max_samples
            0.4806667,  # Synthetic rbf   impurity
            0.8320000,  # Synthetic rbf   max_samples
            0.4786667,  # Synthetic poly  impurity
            0.6340000,  # Synthetic poly  max_samples
        ]
        datasets = [
            ("Wine", load_wine(return_X_y=True)),
            ("Iris", load_iris(return_X_y=True)),
            (
                "Synthetic",
                load_dataset(self._random_state, n_classes=3, n_features=5),
            ),
        ]
        for dataset_name, dataset in datasets:
            X, y = dataset
            for kernel in self._kernels:
                for criteria in [
                    "impurity",
                    "max_samples",
                ]:
                    clf = Stree(
                        C=17,
                        random_state=self._random_state,
                        kernel=kernel,
                        split_criteria=criteria,
                        degree=5,
                        gamma="auto",
                    )
                    clf.fit(X, y)
                    accuracy_score = clf.score(X, y)
                    yp = clf.predict(X)
                    accuracy_computed = np.mean(yp == y)
                    # print(
                    #     "{:.7f},  # {:7} {:5} {}".format(
                    #         accuracy_score, dataset_name, kernel, criteria
                    #     )
                    # )
                    accuracy_expected = accuracies.pop(0)
                    self.assertEqual(accuracy_score, accuracy_computed)
                    self.assertAlmostEqual(accuracy_expected, accuracy_score)

    def test_bogus_splitter_parameter(self):
        clf = Stree(splitter="duck")
        with self.assertRaises(ValueError):
            clf.fit(*load_dataset())

    def test_weights_removing_class(self):
        # This patch solves an stderr message from sklearn svm lib
        # "WARNING: class label x specified in weight is not found"
        X = np.array(
            [
                [0.1, 0.1],
                [0.1, 0.2],
                [0.2, 0.1],
                [5, 6],
                [8, 9],
                [6, 7],
                [0.2, 0.2],
            ]
        )
        y = np.array([0, 0, 0, 1, 1, 1, 0])
        epsilon = 1e-5
        weights = [1, 1, 1, 0, 0, 0, 1]
        weights = np.array(weights, dtype="float64")
        weights_epsilon = [x + epsilon for x in weights]
        weights_no_zero = np.array([1, 1, 1, 0, 0, 2, 1])
        original = weights_no_zero.copy()
        clf = Stree()
        clf.fit(X, y)
        node = clf.train(
            X,
            y,
            weights,
            1,
            "test",
        )
        # if a class is lost with zero weights the patch adds epsilon
        self.assertListEqual(weights.tolist(), weights_epsilon)
        self.assertListEqual(node._sample_weight.tolist(), weights_epsilon)
        # zero weights are ok when they don't erase a class
        _ = clf.train(X, y, weights_no_zero, 1, "test")
        self.assertListEqual(weights_no_zero.tolist(), original.tolist())
