import os
import unittest
import warnings

import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC

from stree import Stree
from stree.Splitter import Snode
from .utils import load_dataset
from .._version import __version__


class Stree_test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._random_state = 1
        self._kernels = ["liblinear", "linear", "rbf", "poly", "sigmoid"]
        super().__init__(*args, **kwargs)

    @classmethod
    def setUp(cls):
        os.environ["TESTING"] = "1"

    def test_valid_kernels(self):
        X, y = load_dataset()
        for kernel in self._kernels:
            clf = Stree(kernel=kernel, multiclass_strategy="ovr")
            clf.fit(X, y)
            self.assertIsNotNone(clf.tree_)

    def test_bogus_kernel(self):
        kernel = "other"
        X, y = load_dataset()
        clf = Stree(kernel=kernel)
        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def _check_tree(self, node: Snode):
        """Check recursively that the nodes that are not leaves have the
        correct number of labels and its sons have the right number of elements
        in their dataset

        Parameters
        ----------
        node : Snode
            node to check
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
        labels_d, count_d = np.unique(y_down, return_counts=True)
        labels_u, count_u = np.unique(y_up, return_counts=True)
        dict_d = {label: count_d[i] for i, label in enumerate(labels_d)}
        dict_u = {label: count_u[i] for i, label in enumerate(labels_u)}
        #
        for i in unique_y:
            try:
                number_up = dict_u[i]
            except KeyError:
                number_up = 0
            try:
                number_down = dict_d[i]
            except KeyError:
                number_down = 0
            self.assertEqual(count_y[i], number_down + number_up)
        # Is the partition made the same as the prediction?
        # as the node is not a leaf...
        _, count_yp = np.unique(y_prediction, return_counts=True)
        self.assertEqual(count_yp[1], y_up.shape[0])
        self.assertEqual(count_yp[0], y_down.shape[0])
        self._check_tree(node.get_down())
        self._check_tree(node.get_up())

    def test_build_tree(self):
        """Check if the tree is built the same way as predictions of models"""
        warnings.filterwarnings("ignore")
        for kernel in self._kernels:
            clf = Stree(
                kernel="sigmoid",
                multiclass_strategy="ovr" if kernel == "liblinear" else "ovo",
                random_state=self._random_state,
            )
            clf.fit(*load_dataset(self._random_state))
            self._check_tree(clf.tree_)

    def test_single_prediction(self):
        X, y = load_dataset(self._random_state)
        for kernel in self._kernels:
            clf = Stree(
                kernel=kernel,
                multiclass_strategy="ovr" if kernel == "liblinear" else "ovo",
                random_state=self._random_state,
            )
            yp = clf.fit(X, y).predict((X[0, :].reshape(-1, X.shape[1])))
            self.assertEqual(yp[0], y[0])

    def test_multiple_prediction(self):
        # First 27 elements the predictions are the same as the truth
        num = 27
        X, y = load_dataset(self._random_state)
        for kernel in ["liblinear", "linear", "rbf", "poly"]:
            clf = Stree(
                kernel=kernel,
                multiclass_strategy="ovr" if kernel == "liblinear" else "ovo",
                random_state=self._random_state,
            )
            yp = clf.fit(X, y).predict(X[:num, :])
            self.assertListEqual(y[:num].tolist(), yp.tolist())

    def test_single_vs_multiple_prediction(self):
        """Check if predicting sample by sample gives the same result as
        predicting all samples at once
        """
        X, y = load_dataset(self._random_state)
        for kernel in self._kernels:
            clf = Stree(
                kernel=kernel,
                multiclass_strategy="ovr" if kernel == "liblinear" else "ovo",
                random_state=self._random_state,
            )
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
            "root feaures=(0, 1, 2) impurity=1.0000 counts=(array([0, 1]), "
            "array([750, 750]))",
            "root - Down(2), <cgaf> - Leaf class=0 belief= 0.928297 impurity="
            "0.3722 counts=(array([0, 1]), array([725,  56]))",
            "root - Up(2) feaures=(0, 1, 2) impurity=0.2178 counts=(array([0, "
            "1]), array([ 25, 694]))",
            "root - Up(2) - Down(3) feaures=(0, 1, 2) impurity=0.8454 counts="
            "(array([0, 1]), array([8, 3]))",
            "root - Up(2) - Down(3) - Down(4), <pure> - Leaf class=0 belief= "
            "1.000000 impurity=0.0000 counts=(array([0]), array([7]))",
            "root - Up(2) - Down(3) - Up(4), <cgaf> - Leaf class=1 belief= "
            "0.750000 impurity=0.8113 counts=(array([0, 1]), array([1, 3]))",
            "root - Up(2) - Up(3), <cgaf> - Leaf class=1 belief= 0.975989 "
            "impurity=0.1634 counts=(array([0, 1]), array([ 17, 691]))",
        ]
        computed = []
        expected_string = ""
        clf = Stree(
            kernel="liblinear",
            multiclass_strategy="ovr",
            random_state=self._random_state,
        )
        clf.fit(*load_dataset(self._random_state))
        for node in iter(clf):
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
            tcl = Stree(
                kernel="liblinear",
                multiclass_strategy="ovr",
                random_state=self._random_state,
                max_depth=depth,
            )
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
                multiclass_strategy="ovr" if kernel == "liblinear" else "ovo",
                random_state=self._random_state,
            )
            px = [[1, 2], [5, 6], [9, 10]]
            py = [0, 1, 2]
            clf.fit(px, py)
            self.assertEqual(1.0, clf.score(px, py))
            self.assertListEqual(py, clf.predict(px).tolist())
            self.assertListEqual(py, clf.classes_.tolist())

    def test_muticlass_dataset(self):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        datasets = {
            "Synt": load_dataset(random_state=self._random_state, n_classes=3),
            "Iris": load_wine(return_X_y=True),
        }
        outcomes = {
            "Synt": {
                "max_samples liblinear": 0.9493333333333334,
                "max_samples linear": 0.9426666666666667,
                "max_samples rbf": 0.9606666666666667,
                "max_samples poly": 0.9373333333333334,
                "max_samples sigmoid": 0.824,
                "impurity liblinear": 0.9493333333333334,
                "impurity linear": 0.9426666666666667,
                "impurity rbf": 0.9606666666666667,
                "impurity poly": 0.9373333333333334,
                "impurity sigmoid": 0.824,
            },
            "Iris": {
                "max_samples liblinear": 0.9550561797752809,
                "max_samples linear": 1.0,
                "max_samples rbf": 0.6685393258426966,
                "max_samples poly": 0.6853932584269663,
                "max_samples sigmoid": 0.6404494382022472,
                "impurity liblinear": 0.9550561797752809,
                "impurity linear": 1.0,
                "impurity rbf": 0.6685393258426966,
                "impurity poly": 0.6853932584269663,
                "impurity sigmoid": 0.6404494382022472,
            },
        }

        for name, dataset in datasets.items():
            px, py = dataset
            for criteria in ["max_samples", "impurity"]:
                for kernel in self._kernels:
                    clf = Stree(
                        max_iter=1e4,
                        multiclass_strategy="ovr"
                        if kernel == "liblinear"
                        else "ovo",
                        kernel=kernel,
                        random_state=self._random_state,
                    )
                    clf.fit(px, py)
                    outcome = outcomes[name][f"{criteria} {kernel}"]
                    # print(f'"{criteria} {kernel}": {clf.score(px, py)},')
                    self.assertAlmostEqual(
                        outcome,
                        clf.score(px, py),
                        5,
                        f"{name} - {criteria} - {kernel}",
                    )

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

    def test_wrong_max_features(self):
        X, y = load_dataset(n_features=15)
        clf = Stree(max_features=16)
        with self.assertRaises(ValueError):
            clf.fit(X, y)

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
        """Check score for binary classification."""
        X, y = load_dataset(self._random_state)
        accuracies = [
            0.9506666666666667,
            0.9493333333333334,
            0.9606666666666667,
            0.9433333333333334,
            0.9153333333333333,
        ]
        for kernel, accuracy_expected in zip(self._kernels, accuracies):
            clf = Stree(
                random_state=self._random_state,
                multiclass_strategy="ovr" if kernel == "liblinear" else "ovo",
                kernel=kernel,
            )
            clf.fit(X, y)
            accuracy_score = clf.score(X, y)
            yp = clf.predict(X)
            accuracy_computed = np.mean(yp == y)
            self.assertEqual(accuracy_score, accuracy_computed)
            self.assertAlmostEqual(accuracy_expected, accuracy_score)

    def test_score_max_features(self):
        """Check score using max_features."""
        X, y = load_dataset(self._random_state)
        clf = Stree(
            kernel="liblinear",
            multiclass_strategy="ovr",
            random_state=self._random_state,
            max_features=2,
        )
        clf.fit(X, y)
        self.assertAlmostEqual(0.9453333333333334, clf.score(X, y))

    def test_bogus_splitter_parameter(self):
        """Check that bogus splitter parameter raises exception."""
        clf = Stree(splitter="duck")
        with self.assertRaises(ValueError):
            clf.fit(*load_dataset())

    def test_multiclass_classifier_integrity(self):
        """Checks if the multiclass operation is done right"""
        X, y = load_iris(return_X_y=True)
        clf = Stree(
            kernel="liblinear", multiclass_strategy="ovr", random_state=0
        )
        clf.fit(X, y)
        score = clf.score(X, y)
        # Check accuracy of the whole model
        self.assertAlmostEquals(0.98, score, 5)
        svm = LinearSVC(random_state=0)
        svm.fit(X, y)
        self.assertAlmostEquals(0.9666666666666667, svm.score(X, y), 5)
        data = svm.decision_function(X)
        expected = [
            0.4444444444444444,
            0.35777777777777775,
            0.4569777777777778,
        ]
        ty = data.copy()
        ty[data <= 0] = 0
        ty[data > 0] = 1
        ty = ty.astype(int)
        for i in range(3):
            self.assertAlmostEquals(
                expected[i],
                clf.splitter_._gini(ty[:, i]),
            )
        # 1st Branch
        # up has to have 50 samples of class 0
        # down should have 100 [50, 50]
        up = data[:, 2] > 0
        resup = np.unique(y[up], return_counts=True)
        resdn = np.unique(y[~up], return_counts=True)
        self.assertListEqual([1, 2], resup[0].tolist())
        self.assertListEqual([3, 50], resup[1].tolist())
        self.assertListEqual([0, 1], resdn[0].tolist())
        self.assertListEqual([50, 47], resdn[1].tolist())
        # 2nd Branch
        # up  should have 53 samples of classes [1, 2] [3, 50]
        # down shoud have 47 samples of class 1
        node_up = clf.tree_.get_down().get_up()
        node_dn = clf.tree_.get_down().get_down()
        resup = np.unique(node_up._y, return_counts=True)
        resdn = np.unique(node_dn._y, return_counts=True)
        self.assertListEqual([1, 2], resup[0].tolist())
        self.assertListEqual([3, 50], resup[1].tolist())
        self.assertListEqual([1], resdn[0].tolist())
        self.assertListEqual([47], resdn[1].tolist())

    def test_score_multiclass_rbf(self):
        """Test score for multiclass classification with rbf kernel."""
        X, y = load_dataset(
            random_state=self._random_state,
            n_classes=3,
            n_features=5,
            n_samples=500,
        )
        clf = Stree(kernel="rbf", random_state=self._random_state)
        clf2 = Stree(
            kernel="rbf", random_state=self._random_state, normalize=True
        )
        self.assertEqual(0.966, clf.fit(X, y).score(X, y))
        self.assertEqual(0.964, clf2.fit(X, y).score(X, y))
        X, y = load_wine(return_X_y=True)
        self.assertEqual(0.6685393258426966, clf.fit(X, y).score(X, y))
        self.assertEqual(1.0, clf2.fit(X, y).score(X, y))

    def test_score_multiclass_poly(self):
        """Test score for multiclass classification with poly kernel."""
        X, y = load_dataset(
            random_state=self._random_state,
            n_classes=3,
            n_features=5,
            n_samples=500,
        )
        clf = Stree(
            kernel="poly", random_state=self._random_state, C=10, degree=5
        )
        clf2 = Stree(
            kernel="poly",
            random_state=self._random_state,
            normalize=True,
        )
        self.assertEqual(0.946, clf.fit(X, y).score(X, y))
        self.assertEqual(0.972, clf2.fit(X, y).score(X, y))
        X, y = load_wine(return_X_y=True)
        self.assertEqual(0.7808988764044944, clf.fit(X, y).score(X, y))
        self.assertEqual(1.0, clf2.fit(X, y).score(X, y))

    def test_score_multiclass_liblinear(self):
        """Test score for multiclass classification with liblinear kernel."""
        X, y = load_dataset(
            random_state=self._random_state,
            n_classes=3,
            n_features=5,
            n_samples=500,
        )
        clf = Stree(
            kernel="liblinear",
            multiclass_strategy="ovr",
            random_state=self._random_state,
            C=10,
        )
        clf2 = Stree(
            kernel="liblinear",
            multiclass_strategy="ovr",
            random_state=self._random_state,
            normalize=True,
        )
        self.assertEqual(0.968, clf.fit(X, y).score(X, y))
        self.assertEqual(0.97, clf2.fit(X, y).score(X, y))
        X, y = load_wine(return_X_y=True)
        self.assertEqual(1.0, clf.fit(X, y).score(X, y))
        self.assertEqual(1.0, clf2.fit(X, y).score(X, y))

    def test_score_multiclass_sigmoid(self):
        """Test score for multiclass classification with sigmoid kernel."""
        X, y = load_dataset(
            random_state=self._random_state,
            n_classes=3,
            n_features=5,
            n_samples=500,
        )
        clf = Stree(kernel="sigmoid", random_state=self._random_state, C=10)
        clf2 = Stree(
            kernel="sigmoid",
            random_state=self._random_state,
            normalize=True,
            C=10,
        )
        self.assertEqual(0.796, clf.fit(X, y).score(X, y))
        self.assertEqual(0.952, clf2.fit(X, y).score(X, y))
        X, y = load_wine(return_X_y=True)
        self.assertEqual(0.6910112359550562, clf.fit(X, y).score(X, y))
        self.assertEqual(0.9662921348314607, clf2.fit(X, y).score(X, y))

    def test_score_multiclass_linear(self):
        """Test score for multiclass classification with linear kernel."""
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        X, y = load_dataset(
            random_state=self._random_state,
            n_classes=3,
            n_features=5,
            n_samples=1500,
        )
        clf = Stree(
            kernel="liblinear",
            multiclass_strategy="ovr",
            random_state=self._random_state,
        )
        self.assertEqual(0.9533333333333334, clf.fit(X, y).score(X, y))
        # Check with context based standardization
        clf2 = Stree(
            kernel="liblinear",
            multiclass_strategy="ovr",
            random_state=self._random_state,
            normalize=True,
        )
        self.assertEqual(0.9526666666666667, clf2.fit(X, y).score(X, y))
        X, y = load_wine(return_X_y=True)
        self.assertEqual(0.9831460674157303, clf.fit(X, y).score(X, y))
        self.assertEqual(1.0, clf2.fit(X, y).score(X, y))

    def test_zero_all_sample_weights(self):
        """Test exception raises when all sample weights are zero."""
        X, y = load_dataset(self._random_state)
        with self.assertRaises(ValueError):
            Stree().fit(X, y, np.zeros(len(y)))

    def test_mask_samples_weighted_zero(self):
        """Check that the weighted zero samples are masked."""
        X = np.array(
            [
                [1, 1],
                [1, 1],
                [1, 1],
                [2, 2],
                [2, 2],
                [2, 2],
                [3, 3],
                [3, 3],
                [3, 3],
            ]
        )
        y = np.array([1, 1, 1, 2, 2, 2, 5, 5, 5])
        yw = np.array([1, 1, 1, 1, 1, 1, 5, 5, 5])
        w = [1, 1, 1, 0, 0, 0, 1, 1, 1]
        model1 = Stree().fit(X, y)
        model2 = Stree().fit(X, y, w)
        predict1 = model1.predict(X)
        predict2 = model2.predict(X)
        self.assertListEqual(y.tolist(), predict1.tolist())
        self.assertListEqual(yw.tolist(), predict2.tolist())
        self.assertEqual(model1.score(X, y), 1)
        self.assertAlmostEqual(model2.score(X, y), 0.66666667)
        self.assertEqual(model2.score(X, y, w), 1)

    def test_depth(self):
        """Check depth of the tree."""
        X, y = load_dataset(
            random_state=self._random_state,
            n_classes=3,
            n_features=5,
            n_samples=1500,
        )
        clf = Stree(random_state=self._random_state)
        clf.fit(X, y)
        self.assertEqual(6, clf.depth_)
        X, y = load_wine(return_X_y=True)
        clf = Stree(random_state=self._random_state)
        clf.fit(X, y)
        self.assertEqual(4, clf.depth_)

    def test_nodes_leaves(self):
        """Check number of nodes and leaves."""
        X, y = load_dataset(
            random_state=self._random_state,
            n_classes=3,
            n_features=5,
            n_samples=1500,
        )
        clf = Stree(random_state=self._random_state)
        clf.fit(X, y)
        nodes, leaves = clf.nodes_leaves()
        self.assertEqual(31, nodes)
        self.assertEqual(16, leaves)
        X, y = load_wine(return_X_y=True)
        clf = Stree(random_state=self._random_state)
        clf.fit(X, y)
        nodes, leaves = clf.nodes_leaves()
        self.assertEqual(11, nodes)
        self.assertEqual(6, leaves)

    def test_nodes_leaves_artificial(self):
        """Check leaves of artificial dataset."""
        n1 = Snode(None, [1, 2, 3, 4], [1, 0, 1, 1], [], 0.0, "test1")
        n2 = Snode(None, [1, 2, 3, 4], [1, 0, 1, 1], [], 0.0, "test2")
        n3 = Snode(None, [1, 2, 3, 4], [1, 0, 1, 1], [], 0.0, "test3")
        n4 = Snode(None, [1, 2, 3, 4], [1, 0, 1, 1], [], 0.0, "test4")
        n5 = Snode(None, [1, 2, 3, 4], [1, 0, 1, 1], [], 0.0, "test5")
        n6 = Snode(None, [1, 2, 3, 4], [1, 0, 1, 1], [], 0.0, "test6")
        n1.set_up(n2)
        n2.set_up(n3)
        n2.set_down(n4)
        n3.set_up(n5)
        n4.set_down(n6)
        clf = Stree(random_state=self._random_state)
        clf.tree_ = n1
        nodes, leaves = clf.nodes_leaves()
        self.assertEqual(6, nodes)
        self.assertEqual(2, leaves)

    def test_bogus_multiclass_strategy(self):
        """Check invalid multiclass strategy."""
        clf = Stree(multiclass_strategy="other")
        X, y = load_wine(return_X_y=True)
        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_multiclass_strategy(self):
        """Check multiclass strategy."""
        X, y = load_wine(return_X_y=True)
        clf_o = Stree(multiclass_strategy="ovo")
        clf_r = Stree(multiclass_strategy="ovr")
        score_o = clf_o.fit(X, y).score(X, y)
        score_r = clf_r.fit(X, y).score(X, y)
        self.assertEqual(1.0, score_o)
        self.assertEqual(0.9269662921348315, score_r)

    def test_incompatible_hyperparameters(self):
        """Check incompatible hyperparameters."""
        X, y = load_wine(return_X_y=True)
        clf = Stree(kernel="liblinear", multiclass_strategy="ovo")
        with self.assertRaises(ValueError):
            clf.fit(X, y)
        clf = Stree(multiclass_strategy="ovo", split_criteria="max_samples")
        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_version(self):
        """Check STree version."""
        clf = Stree()
        self.assertEqual(__version__, clf.version())

    def test_graph(self):
        """Check graphviz representation of the tree."""
        X, y = load_wine(return_X_y=True)
        clf = Stree(random_state=self._random_state)

        expected_head = (
            "digraph STree {\nlabel=<STree >\nfontsize=30\n"
            "fontcolor=blue\nlabelloc=t\n"
        )
        expected_tail = (
            ' [shape=box style=filled label="class=1 impurity=0.000 '
            'classes=[1] samples=[1]"];\n}\n'
        )
        self.assertEqual(clf.graph(), expected_head + "}\n")
        clf.fit(X, y)
        computed = clf.graph()
        computed_head = computed[: len(expected_head)]
        num = -len(expected_tail)
        computed_tail = computed[num:]
        self.assertEqual(computed_head, expected_head)
        self.assertEqual(computed_tail, expected_tail)

    def test_graph_title(self):
        X, y = load_wine(return_X_y=True)
        clf = Stree(random_state=self._random_state)
        expected_head = (
            "digraph STree {\nlabel=<STree Sample title>\nfontsize=30\n"
            "fontcolor=blue\nlabelloc=t\n"
        )
        expected_tail = (
            ' [shape=box style=filled label="class=1 impurity=0.000 '
            'classes=[1] samples=[1]"];\n}\n'
        )
        self.assertEqual(clf.graph("Sample title"), expected_head + "}\n")
        clf.fit(X, y)
        computed = clf.graph("Sample title")
        computed_head = computed[: len(expected_head)]
        num = -len(expected_tail)
        computed_tail = computed[num:]
        self.assertEqual(computed_head, expected_head)
        self.assertEqual(computed_tail, expected_tail)
