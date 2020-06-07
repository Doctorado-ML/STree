import os
import unittest

import numpy as np
from sklearn.datasets import make_classification

from stree import Stree, Snode


def get_dataset(random_state=0):
    X, y = make_classification(
        n_samples=1500,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        class_sep=1.5,
        flip_y=0,
        weights=[0.5, 0.5],
        random_state=random_state,
    )
    return X, y


class Stree_test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        os.environ["TESTING"] = "1"
        self._random_state = 1
        self._clf = Stree(
            random_state=self._random_state, use_predictions=False
        )
        self._clf.fit(*get_dataset(self._random_state))
        super().__init__(*args, **kwargs)

    @classmethod
    def tearDownClass(cls):
        try:
            os.environ.pop("TESTING")
        except KeyError:
            pass

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
            try:
                number_down = count_d[i]
            except IndexError:
                number_down = 0
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
        """Check if the tree is built the same way as predictions of models
        """
        self._check_tree(self._clf.tree_)

    def _find_out(
        self, px: np.array, x_original: np.array, y_original
    ) -> list:
        """Find the original values of y for a given array of samples

        Arguments:
            px {np.array} -- array of samples to search for
            x_original {np.array} -- original dataset
            y_original {[type]} -- original classes

        Returns:
            np.array -- classes of the given samples
        """
        res = []
        for needle in px:
            for row in range(x_original.shape[0]):
                if all(x_original[row, :] == needle):
                    res.append(y_original[row])
        return res

    def test_single_prediction(self):
        X, y = get_dataset(self._random_state)
        yp = self._clf.predict((X[0, :].reshape(-1, X.shape[1])))
        self.assertEqual(yp[0], y[0])

    def test_multiple_prediction(self):
        # First 27 elements the predictions are the same as the truth
        num = 27
        X, y = get_dataset(self._random_state)
        yp = self._clf.predict(X[:num, :])
        self.assertListEqual(y[:num].tolist(), yp.tolist())

    def test_score(self):
        X, y = get_dataset(self._random_state)
        for kernel in ["linear"]:
            clf = Stree(
                random_state=self._random_state,
                kernel=kernel,
                use_predictions=True,
            )
            clf.fit(X, y)
            accuracy_score = clf.score(X, y)
            yp = clf.predict(X)
            accuracy_computed = np.mean(yp == y)
            self.assertEqual(accuracy_score, accuracy_computed)
            self.assertGreater(accuracy_score, 0.9)

    def test_single_predict_proba(self):
        """Check that element 28 has a prediction different that the current
        label
        """
        # Element 28 has a different prediction than the truth
        decimals = 5
        prob = 0.29026400766
        X, y = get_dataset(self._random_state)
        yp = self._clf.predict_proba(X[28, :].reshape(-1, X.shape[1]))
        self.assertEqual(
            np.round(1 - prob, decimals), np.round(yp[0:, 0], decimals)
        )
        self.assertEqual(1, y[28])

        self.assertAlmostEqual(
            round(prob, decimals), round(yp[0, 1], decimals), decimals
        )

    def test_multiple_predict_proba(self):
        # First 27 elements the predictions are the same as the truth
        num = 27
        decimals = 5
        X, y = get_dataset(self._random_state)
        yp = self._clf.predict_proba(X[:num, :])
        self.assertListEqual(
            y[:num].tolist(), np.argmax(yp[:num], axis=1).tolist()
        )
        expected_proba = [
            0.88395641,
            0.36746962,
            0.84158767,
            0.34106833,
            0.14269291,
            0.85193236,
            0.29876058,
            0.7282164,
            0.85958616,
            0.89517877,
            0.99745224,
            0.18860349,
            0.30756427,
            0.8318412,
            0.18981198,
            0.15564624,
            0.25740655,
            0.22923355,
            0.87365959,
            0.49928689,
            0.95574351,
            0.28761257,
            0.28906333,
            0.32643692,
            0.29788483,
            0.01657364,
            0.81149083,
        ]
        expected = np.round(expected_proba, decimals=decimals).tolist()
        computed = np.round(yp[:, 1], decimals=decimals).tolist()
        for i in range(len(expected)):
            self.assertAlmostEqual(expected[i], computed[i], decimals)

    def build_models(self):
        """Build and train two models, model_clf will use the sklearn
        classifier to compute predictions and split data. model_computed will
        use vector of coefficients to compute both predictions and splitted
        data
        """
        model_clf = Stree(
            random_state=self._random_state, use_predictions=True
        )
        model_computed = Stree(
            random_state=self._random_state, use_predictions=False
        )
        X, y = get_dataset(self._random_state)
        model_clf.fit(X, y)
        model_computed.fit(X, y)
        return model_clf, model_computed, X, y

    def test_use_model_predict(self):
        """Check that we get the same results wether we use the estimator in
        nodes to compute labels or we use the hyperplane and the position of
        samples wrt to it
        """
        use_clf, use_math, X, _ = self.build_models()
        self.assertListEqual(
            use_clf.predict(X).tolist(), use_math.predict(X).tolist()
        )

    def test_use_model_score(self):
        use_clf, use_math, X, y = self.build_models()
        b = use_math.score(X, y)
        self.assertEqual(use_clf.score(X, y), b)
        self.assertGreater(b, 0.95)

    def test_use_model_predict_proba(self):
        use_clf, use_math, X, _ = self.build_models()
        self.assertListEqual(
            use_clf.predict_proba(X).tolist(),
            use_math.predict_proba(X).tolist(),
        )

    def test_single_vs_multiple_prediction(self):
        """Check if predicting sample by sample gives the same result as
        predicting all samples at once
        """
        X, _ = get_dataset(self._random_state)
        # Compute prediction line by line
        yp_line = np.array([], dtype=int)
        for xp in X:
            yp_line = np.append(
                yp_line, self._clf.predict(xp.reshape(-1, X.shape[1]))
            )
        # Compute prediction at once
        yp_once = self._clf.predict(X)
        #
        self.assertListEqual(yp_line.tolist(), yp_once.tolist())

    def test_iterator_and_str(self):
        """Check preorder iterator
        """
        expected = [
            "root",
            "root - Down",
            "root - Down - Down, <cgaf> - Leaf class=1 belief= 0.975989 counts"
            "=(array([0, 1]), array([ 17, 691]))",
            "root - Down - Up",
            "root - Down - Up - Down, <cgaf> - Leaf class=1 belief= 0.750000 "
            "counts=(array([0, 1]), array([1, 3]))",
            "root - Down - Up - Up, <pure> - Leaf class=0 belief= 1.000000 "
            "counts=(array([0]), array([7]))",
            "root - Up, <cgaf> - Leaf class=0 belief= 0.928297 counts=(array("
            "[0, 1]), array([725,  56]))",
        ]
        computed = []
        expected_string = ""
        for node in self._clf:
            computed.append(str(node))
            expected_string += str(node) + "\n"
        self.assertListEqual(expected, computed)
        self.assertEqual(expected_string, str(self._clf))

    def test_is_a_sklearn_classifier(self):
        import warnings
        from sklearn.exceptions import ConvergenceWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        from sklearn.utils.estimator_checks import check_estimator

        check_estimator(Stree())

    def test_exception_if_C_is_negative(self):
        tclf = Stree(C=-1)
        with self.assertRaises(ValueError):
            tclf.fit(*get_dataset(self._random_state))

    def test_check_max_depth_is_positive_or_None(self):
        tcl = Stree()
        self.assertIsNone(tcl.max_depth)
        tcl = Stree(max_depth=1)
        self.assertGreaterEqual(1, tcl.max_depth)
        with self.assertRaises(ValueError):
            tcl = Stree(max_depth=-1)
            tcl.fit(*get_dataset(self._random_state))

    def test_check_max_depth(self):
        depths = (3, 4)
        for depth in depths:
            tcl = Stree(random_state=self._random_state, max_depth=depth)
            tcl.fit(*get_dataset(self._random_state))
            self.assertEqual(depth, tcl.depth_)

    def test_unfitted_tree_is_iterable(self):
        tcl = Stree()
        self.assertEqual(0, len(list(tcl)))

    def test_min_samples_split(self):
        tcl_split = Stree(min_samples_split=3)
        tcl_nosplit = Stree(min_samples_split=4)
        dataset = [[1], [2], [3]], [1, 1, 0]
        tcl_split.fit(*dataset)
        self.assertIsNotNone(tcl_split.tree_.get_down())
        self.assertIsNotNone(tcl_split.tree_.get_up())
        tcl_nosplit.fit(*dataset)
        self.assertIsNone(tcl_nosplit.tree_.get_down())
        self.assertIsNone(tcl_nosplit.tree_.get_up())


class Snode_test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        os.environ["TESTING"] = "1"
        self._random_state = 1
        self._clf = Stree(
            random_state=self._random_state, use_predictions=True
        )
        self._clf.fit(*get_dataset(self._random_state))
        super().__init__(*args, **kwargs)

    @classmethod
    def tearDownClass(cls):
        """[summary]
        """
        try:
            os.environ.pop("TESTING")
        except KeyError:
            pass

    def test_attributes_in_leaves(self):
        """Check if the attributes in leaves have correct values so they form a
        predictor
        """

        def check_leave(node: Snode):
            if not node.is_leaf():
                check_leave(node.get_down())
                check_leave(node.get_up())
                return
            # Check Belief in leave
            classes, card = np.unique(node._y, return_counts=True)
            max_card = max(card)
            min_card = min(card)
            if len(classes) > 1:
                try:
                    belief = max_card / (max_card + min_card)
                except ZeroDivisionError:
                    belief = 0.0
            else:
                belief = 1
            self.assertEqual(belief, node._belief)
            # Check Class
            class_computed = classes[card == max_card]
            self.assertEqual(class_computed, node._class)

        check_leave(self._clf.tree_)

    def test_nodes_coefs(self):
        """Check if the nodes of the tree have the right attributes filled
        """

        def run_tree(node: Snode):
            if node._belief < 1:
                # only exclude pure leaves
                self.assertIsNotNone(node._clf)
                self.assertIsNotNone(node._clf.coef_)
            if node.is_leaf():
                return
            run_tree(node.get_down())
            run_tree(node.get_up())

        run_tree(self._clf.tree_)

    def test_make_predictor_on_leaf(self):
        test = Snode(None, [1, 2, 3, 4], [1, 0, 1, 1], "test")
        test.make_predictor()
        self.assertEqual(1, test._class)
        self.assertEqual(0.75, test._belief)

    def test_make_predictor_on_not_leaf(self):
        test = Snode(None, [1, 2, 3, 4], [1, 0, 1, 1], "test")
        test.set_up(Snode(None, [1], [1], "another_test"))
        test.make_predictor()
        self.assertIsNone(test._class)
        self.assertEqual(0, test._belief)
