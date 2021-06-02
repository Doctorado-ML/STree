import os
import unittest
import numpy as np
from stree import Stree
from stree.Splitter import Snode
from .utils import load_dataset


class Snode_test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._random_state = 1
        self._clf = Stree(
            random_state=self._random_state,
            kernel="liblinear",
            multiclass_strategy="ovr",
        )
        self._clf.fit(*load_dataset(self._random_state))
        super().__init__(*args, **kwargs)

    @classmethod
    def setUp(cls):
        os.environ["TESTING"] = "1"

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
                belief = max_card / (max_card + min_card)
            else:
                belief = 1
            self.assertEqual(belief, node._belief)
            # Check Class
            class_computed = classes[card == max_card]
            self.assertEqual(class_computed, node._class)
            # Check Partition column
            self.assertEqual(node._partition_column, -1)

        check_leave(self._clf.tree_)

    def test_nodes_coefs(self):
        """Check if the nodes of the tree have the right attributes filled"""

        def run_tree(node: Snode):
            if node._belief < 1:
                # only exclude pure leaves
                self.assertIsNotNone(node._clf)
                self.assertIsNotNone(node._clf.coef_)
            if node.is_leaf():
                return
            run_tree(node.get_up())
            run_tree(node.get_down())

        model = Stree(self._random_state)
        model.fit(*load_dataset(self._random_state, 3, 4))
        run_tree(model.tree_)

    def test_make_predictor_on_leaf(self):
        test = Snode(None, [1, 2, 3, 4], [1, 0, 1, 1], [], 0.0, "test")
        test.make_predictor()
        self.assertEqual(1, test._class)
        self.assertEqual(0.75, test._belief)
        self.assertEqual(-1, test._partition_column)

    def test_set_title(self):
        test = Snode(None, [1, 2, 3, 4], [1, 0, 1, 1], [], 0.0, "test")
        self.assertEqual("test", test.get_title())
        test.set_title("another")
        self.assertEqual("another", test.get_title())

    def test_set_classifier(self):
        test = Snode(None, [1, 2, 3, 4], [1, 0, 1, 1], [], 0.0, "test")
        clf = Stree()
        self.assertIsNone(test.get_classifier())
        test.set_classifier(clf)
        self.assertEqual(clf, test.get_classifier())

    def test_set_impurity(self):
        test = Snode(None, [1, 2, 3, 4], [1, 0, 1, 1], [], 0.0, "test")
        self.assertEqual(0.0, test.get_impurity())
        test.set_impurity(54.7)
        self.assertEqual(54.7, test.get_impurity())

    def test_set_features(self):
        test = Snode(None, [1, 2, 3, 4], [1, 0, 1, 1], [0, 1], 0.0, "test")
        self.assertListEqual([0, 1], test.get_features())
        test.set_features([1, 2])
        self.assertListEqual([1, 2], test.get_features())

    def test_make_predictor_on_not_leaf(self):
        test = Snode(None, [1, 2, 3, 4], [1, 0, 1, 1], [], 0.0, "test")
        test.set_up(Snode(None, [1], [1], [], 0.0, "another_test"))
        test.make_predictor()
        self.assertIsNone(test._class)
        self.assertEqual(0, test._belief)
        self.assertEqual(-1, test._partition_column)
        self.assertEqual(-1, test.get_up()._partition_column)

    def test_make_predictor_on_leaf_bogus_data(self):
        test = Snode(None, [1, 2, 3, 4], [], [], 0.0, "test")
        test.make_predictor()
        self.assertIsNone(test._class)
        self.assertEqual(-1, test._partition_column)

    def test_copy_node(self):
        px = [1, 2, 3, 4]
        py = [1]
        test = Snode(Stree(), px, py, [], 0.0, "test")
        computed = Snode.copy(test)
        self.assertListEqual(computed._X, px)
        self.assertListEqual(computed._y, py)
        self.assertEqual("test", computed._title)
        self.assertIsInstance(computed._clf, Stree)
        self.assertEqual(test._partition_column, computed._partition_column)
        self.assertEqual(test._sample_weight, computed._sample_weight)
        self.assertEqual(test._scaler, computed._scaler)
