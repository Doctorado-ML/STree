import os
import unittest

import numpy as np

from stree import Stree, Snode
from .utils import get_dataset


class Snode_test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._random_state = 1
        self._clf = Stree(random_state=self._random_state)
        self._clf.fit(*get_dataset(self._random_state))
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

    def test_make_predictor_on_leaf_bogus_data(self):
        test = Snode(None, [1, 2, 3, 4], [], "test")
        test.make_predictor()
        self.assertIsNone(test._class)

    def test_copy_node(self):
        px = [1, 2, 3, 4]
        py = [1]
        test = Snode(Stree(), px, py, "test")
        computed = Snode.copy(test)
        self.assertListEqual(computed._X, px)
        self.assertListEqual(computed._y, py)
        self.assertEqual("test", computed._title)
        self.assertIsInstance(computed._clf, Stree)
