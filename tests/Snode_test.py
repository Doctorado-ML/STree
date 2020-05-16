import unittest

from sklearn.datasets import make_classification
import os
import numpy as np
import csv

from trees.Stree import Stree, Snode


class Snode_test(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        os.environ['TESTING'] = '1'
        self._random_state = 1
        self._clf = Stree(random_state=self._random_state,
                            use_predictions=True)
        self._clf.fit(*self._get_Xy())
        super(Snode_test, self).__init__(*args, **kwargs)

    @classmethod
    def tearDownClass(cls):
        try:
            os.environ.pop('TESTING')
        except:
            pass

    def _get_Xy(self):
        X, y = make_classification(n_samples=1500, n_features=3, n_informative=3,
                                   n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2,
                                   class_sep=1.5, flip_y=0, weights=[0.5, 0.5], random_state=self._random_state)
        return X, y

    def test_attributes_in_leaves(self):
        """Check if the attributes in leaves have correct values so they form a predictor
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
                except:
                    belief = 0.
            else:
                belief = 1
            self.assertEqual(belief, node._belief)
            # Check Class
            class_computed = classes[card == max_card]
            self.assertEqual(class_computed, node._class)
        check_leave(self._clf._tree)
    
    def test_nodes_coefs(self):
        """Check if the nodes of the tree have the right attributes filled
        """
        def run_tree(node: Snode):
            if node._belief < 1:
                # only exclude pure leaves
                self.assertIsNotNone(node._clf)
                self.assertIsNotNone(node._clf.coef_)
                self.assertIsNotNone(node._vector)
                self.assertIsNotNone(node._interceptor)
            if node.is_leaf():
                return
            run_tree(node.get_down())
            run_tree(node.get_up())
        run_tree(self._clf._tree)
