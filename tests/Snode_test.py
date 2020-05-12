import unittest

from sklearn.datasets import make_classification
import numpy as np
import csv

from trees.Stree import Stree, Snode


class Snode_test(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self._random_state = 1
        self._model = Stree(random_state=self._random_state,
                            use_predictions=True)
        self._model.fit(*self._get_Xy())
        super(Snode_test, self).__init__(*args, **kwargs)

    def _get_Xy(self):
        X, y = make_classification(n_samples=1500, n_features=3, n_informative=3,
                                   n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2,
                                   class_sep=1.5, flip_y=0, weights=[0.5, 0.5], random_state=self._random_state)
        return X, y

    def test_attributes_in_leaves(self):
        """Check if the attributes in leaves have correct values so they form a predictor
        """
        def check_leave(node: Snode):
            if node.is_leaf():
                # Check Belief
                classes, card = np.unique(node._y, return_counts=True)
                max_card = max(card)
                min_card = min(card)
                try:
                    accuracy = max_card / min_card
                except:
                    accuracy = 0
                self.assertEqual(accuracy, node._belief)
                # Check Class
                class_computed = classes[card == max_card]
                self.assertEqual(class_computed, node._class)
                return
            check_leave(node.get_down())
            check_leave(node.get_up())
        check_leave(self._model._tree)
