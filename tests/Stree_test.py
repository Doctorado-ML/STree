import unittest

from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
import numpy as np
import csv

from trees.Stree import Stree, Snode

class Stree_test(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self._random_state = 1
        self._model_tree = Stree(random_state=self._random_state, use_predictions=True)
        self._model_tree.fit(*self._get_Xy())
        self._model_svm = LinearSVC(random_state=self._random_state, max_iter=self._model_tree._max_iter)
        super(Stree_test, self).__init__(*args, **kwargs)

    def _get_Xy(self):
        X, y = make_classification(n_samples=1500, n_features=3, n_informative=3, 
                                n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2,
                                class_sep=1.5, flip_y=0,weights=[0.5,0.5], random_state=self._random_state)
        return X, y
    
    def test_split_data(self):
        self.assertTrue(True)

    def _check_tree(self, node: Snode):
        if node.is_leaf():
            return
        self._model_svm.fit(node._X, node._y)
        y_prediction = self._model_svm.predict(node._X)
        y_down = node.get_down()._y
        y_up = node.get_up()._y
        # Is a correct partition in terms of cadinality?
        # i.e. The partition algorithm didn't forget any sample
        self.assertEqual(node._y.shape[0], y_down.shape[0] + y_up.shape[0])
        unique_y, count_y = np.unique(node._y, return_counts=True)
        _, count_d = np.unique(y_down, return_counts=True)
        _, count_u = np.unique(y_up, return_counts=True)
        for i in unique_y:
            try:
                number_down = count_d[i]
            except:
                number_down = 0
            try:
                number_up = count_u[i]
            except:
                number_up = 0
            self.assertEqual(count_y[i], number_down + number_up)
        # Is the partition made the same as the prediction?
        # as the node is not a leaf...
        unique_yp, count_yp = np.unique(y_prediction, return_counts=True)
        self.assertEqual(count_yp[1], y_down.shape[0])
        self.assertEqual(count_yp[0], y_up.shape[0])
        self._check_tree(node.get_down())
        self._check_tree(node.get_up())

    def test_build_tree(self):
        """Check if the tree is built the same way as predictions of models
        """
        self._check_tree(self._model_tree._tree)

    def _get_file_data(self, file_name: str) -> tuple:
        """Return X, y from data, y is the last column in array

        Arguments:
            file_name {str} -- the file name

        Returns:
            tuple -- tuple with samples, categories
        """        
        data = np.genfromtxt(file_name, delimiter=',')
        data = np.array(data)
        column_y = data.shape[1] - 1
        fy = data[:, column_y]
        fx = np.delete(data, column_y, axis=1)
        return fx, fy

    def _find_out(self, px: np.array, x_original: np.array, y_original) -> list:
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
        
    def test_subdatasets(self):
        """Check if the subdatasets files have the same predictions as the tree itself
        """
        model = LinearSVC(random_state=self._random_state, max_iter=self._model_tree._max_iter)
        X, y = self._get_Xy()
        model.fit(X, y)
        self._model_tree.save_sub_datasets()
        with open(self._model_tree.get_catalog_name()) as cat_file:
            catalog = csv.reader(cat_file, delimiter=',')
            for row in catalog:
                X, y = self._get_Xy()
                x_file, y_file = self._get_file_data(row[0])
                y_original = np.array(self._find_out(x_file, X, y), dtype=int)
                self.assertTrue(np.array_equal(y_file, y_original))
