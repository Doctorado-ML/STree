'''
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
__version__ = "1.0"
Create a oblique tree classifier based on SVM Trees
Uses LinearSVC
'''

import numpy as np
import typing
from sklearn.svm import LinearSVC

from trees.Snode import Snode

class Stree:
    """
    """
    def __init__(self, max_iter: int=1000, random_state: int=0, use_predictions: bool=False):
        self._max_iter = max_iter
        self._random_state = random_state
        self._outcomes = None
        self._tree = None
        self.__folder = 'data/'
        self.__use_predictions = use_predictions

    def _split_data(self, clf: LinearSVC, X: np.ndarray, y: np.ndarray) -> list:
        if self.__use_predictions:
            yp = clf.predict(X)
            down = (yp == 1).reshape(-1, 1)
        else:
            # doesn't work with multiclass as each sample has to do inner product with its own coeficients
            # computes positition of every sample is w.r.t. the hyperplane
            coef = clf.coef_[0, :].reshape(-1, X.shape[1])
            intercept = clf.intercept_[0]
            res = X.dot(coef.T) + intercept
            down = res > 0
        up = ~down
        X_down = X[down[:, 0]] if any(down) else None
        y_down = y[down[:, 0]] if any(down) else None
        X_up = X[up[:, 0]] if any(up) else None
        y_up = y[up[:, 0]] if any(up) else None
        return [X_up, y_up, X_down, y_down]

    def fit(self, X: np.ndarray, y: np.ndarray, title: str = 'root') -> 'Stree':
        self._tree = self.train(X, y, title)
        return self

    def train(self, X: np.ndarray, y: np.ndarray, title: str = 'root') -> Snode:
        if np.unique(y).shape[0] == 1:
            # only 1 class => pure dataset
            return Snode(np.array([]), 0, X, y, title + f', class={np.unique(y)}, items={y.shape[0]}, rest=0,  <pure> ')
        # Train the model
        clf = LinearSVC(max_iter=self._max_iter, random_state=self._random_state)
        clf.fit(X, y)
        tree = Snode(clf.coef_, clf.intercept_, X, y, title)
        X_U, y_u, X_D, y_d = self._split_data(clf, X, y)
        if X_U is None or X_D is None:
            # didn't part anything
            return Snode(clf.coef_, clf.intercept_, X, y, title + f', classes={np.unique(y)}, items<0>={y[y==0].shape[0]}, items<1>={y[y==1].shape[0]}, <couldn\'t go any further>')
        tree.set_up(self.train(X_U, y_u, title + ' - Up' + str(np.unique(y_u, return_counts=True))))
        tree.set_down(self.train(X_D, y_d, title + ' - Down' + str(np.unique(y_d, return_counts=True))))
        return tree

    def _print_tree(self, tree: Snode):
        print(tree)
        if tree.is_leaf():
            return
        self._print_tree(tree.get_down())
        self._print_tree(tree.get_up())
    
    def show_outcomes(self):
        pointer = self._tree
        self._print_tree(pointer)

    def _save_datasets(self, tree: Snode, catalog: typing.TextIO, number: int):
        """Save the dataset of the node in a csv file

        Arguments:
            tree {Snode} -- node with data to save
            number {int} -- a number to make different file names
        """
        data = np.append(tree._X, tree._y.reshape(-1,1), axis=1)
        name = f"{self.__folder}dataset{number}.csv"
        np.savetxt(name, data, delimiter=",")
        catalog.write(f"{name}, - {str(tree)}\n")
        if tree.is_leaf():
            return
        self._save_datasets(tree.get_down(), catalog, number + 1)
        self._save_datasets(tree.get_up(), catalog, number + 2)

    def get_catalog_name(self):
        return self.__folder + "catalog.txt"

    def save_sub_datasets(self):
        """Save the every dataset stored in the tree to check with manual classifier
        """
        pointer = self._tree
        with open(self.get_catalog_name(), 'w', encoding = 'utf-8') as catalog:
            self._save_datasets(pointer, catalog, 1)


