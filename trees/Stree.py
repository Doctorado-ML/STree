'''
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
__version__ = "1.0"
Create a oblique tree classifier based on SVM Trees
Uses LinearSVC
'''

import numpy as np
from sklearn.svm import LinearSVC

from trees.Snode import Snode

class Stree:
    """
    """
    def __init__(self, max_iter: int=1000, random_state: int=0):
        self._max_iter = max_iter
        self._random_state = random_state
        self._outcomes = None
        self._tree = None

    def _split_data(self, clf: LinearSVC, X: np.ndarray, y: np.ndarray) -> list:
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
        return X_up, y_up, X_down, y_down

    def fit(self, X: np.ndarray, y: np.ndarray, title: str = 'root') -> list:
        self._tree = self.train(X, y, title)
        return self
    
    def train(self: Snode, X: np.ndarray, y: np.ndarray, title: str='') -> list:
        if np.unique(y).shape[0] == 1:
            # onlyt 1 class => pure dataset
            return Snode(np.array([]), 0, X, y, title + f', <pure> class={np.unique(y)} items={y.shape[0]}')
        # Train the model
        clf = LinearSVC(max_iter=self._max_iter, random_state=self._random_state)
        clf.fit(X, y)
        tree = Snode(clf.coef_, clf.intercept_, X, y, title)
        #plot_hyperplane(clf, X, y, title)
        X_T, y_t, X_O, y_o = self._split_data(clf, X, y)
        if X_T is None or X_O is None:
            # didn't part anything
            return Snode(clf.coef_, clf.intercept_, X, y, title + f', <couldn\'t go any further> classes={np.unique(y)} items<0>={y[y==0].shape[0]} items<1>={y[y==1].shape[0]}')
        tree.set_up( self.train(X_T, y_t, title + ' - Up'))
        tree.set_down(self.train(X_O, y_o, title + ' - Down'))
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


