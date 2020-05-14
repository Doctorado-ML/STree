# This Python file uses the following encoding: utf-8
'''
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
__version__ = "0.9"
Build an oblique tree classifier based on SVM Trees
Uses LinearSVC
'''

import numpy as np
import typing
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from trees.Snode import Snode


class Stree(BaseEstimator, ClassifierMixin):
    """
    """

    def __init__(self, C=1.0, max_iter: int=1000, random_state: int=0, use_predictions: bool=False):
        self._max_iter = max_iter
        self._C = C
        self._random_state = random_state
        self._tree = None
        self.__folder = 'data/'
        self.__use_predictions = use_predictions
        self.__trained = False
        self.__proba = False

    def get_params(self, deep=True):
        """Get dict with hyperparameters and its values to accomplish sklearn rules
        """
        return {"C": self._C, "random_state": self._random_state, 'max_iter': self._max_iter}

    def set_params(self, **parameters):
        """Set hyperparmeters as specified by sklearn, needed in Gridsearchs
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

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
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self._tree = self.train(X, y.ravel(), title)
        self._build_predictor()
        self.__trained = True
        return self

    def _build_predictor(self):
        """Process the leaves to make them predictors
        """
        def run_tree(node: Snode):
            if node.is_leaf():
                node.make_predictor()
                return
            run_tree(node.get_down())
            run_tree(node.get_up())
        run_tree(self._tree)

    def train(self, X: np.ndarray, y: np.ndarray, title: str = 'root') -> Snode:
        if np.unique(y).shape[0] == 1:
            # only 1 class => pure dataset
            return Snode(None, X, y, title + ', <pure> ')
        # Train the model
        clf = LinearSVC(max_iter=self._max_iter, C=self._C,
                        random_state=self._random_state)
        clf.fit(X, y)
        tree = Snode(clf, X, y, title)
        X_U, y_u, X_D, y_d = self._split_data(clf, X, y)
        if X_U is None or X_D is None:
            # didn't part anything
            return Snode(clf, X, y, title + ', <couldn\'t go any further>')
        tree.set_up(self.train(X_U, y_u, title + ' - Up'))
        tree.set_down(self.train(X_D, y_d, title + ' - Down'))
        return tree

    def predict(self, X: np.array) -> np.array:
        def predict_class(xp: np.array, tree: Snode) -> np.array:
            if tree.is_leaf():
                if self.__proba:
                    return [tree._class, tree._belief]
                else:
                    return tree._class
            coef = tree._vector[0, :].reshape(-1, xp.shape[1])
            if xp.dot(coef.T) + tree._interceptor[0] > 0:
                return predict_class(xp, tree.get_down())
            return predict_class(xp, tree.get_up())

        # sklearn check
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        # setup prediction & make it happen
        y = np.array([], dtype=int)
        for xp in X:
            y = np.append(y, predict_class(xp.reshape(-1, X.shape[1]), self._tree))
        return y

    def predict_proba(self, X: np.array) -> np.array:
        self.__proba = True
        result = self.predict(X).reshape(X.shape[0], 2)
        self.__proba = False
        return result

    def score(self, X: np.array, y: np.array, print_out=True) -> float:
        if not self.__trained:
            self.fit(X, y)
        yp = self.predict(X).reshape(y.shape)
        right = (yp == y).astype(int)
        accuracy = np.sum(right) / len(y)
        if print_out:
            print(f"Accuracy: {accuracy:.6f}")
        return accuracy

    def __print_tree(self, tree: Snode, only_leaves=False) -> str:
        if not only_leaves:
            output = str(tree)
        else:
            output = ''
        if tree.is_leaf():
            if only_leaves:
                output = str(tree)
            return output
        output += self.__print_tree(tree.get_down(), only_leaves)
        output += self.__print_tree(tree.get_up(), only_leaves)
        return output

    def show_tree(self, only_leaves=False):
        if only_leaves:
            print(self.__print_tree(self._tree, only_leaves=True))
        else:
            print(self)

    def __str__(self):
        return self.__print_tree(self._tree)

    def _save_datasets(self, tree: Snode, catalog: typing.TextIO, number: int):
        """Save the dataset of the node in a csv file

        Arguments:
            tree {Snode} -- node with data to save
            number {int} -- a number to make different file names
        """
        data = np.append(tree._X, tree._y.reshape(-1, 1), axis=1)
        name = f"{self.__folder}dataset{number}.csv"
        np.savetxt(name, data, delimiter=",")
        catalog.write(f"{name}, - {str(tree)}")
        if tree.is_leaf():
            return
        self._save_datasets(tree.get_down(), catalog, number + 1)
        self._save_datasets(tree.get_up(), catalog, number + 2)

    def get_catalog_name(self):
        return self.__folder + "catalog.txt"

    def save_sub_datasets(self):
        """Save the every dataset stored in the tree to check with manual classifier
        """
        with open(self.get_catalog_name(), 'w', encoding='utf-8') as catalog:
            self._save_datasets(self._tree, catalog, 1)
