# This Python file uses the following encoding: utf-8
'''
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
__version__ = "0.9"
Build an oblique tree classifier based on SVM Trees
Uses LinearSVC
'''

import typing

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
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

    def _linear_function(self, data: np.array, node: Snode) -> np.array:
        coef = node._vector[0, :].reshape(-1, data.shape[1])
        return data.dot(coef.T) + node._interceptor[0]

    def _split_data(self, node: Snode, data: np.ndarray, indices: np.ndarray) -> list:
        if self.__use_predictions:
            yp = node._clf.predict(data)
            down = (yp == 1).reshape(-1, 1)
        else:
            # doesn't work with multiclass as each sample has to do inner product with its own coeficients
            # computes positition of every sample is w.r.t. the hyperplane
            res = self._linear_function(data, node)
            down = res > 0
        up = ~down
        data_down = data[down[:, 0]] if any(down) else None
        indices_down = indices[down[:, 0]] if any(down) else None
        data_up = data[up[:, 0]] if any(up) else None
        indices_up = indices[up[:, 0]] if any(up) else None
        return [data_down, indices_down, data_up, indices_up]

    def fit(self, X: np.ndarray, y: np.ndarray, title: str = 'root') -> 'Stree':
        X, y = check_X_y(X, y.ravel())
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
            return Snode(None, X, y, title + ', <pure>')
        # Train the model
        clf = LinearSVC(max_iter=self._max_iter, C=self._C,
                        random_state=self._random_state)
        clf.fit(X, y)
        tree = Snode(clf, X, y, title)
        X_U, y_u, X_D, y_d = self._split_data(tree, X, y)
        if X_U is None or X_D is None:
            # didn't part anything
            return Snode(clf, X, y, title + ', <cgaf>')
        tree.set_up(self.train(X_U, y_u, title + ' - Up'))
        tree.set_down(self.train(X_D, y_d, title + ' - Down'))
        return tree

    def _predict_values(self, X: np.array) -> np.array:
        def predict_class(xp: np.array, indices: np.array, node: Snode) -> np.array:
            if xp is None:
                return [], []
            if node.is_leaf():
                # set a class for every sample in dataset
                prediction = np.full((xp.shape[0], 1), node._class)
                if self.__proba:
                    prediction_proba = np.full((xp.shape[0], 1), node._belief)
                    #prediction_proba = self._linear_function(xp, node)
                    return np.append(prediction, prediction_proba, axis=1), indices
                else:
                    return prediction, indices
            u, i_u, d, i_d = self._split_data(node, xp, indices)
            k, l = predict_class(d, i_d, node.get_down())
            m, n = predict_class(u, i_u, node.get_up())
            return np.append(k, m), np.append(l, n)
        # sklearn check
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        # setup prediction & make it happen
        indices = np.arange(X.shape[0])
        return predict_class(X, indices, self._tree)
    
    def _reorder_results(self, y: np.array, indices: np.array) -> np.array:
        y_ordered = np.zeros(y.shape, dtype=int if y.ndim == 1 else float)
        indices = indices.astype(int)
        for i, index in enumerate(indices):
            y_ordered[index] = y[i]
        return y_ordered

    def predict(self, X: np.array) -> np.array:
        return self._reorder_results(*self._predict_values(X))

    def predict_proba(self, X: np.array) -> np.array:
        self.__proba = True
        result, indices = self._predict_values(X)
        self.__proba = False
        result = result.reshape(X.shape[0], 2)
        # Sigmoidize distance like in sklearn based on Platt(1999)
        #result[:, 1] = 1 / (1 + np.exp(-result[:, 1]))
        return self._reorder_results(result, indices)

    def score(self, X: np.array, y: np.array) -> float:
        if not self.__trained:
            self.fit(X, y)
        yp = self.predict(X).reshape(y.shape)
        right = (yp == y).astype(int)
        return np.sum(right) / len(y)

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