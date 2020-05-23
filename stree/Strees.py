'''
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
__version__ = "0.9"
Build an oblique tree classifier based on SVM Trees
Uses LinearSVC
'''

import typing
import os

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class Snode:
    def __init__(self, clf: LinearSVC, X: np.ndarray, y: np.ndarray, title: str):
        self._clf = clf
        self._vector = None if clf is None else clf.coef_
        self._interceptor = 0. if clf is None else clf.intercept_
        self._title = title
        self._belief = 0.  # belief of the prediction in a leaf node based on samples
        # Only store dataset in Testing 
        self._X = X if os.environ.get('TESTING', 'NS') != 'NS' else None
        self._y = y
        self._down = None
        self._up = None
        self._class = None

    @classmethod
    def copy(cls, node: 'Snode') -> 'Snode':
        return cls(node._clf, node._X, node._y, node._title)

    def set_down(self, son):
        self._down = son

    def set_up(self, son):
        self._up = son

    def is_leaf(self) -> bool:
        return self._up is None and self._down is None

    def get_down(self) -> 'Snode':
        return self._down

    def get_up(self) -> 'Snode':
        return self._up

    def make_predictor(self):
        """Compute the class of the predictor and its belief based on the subdataset of the node
        only if it is a leaf
        """
        if not self.is_leaf():
            return
        classes, card = np.unique(self._y, return_counts=True)
        if len(classes) > 1:
            max_card = max(card)
            min_card = min(card)
            try:
                self._belief = max_card / (max_card + min_card)
            except:
                self._belief = 0.
            self._class = classes[card == max_card][0]
        else:
            self._belief = 1
            self._class = classes[0]

    def __str__(self) -> str:
        if self.is_leaf():
            return f"{self._title} - Leaf class={self._class} belief={self._belief:.6f} counts={np.unique(self._y, return_counts=True)}"
        else:
            return f"{self._title}"


class Siterator:
    """Stree preorder iterator
    """

    def __init__(self, tree: Snode):
        self._stack = []
        self._push(tree)

    def __iter__(self):
        return self

    def _push(self, node: Snode):
        if node is not None:
            self._stack.append(node)

    def __next__(self) -> Snode:
        if len(self._stack) == 0:
            raise StopIteration()
        node = self._stack.pop()
        self._push(node.get_up())
        self._push(node.get_down())
        return node

class Stree(BaseEstimator, ClassifierMixin):
    """
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 0, use_predictions: bool = False):
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
            res = np.expand_dims(node._clf.decision_function(data), 1)
        else:
            # doesn't work with multiclass as each sample has to do inner product with its own coeficients
            # computes positition of every sample is w.r.t. the hyperplane
            res = self._linear_function(data, node)
            down = res > 0
        up = ~down
        data_down = data[down[:, 0]] if any(down) else None
        indices_down = indices[down[:, 0]] if any(down) else None
        res_down = res[down[:, 0]] if any(down) else None
        data_up = data[up[:, 0]] if any(up) else None
        indices_up = indices[up[:, 0]] if any(up) else None
        res_up = res[up[:, 0]] if any(up) else None
        return [data_up, indices_up, data_down, indices_down, res_up, res_down]

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
        X_U, y_u, X_D, y_d, _, _ = self._split_data(tree, X, y)
        if X_U is None or X_D is None:
            # didn't part anything
            return Snode(clf, X, y, title + ', <cgaf>')
        tree.set_up(self.train(X_U, y_u, title + ' - Up'))
        tree.set_down(self.train(X_D, y_d, title + ' - Down'))
        return tree

    def _reorder_results(self, y: np.array, indices: np.array) -> np.array:
        y_ordered = np.zeros(y.shape, dtype=int if y.ndim == 1 else float)
        indices = indices.astype(int)
        for i, index in enumerate(indices):
            y_ordered[index] = y[i]
        return y_ordered

    def predict(self, X: np.array) -> np.array:
        def predict_class(xp: np.array, indices: np.array, node: Snode) -> np.array:
            if xp is None:
                return [], []
            if node.is_leaf():
                # set a class for every sample in dataset
                prediction = np.full((xp.shape[0], 1), node._class)
                return prediction, indices
            u, i_u, d, i_d, _, _ = self._split_data(node, xp, indices)
            k, l = predict_class(d, i_d, node.get_down())
            m, n = predict_class(u, i_u, node.get_up())
            return np.append(k, m), np.append(l, n)

        # sklearn check
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        # setup prediction & make it happen
        indices = np.arange(X.shape[0])
        return self._reorder_results(*predict_class(X, indices, self._tree))

    def predict_proba(self, X: np.array) -> np.array:
        """Computes an approximation of the probability of samples belonging to class 1 
        (nothing more, nothing less)

        :param X: dataset
        :type X: np.array
        """

        def predict_class(xp: np.array, indices: np.array, dist: np.array, node: Snode) -> np.array:
            """Run the tree to compute predictions

            :param xp: subdataset of samples
            :type xp: np.array
            :param indices: indices of subdataset samples to rebuild original order
            :type indices: np.array
            :param dist: distances of every sample to the hyperplane or the father node
            :type dist: np.array
            :param node: node of the leaf with the class
            :type node: Snode
            :return: array of labels and distances, array of indices
            :rtype: np.array
            """
            if xp is None:
                return [], []
            if node.is_leaf():
                # set a class for every sample in dataset
                prediction = np.full((xp.shape[0], 1), node._class)
                prediction_proba = dist
                return np.append(prediction, prediction_proba, axis=1), indices
            u, i_u, d, i_d, r_u, r_d = self._split_data(node, xp, indices)
            k, l = predict_class(d, i_d, r_d, node.get_down())
            m, n = predict_class(u, i_u, r_u, node.get_up())
            return np.append(k, m), np.append(l, n)

        # sklearn check
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        # setup prediction & make it happen
        indices = np.arange(X.shape[0])
        result, indices = predict_class(X, indices, [], self._tree)
        result = result.reshape(X.shape[0], 2)
        # Turn distances to hyperplane into probabilities based on fitting distances
        # of samples to its hyperplane that classified them, to the sigmoid function
        result[:, 1] = 1 / (1 + np.exp(-result[:, 1]))
        return self._reorder_results(result, indices)

    def score(self, X: np.array, y: np.array) -> float:
        """Return accuracy
        """
        if not self.__trained:
            self.fit(X, y)
        yp = self.predict(X).reshape(y.shape)
        right = (yp == y).astype(int)
        return np.sum(right) / len(y)

    def __iter__(self):
        return Siterator(self._tree)

    def __str__(self) -> str:
        output = ''
        for i in self:
            output += str(i) + '\n'
        return output

    def _save_datasets(self, tree: Snode, catalog: typing.TextIO, number: int):
        """Save the dataset of the node in a csv file

        :param tree: node with data to save
        :type tree: Snode
        :param catalog: catalog file handler
        :type catalog: typing.TextIO
        :param number: sequential number for the generated file name
        :type number: int
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
        if not os.path.isdir(self.__folder):
            os.mkdir(self.__folder)
        with open(self.get_catalog_name(), 'w', encoding='utf-8') as catalog:
            self._save_datasets(self._tree, catalog, 1)



