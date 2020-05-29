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
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, _check_sample_weight, check_random_state


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

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = None,
                max_depth: int=None, tol: float=1e-4, use_predictions: bool = False):
        self.max_iter = max_iter
        self.C = C
        self.random_state = random_state
        self.use_predictions = use_predictions
        self.max_depth = max_depth
        self.tol = tol

    def get_params(self, deep: bool=True) -> dict:
        """Get dict with hyperparameters and its values to accomplish sklearn rules
        """
        return {
            'C': self.C,
            'random_state': self.random_state,
            'max_iter': self.max_iter,
            'use_predictions': self.use_predictions,
            'max_depth': self.max_depth,
            'tol': self.tol
        }

    def set_params(self, **parameters: dict):
        """Set hyperparmeters as specified by sklearn, needed in Gridsearchs
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    # Added binary_only tag as required by sklearn check_estimator
    def _more_tags(self) -> dict:
        return {'binary_only': True}

    def _linear_function(self, data: np.array, node: Snode) -> np.array:
        coef = node._vector[0, :].reshape(-1, data.shape[1])
        return data.dot(coef.T) + node._interceptor[0]

    def _split_array(self, origin: np.array, down: np.array) -> list:
        up = ~down
        return origin[up[:, 0]] if any(up) else None, \
            origin[down[:, 0]] if any(down) else None

    def _distances(self, node: Snode, data: np.ndarray) -> np.array:
        if self.use_predictions:
            res = np.expand_dims(node._clf.decision_function(data), 1)
        else:
            # doesn't work with multiclass as each sample has to do inner product with its own coeficients
            # computes positition of every sample is w.r.t. the hyperplane
            res = self._linear_function(data, node)
        # data_up, data_down = self._split_array(data, down)
        # indices_up, indices_down = self._split_array(indices, down)
        # res_up, res_down = self._split_array(res, down)
        # weight_up, weight_down = self._split_array(weights, down)
        #return [data_up, indices_up, data_down, indices_down, weight_up, weight_down, res_up, res_down]
        return res

    def _split_criteria(self, data: np.array) -> np.array:
        return data > 0

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.array = None) -> 'Stree':
        # Check parameters are Ok.
        if type(y).__name__ == 'np.ndarray':
            y = y.ravel()
        if self.C < 0:
            raise ValueError(f"Penalty term must be positive... got (C={self.C:f})")
        self.__max_depth = np.iinfo(np.int32).max if self.max_depth is None else self.max_depth
        if self.__max_depth < 1:
            raise ValueError(f"Maximum depth has to be greater than 1... got (max_depth={self.max_depth})")
        check_classification_targets(y)
        X, y = check_X_y(X, y)
        sample_weight = _check_sample_weight(sample_weight, X)
        check_classification_targets(y)
        # Initialize computed parameters
        #self.random_state = check_random_state(self.random_state)
        self.classes_ = np.unique(y)
        self.n_iter_ = self.max_iter
        self.depth_ = 0
        self.n_features_in_ = X.shape[1]
        self.tree_ = self.train(X, y, sample_weight, 1, 'root')
        self._build_predictor()
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

        run_tree(self.tree_)

    def train(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, depth: int, title: str) -> Snode:
        
        if depth > self.__max_depth:
            return None
        if np.unique(y).shape[0] == 1 :
            # only 1 class => pure dataset
            return Snode(None, X, y, title + ', <pure>')
        # Train the model
        clf = LinearSVC(max_iter=self.max_iter, random_state=self.random_state,
                        C=self.C)  #, sample_weight=sample_weight)
        clf.fit(X, y, sample_weight=sample_weight)
        tree = Snode(clf, X, y, title)
        self.depth_ = max(depth, self.depth_)
        down = self._split_criteria(self._distances(tree, X))
        X_U, X_D = self._split_array(X, down)
        y_u, y_d = self._split_array(y, down)
        sw_u, sw_d = self._split_array(sample_weight, down)
        if X_U is None or X_D is None:
            # didn't part anything
            return Snode(clf, X, y, title + ', <cgaf>')
        tree.set_up(self.train(X_U, y_u, sw_u, depth + 1, title + ' - Up'))
        tree.set_down(self.train(X_D, y_d, sw_d, depth + 1, title + ' - Down'))
        return tree

    def _reorder_results(self, y: np.array, indices: np.array) -> np.array:
        if y.ndim > 1 and y.shape[1] > 1:
            # if predict_proba return np.array of floats
            y_ordered = np.zeros(y.shape, dtype=float)
        else:
            # return array of same type given in y
            y_ordered = y.copy()
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
            down = self._split_criteria(self._distances(node, xp))
            X_U, X_D = self._split_array(xp, down)
            i_u, i_d = self._split_array(indices, down)
            prx_u, prin_u = predict_class(X_U, i_u, node.get_up())
            prx_d, prin_d = predict_class(X_D, i_d, node.get_down())
            return np.append(prx_u, prx_d), np.append(prin_u, prin_d)

        # sklearn check
        check_is_fitted(self, ['tree_'])
        # Input validation
        X = check_array(X)
        # setup prediction & make it happen
        indices = np.arange(X.shape[0])
        return self._reorder_results(*predict_class(X, indices, self.tree_)).ravel()

    def predict_proba(self, X: np.array) -> np.array:
        """Computes an approximation of the probability of samples belonging to class 0 and 1
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
            distances = self._distances(node, xp)
            down = self._split_criteria(distances)
            
            X_U, X_D = self._split_array(xp, down)
            i_u, i_d = self._split_array(indices, down)
            di_u, di_d = self._split_array(distances, down)
            prx_u, prin_u = predict_class(X_U, i_u, di_u, node.get_up())
            prx_d, prin_d = predict_class(X_D, i_d, di_d, node.get_down())
            return np.append(prx_u, prx_d), np.append(prin_u, prin_d)

        # sklearn check
        check_is_fitted(self, ['tree_'])
        # Input validation
        X = check_array(X)
        # setup prediction & make it happen
        indices = np.arange(X.shape[0])
        empty_dist = np.empty((X.shape[0], 1), dtype=float)
        result, indices = predict_class(X, indices, empty_dist, self.tree_)
        result = result.reshape(X.shape[0], 2)
        # Turn distances to hyperplane into probabilities based on fitting distances
        # of samples to its hyperplane that classified them, to the sigmoid function
        # Probability of being 1
        result[:, 1] = 1 / (1 + np.exp(-result[:, 1]))
        result[:, 0] = 1 - result[:, 1]  # Probability of being 0
        return self._reorder_results(result, indices)

    def score(self, X: np.array, y: np.array) -> float:
        """Return accuracy
        """
        # sklearn check
        check_is_fitted(self)
        yp = self.predict(X).reshape(y.shape)
        right = (yp == y).astype(int)
        return np.sum(right) / len(y)

    def __iter__(self) -> Siterator:
        try:
            tree = self.tree_
        except:
            tree = None
        return Siterator(tree)

    def __str__(self) -> str:
        output = ''
        for i in self:
            output += str(i) + '\n'
        return output

