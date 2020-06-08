"""
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
__version__ = "0.9"
Build an oblique tree classifier based on SVM Trees
"""

import os

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    _check_sample_weight,
)


class Snode:
    """Nodes of the tree that keeps the svm classifier and if testing the
    dataset assigned to it
    """

    def __init__(self, clf: SVC, X: np.ndarray, y: np.ndarray, title: str):
        self._clf = clf
        self._title = title
        self._belief = 0.0
        # Only store dataset in Testing
        self._X = X if os.environ.get("TESTING", "NS") != "NS" else None
        self._y = y
        self._down = None
        self._up = None
        self._class = None

    @classmethod
    def copy(cls, node: "Snode") -> "Snode":
        return cls(node._clf, node._X, node._y, node._title)

    def set_down(self, son):
        self._down = son

    def set_up(self, son):
        self._up = son

    def is_leaf(self) -> bool:
        return self._up is None and self._down is None

    def get_down(self) -> "Snode":
        return self._down

    def get_up(self) -> "Snode":
        return self._up

    def make_predictor(self):
        """Compute the class of the predictor and its belief based on the
        subdataset of the node only if it is a leaf
        """
        if not self.is_leaf():
            return
        classes, card = np.unique(self._y, return_counts=True)
        if len(classes) > 1:
            max_card = max(card)
            min_card = min(card)
            try:
                self._belief = max_card / (max_card + min_card)
            except ZeroDivisionError:
                self._belief = 0.0
            self._class = classes[card == max_card][0]
        else:
            self._belief = 1
            self._class = classes[0]

    def __str__(self) -> str:
        if self.is_leaf():
            count_values = np.unique(self._y, return_counts=True)
            result = (
                f"{self._title} - Leaf class={self._class} belief="
                f"{self._belief: .6f} counts={count_values}"
            )
            return result
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
    """Estimator that is based on binary trees of svm nodes
    can deal with sample_weights in predict, used in boosting sklearn methods
    inheriting from BaseEstimator implements get_params and set_params methods
    inheriting from ClassifierMixin implement the attribute _estimator_type
    with "classifier" as value
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "linear",
        max_iter: int = 1000,
        random_state: int = None,
        max_depth: int = None,
        tol: float = 1e-4,
        degree: int = 3,
        gamma="scale",
        min_samples_split: int = 0,
    ):
        self.max_iter = max_iter
        self.C = C
        self.kernel = kernel
        self.random_state = random_state
        self.max_depth = max_depth
        self.tol = tol
        self.gamma = gamma
        self.degree = degree
        self.min_samples_split = min_samples_split

    def _more_tags(self) -> dict:
        """Required by sklearn to tell that this estimator is a binary classifier

        :return: the tag required
        :rtype: dict
        """
        return {"binary_only": True, "requires_y": True}

    def _split_array(self, origin: np.array, down: np.array) -> list:
        """Split an array in two based on indices passed as down and its complement

        :param origin: dataset to split
        :type origin: np.array
        :param down: indices to use to split array
        :type down: np.array
        :return: list with two splits of the array
        :rtype: list
        """
        up = ~down
        return (
            origin[up[:, 0]] if any(up) else None,
            origin[down[:, 0]] if any(down) else None,
        )

    def _distances(self, node: Snode, data: np.ndarray) -> np.array:
        """Compute distances of the samples to the hyperplane of the node

        :param node: node containing the svm classifier
        :type node: Snode
        :param data: samples to find out distance to hyperplane
        :type data: np.ndarray
        :return: array of shape (m, 1) with the distances of every sample to
        the hyperplane of the node
        :rtype: np.array
        """
        res = node._clf.decision_function(data)
        if res.ndim == 1:
            return np.expand_dims(res, 1)
        elif res.shape[1] > 1:
            res = np.delete(res, slice(1, res.shape[1]), axis=1)
        return res

    def _split_criteria(self, data: np.array) -> np.array:
        """Set the criteria to split arrays

        :param data: [description]
        :type data: np.array
        :return: [description]
        :rtype: np.array
        """
        return (
            data > 0
            if data.shape[0] >= self.min_samples_split
            else np.ones((data.shape[0], 1), dtype=bool)
        )

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.array = None
    ) -> "Stree":
        """Build the tree based on the dataset of samples and its labels

        :raises ValueError: if parameters C or max_depth are out of bounds
        :return: itself to be able to chain actions: fit().predict() ...
        :rtype: Stree
        """
        # Check parameters are Ok.
        if type(y).__name__ == "np.ndarray":
            y = y.ravel()
        if self.C < 0:
            raise ValueError(
                f"Penalty term must be positive... got (C={self.C:f})"
            )
        self.__max_depth = (
            np.iinfo(np.int32).max
            if self.max_depth is None
            else self.max_depth
        )
        if self.__max_depth < 1:
            raise ValueError(
                f"Maximum depth has to be greater than 1... got (max_depth=\
                    {self.max_depth})"
            )
        check_classification_targets(y)
        X, y = check_X_y(X, y)
        sample_weight = _check_sample_weight(sample_weight, X)
        check_classification_targets(y)
        # Initialize computed parameters
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_iter_ = self.max_iter
        self.depth_ = 0
        self.n_features_in_ = X.shape[1]
        self.tree_ = self.train(X, y, sample_weight, 1, "root")
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

    def _build_clf(self):
        """ Build the correct classifier for the node
        """
        return (
            LinearSVC(
                max_iter=self.max_iter,
                random_state=self.random_state,
                C=self.C,
                tol=self.tol,
            )
            if self.kernel == "linear"
            else SVC(
                kernel=self.kernel,
                max_iter=self.max_iter,
                tol=self.tol,
                C=self.C,
                gamma=self.gamma,
                degree=self.degree,
            )
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        depth: int,
        title: str,
    ) -> Snode:
        """Recursive function to split the original dataset into predictor
        nodes (leaves)

        :param X: samples dataset
        :type X: np.ndarray
        :param y: samples labels
        :type y: np.ndarray
        :param sample_weight: weight of samples (used in boosting)
        :type sample_weight: np.ndarray
        :param depth: actual depth in the tree
        :type depth: int
        :param title: description of the node
        :type title: str
        :return: binary tree
        :rtype: Snode
        """
        if depth > self.__max_depth:
            return None
        if np.unique(y).shape[0] == 1:
            # only 1 class => pure dataset
            return Snode(None, X, y, title + ", <pure>")
        # Train the model
        clf = self._build_clf()
        clf.fit(X, y, sample_weight=sample_weight)
        tree = Snode(clf, X, y, title)
        self.depth_ = max(depth, self.depth_)
        down = self._split_criteria(self._distances(tree, X))
        X_U, X_D = self._split_array(X, down)
        y_u, y_d = self._split_array(y, down)
        sw_u, sw_d = self._split_array(sample_weight, down)
        if X_U is None or X_D is None:
            # didn't part anything
            return Snode(clf, X, y, title + ", <cgaf>")
        tree.set_up(self.train(X_U, y_u, sw_u, depth + 1, title + " - Up"))
        tree.set_down(self.train(X_D, y_d, sw_d, depth + 1, title + " - Down"))
        return tree

    def _reorder_results(self, y: np.array, indices: np.array) -> np.array:
        """Reorder an array based on the array of indices passed

        :param y: data untidy
        :type y: np.array
        :param indices: indices used to set order
        :type indices: np.array
        :return: array y ordered
        :rtype: np.array
        """
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
        """Predict labels for each sample in dataset passed

        :param X: dataset of samples
        :type X: np.array
        :return: array of labels
        :rtype: np.array
        """

        def predict_class(
            xp: np.array, indices: np.array, node: Snode
        ) -> np.array:
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
        check_is_fitted(self, ["tree_"])
        # Input validation
        X = check_array(X)
        # setup prediction & make it happen
        indices = np.arange(X.shape[0])
        result = (
            self._reorder_results(*predict_class(X, indices, self.tree_))
            .astype(int)
            .ravel()
        )
        return self.classes_[result]

    def predict_proba(self, X: np.array) -> np.array:
        """Computes an approximation of the probability of samples belonging to
        class 0 and 1
        :param X: dataset
        :type X: np.array
        :return: array array of shape (m, num_classes), probability of being
        each class
        :rtype: np.array
        """

        def predict_class(
            xp: np.array, indices: np.array, dist: np.array, node: Snode
        ) -> np.array:
            """Run the tree to compute predictions

            :param xp: subdataset of samples
            :type xp: np.array
            :param indices: indices of subdataset samples to rebuild original
            order
            :type indices: np.array
            :param dist: distances of every sample to the hyperplane or the
            father node
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
        check_is_fitted(self, ["tree_"])
        # Input validation
        X = check_array(X)
        # setup prediction & make it happen
        indices = np.arange(X.shape[0])
        empty_dist = np.empty((X.shape[0], 1), dtype=float)
        result, indices = predict_class(X, indices, empty_dist, self.tree_)
        result = result.reshape(X.shape[0], 2)
        # Turn distances to hyperplane into probabilities based on fitting
        # distances of samples to its hyperplane that classified them, to the
        # sigmoid function
        # Probability of being 1
        result[:, 1] = 1 / (1 + np.exp(-result[:, 1]))
        # Probability of being 0
        result[:, 0] = 1 - result[:, 1]
        return self._reorder_results(result, indices)

    def score(self, X: np.array, y: np.array) -> float:
        """Compute accuracy of the prediction

        :param X: dataset of samples to make predictions
        :type X: np.array
        :param y: samples labels
        :type y: np.array
        :return: accuracy of the prediction
        :rtype: float
        """
        # sklearn check
        check_is_fitted(self)
        yp = self.predict(X).reshape(y.shape)
        return np.mean(yp == y)

    def __iter__(self) -> Siterator:
        """Create an iterator to be able to visit the nodes of the tree in preorder,
        can make a list with all the nodes in preorder

        :return: an iterator, can for i in... and list(...)
        :rtype: Siterator
        """
        try:
            tree = self.tree_
        except AttributeError:
            tree = None
        return Siterator(tree)

    def __str__(self) -> str:
        """String representation of the tree

        :return: description of nodes in the tree in preorder
        :rtype: str
        """
        output = ""
        for i in self:
            output += str(i) + "\n"
        return output
