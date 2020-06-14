"""
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
__version__ = "0.9"
Build an oblique tree classifier based on SVM Trees
"""

import os
import numbers
import random
from itertools import combinations
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    _check_sample_weight,
)
from sklearn.metrics._classification import _weighted_sum, _check_targets


class Snode:
    """Nodes of the tree that keeps the svm classifier and if testing the
    dataset assigned to it
    """

    def __init__(
        self,
        clf: SVC,
        X: np.ndarray,
        y: np.ndarray,
        features: np.array,
        impurity: float,
        title: str,
    ):
        self._clf = clf
        self._title = title
        self._belief = 0.0
        # Only store dataset in Testing
        self._X = X if os.environ.get("TESTING", "NS") != "NS" else None
        self._y = y
        self._down = None
        self._up = None
        self._class = None
        self._feature = None
        self._sample_weight = None
        self._features = features
        self._impurity = impurity

    @classmethod
    def copy(cls, node: "Snode") -> "Snode":
        return cls(
            node._clf,
            node._X,
            node._y,
            node._features,
            node._impurity,
            node._title,
        )

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
            self._class = classes[card == max_card][0]
            self._belief = max_card / (max_card + min_card)
        else:
            self._belief = 1
            try:
                self._class = classes[0]
            except IndexError:
                self._class = None

    def __str__(self) -> str:
        if self.is_leaf():
            count_values = np.unique(self._y, return_counts=True)
            result = (
                f"{self._title} - Leaf class={self._class} belief="
                f"{self._belief: .6f} impurity={self._impurity:.4f} "
                f"counts={count_values}"
            )
            return result
        else:
            return (
                f"{self._title} feaures={self._features} impurity="
                f"{self._impurity:.4f}"
            )


class Siterator:
    """Stree preorder iterator
    """

    def __init__(self, tree: Snode):
        self._stack = []
        self._push(tree)

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
        split_criteria: str = "max_samples",
        criterion: str = "gini",
        min_samples_split: int = 0,
        max_features=None,
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
        self.split_criteria = split_criteria
        self.max_features = max_features
        self.criterion = criterion

    def _more_tags(self) -> dict:
        """Required by sklearn to supply features of the classifier

        :return: the tag required
        :rtype: dict
        """
        return {"requires_y": True}

    def _split_array(self, origin: np.array, down: np.array) -> list:
        """Split an array in two based on indices (down) and its complement

        :param origin: dataset to split
        :type origin: np.array
        :param down: indices to use to split array
        :type down: np.array
        :return: list with two splits of the array
        :rtype: list
        """
        up = ~down
        return [
            origin[up] if any(up) else None,
            origin[down] if any(down) else None,
        ]

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
        return node._clf.decision_function(data[:, node._features])

    def _min_distance(self, data: np.array, _) -> np.array:
        # chooses the lowest distance of every sample
        indices = np.argmin(np.abs(data), axis=1)
        return np.array(
            [data[x, y] for x, y in zip(range(len(data[:, 0])), indices)]
        )

    def _max_samples(self, data: np.array, y: np.array) -> np.array:
        # select the class with max number of samples
        _, samples = np.unique(y, return_counts=True)
        selected = np.argmax(samples)
        return data[:, selected]

    def _split_criteria(self, data: np.array, node: Snode) -> np.array:
        """Set the criteria to split arrays

        :param data: distances of samples to hyperplanes shape (m, nclasses)
        if nclasses > 2 else (m,)
        :type data: np.array
        :param node: node containing the svm classifier
        :type node: Snode
        :return: array of booleans of samples under or above zero
        :rtype: np.array
        """

        if data.shape[0] < self.min_samples_split:
            return np.ones((data.shape[0]), dtype=bool)
        if data.ndim > 1:
            # split criteria for multiclass
            data = getattr(self, f"_{self.split_criteria}")(data, node._y)
        res = data > 0
        return res

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.array = None
    ) -> "Stree":
        """Build the tree based on the dataset of samples and its labels

        :param X: dataset of samples to make predictions
        :type X: np.array
        :param y: samples labels
        :type y: np.array
        :param sample_weight: weights of the samples. Rescale C per sample.
        Hi' weights force the classifier to put more emphasis on these points
        :type sample_weight: np.array optional
        :raises ValueError: if parameters C or max_depth are out of bounds
        :return: itself to be able to chain actions: fit().predict() ...
        :rtype: Stree
        """
        # Check parameters are Ok.
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
        if self.split_criteria not in ["min_distance", "max_samples"]:
            raise ValueError(
                f"split_criteria has to be min_distance or \
                max_samples got ({self.split_criteria})"
            )
        if self.criterion not in ["gini", "entropy"]:
            raise ValueError(
                f"criterion must be gini or entropy got({self.criterion})"
            )

        check_classification_targets(y)
        X, y = check_X_y(X, y)
        sample_weight = _check_sample_weight(sample_weight, X)
        check_classification_targets(y)
        # Initialize computed parameters
        if self.random_state is not None:
            random.seed(self.random_state)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]
        self.n_iter_ = self.max_iter
        self.depth_ = 0
        self.n_features_ = X.shape[1]
        self.n_features_in_ = X.shape[1]
        self.max_features_ = self._initialize_max_features()
        self.criterion_function_ = getattr(self, f"_{self.criterion}")
        self.tree_ = self.train(X, y, sample_weight, 1, "root")
        self._build_predictor()
        return self

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
        :param sample_weight: weight of samples. Rescale C per sample.
        Hi weights force the classifier to put more emphasis on these points.
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
            return Snode(
                clf=None,
                X=X,
                y=y,
                features=X.shape[1],
                impurity=0.0,
                title=title + ", <pure>",
            )
        # Train the model
        clf = self._build_clf()
        Xs, features = self._get_subspace(X)
        clf.fit(Xs, y, sample_weight=sample_weight)
        impurity = self.criterion_function_(y)
        node = Snode(clf, X, y, features, impurity, title)
        self.depth_ = max(depth, self.depth_)
        down = self._split_criteria(self._distances(node, X), node)
        X_U, X_D = self._split_array(X, down)
        y_u, y_d = self._split_array(y, down)
        sw_u, sw_d = self._split_array(sample_weight, down)
        if X_U is None or X_D is None:
            # didn't part anything
            return Snode(
                clf,
                X,
                y,
                features=X.shape[1],
                impurity=impurity,
                title=title + ", <cgaf>",
            )
        node.set_up(self.train(X_U, y_u, sw_u, depth + 1, title + " - Up"))
        node.set_down(self.train(X_D, y_d, sw_d, depth + 1, title + " - Down"))
        return node

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

    def _reorder_results(self, y: np.array, indices: np.array) -> np.array:
        """Reorder an array based on the array of indices passed

        :param y: data untidy
        :type y: np.array
        :param indices: indices used to set order
        :type indices: np.array
        :return: array y ordered
        :rtype: np.array
        """
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
            down = self._split_criteria(self._distances(node, xp), node)
            x_u, x_d = self._split_array(xp, down)
            i_u, i_d = self._split_array(indices, down)
            prx_u, prin_u = predict_class(x_u, i_u, node.get_up())
            prx_d, prin_d = predict_class(x_d, i_d, node.get_down())
            return np.append(prx_u, prx_d), np.append(prin_u, prin_d)

        # sklearn check
        check_is_fitted(self, ["tree_"])
        # Input validation
        X = check_array(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features but got "
                f"({X.shape[1]})"
            )
        # setup prediction & make it happen
        indices = np.arange(X.shape[0])
        result = (
            self._reorder_results(*predict_class(X, indices, self.tree_))
            .astype(int)
            .ravel()
        )
        return self.classes_[result]

    def score(
        self, X: np.array, y: np.array, sample_weight: np.array = None
    ) -> float:
        """Compute accuracy of the prediction

        :param X: dataset of samples to make predictions
        :type X: np.array
        :param y_true: samples labels
        :type y_true: np.array
        :param sample_weight: weights of the samples. Rescale C per sample.
        Hi' weights force the classifier to put more emphasis on these points
        :type sample_weight: np.array optional
        :return: accuracy of the prediction
        :rtype: float
        """
        # sklearn check
        check_is_fitted(self)
        check_classification_targets(y)
        X, y = check_X_y(X, y)
        y_pred = self.predict(X).reshape(y.shape)
        # Compute accuracy for each possible representation
        y_type, y_true, y_pred = _check_targets(y, y_pred)
        check_consistent_length(y_true, y_pred, sample_weight)
        score = y_true == y_pred
        return _weighted_sum(score, sample_weight, normalize=True)

    def __iter__(self) -> Siterator:
        """Create an iterator to be able to visit the nodes of the tree in
        preorder, can make a list with all the nodes in preorder

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

    def _initialize_max_features(self) -> int:
        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    "Invalid value for max_features. "
                    "Allowed string values are 'auto', "
                    "'sqrt' or 'log2'."
                )
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(
                    1, int(self.max_features * self.n_features_)
                )
            else:
                raise ValueError(
                    "Invalid value for max_features."
                    "Allowed float must be in range (0, 1] "
                    f"got ({self.max_features})"
                )
        return max_features

    @staticmethod
    def _gini(y: np.array) -> float:
        _, count = np.unique(y, return_counts=True)
        return 1 - np.sum(np.square(count / np.sum(count)))

    @staticmethod
    def _entropy(y: np.array) -> float:
        _, count = np.unique(y, return_counts=True)
        proportion = count / np.sum(count)
        return -np.sum(proportion * np.log2(proportion))

    def _get_subspace(self, dataset: np.array) -> list:
        """Return the best subspace to make a split
        """

        def get_subspaces_set(dataset: np.array) -> np.array:
            features = range(dataset.shape[1])
            features_sets = list(combinations(features, self.max_features_))
            if len(features_sets) > 1:
                return features_sets[random.randint(0, len(features_sets) - 1)]
            else:
                return features_sets[0]

        indices = get_subspaces_set(dataset)
        return dataset[:, indices], indices
