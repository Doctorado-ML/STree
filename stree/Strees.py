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
import warnings
from math import log
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import check_classification_targets
from sklearn.exceptions import ConvergenceWarning
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
        weight: np.ndarray = None,
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
        self._sample_weight = (
            weight if os.environ.get("TESTING", "NS") != "NS" else None
        )
        self._features = features
        self._impurity = impurity
        self._partition_column: int = -1

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

    def set_partition_column(self, col: int):
        self._partition_column = col

    def get_partition_column(self) -> int:
        return self._partition_column

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
            self._class = classes[card == max_card][0]
            self._belief = max_card / np.sum(card)
        else:
            self._belief = 1
            try:
                self._class = classes[0]
            except IndexError:
                self._class = None

    def __str__(self) -> str:
        count_values = np.unique(self._y, return_counts=True)
        if self.is_leaf():
            return (
                f"{self._title} - Leaf class={self._class} belief="
                f"{self._belief: .6f} impurity={self._impurity:.4f} "
                f"counts={count_values}"
            )
        else:
            return (
                f"{self._title} feaures={self._features} impurity="
                f"{self._impurity:.4f} "
                f"counts={count_values}"
            )


class Siterator:
    """Stree preorder iterator"""

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


class Splitter:
    def __init__(
        self,
        clf: SVC = None,
        criterion: str = None,
        splitter_type: str = None,
        criteria: str = None,
        min_samples_split: int = None,
        random_state=None,
    ):
        self._clf = clf
        self._random_state = random_state
        if random_state is not None:
            random.seed(random_state)
        self._criterion = criterion
        self._min_samples_split = min_samples_split
        self._criteria = criteria
        self._splitter_type = splitter_type

        if clf is None:
            raise ValueError(f"clf has to be a sklearn estimator, got({clf})")

        if criterion not in ["gini", "entropy"]:
            raise ValueError(
                f"criterion must be gini or entropy got({criterion})"
            )

        if criteria not in [
            "max_samples",
            "impurity",
        ]:
            raise ValueError(
                f"criteria has to be max_samples or impurity; got ({criteria})"
            )

        if splitter_type not in ["random", "best"]:
            raise ValueError(
                f"splitter must be either random or best, got({splitter_type})"
            )
        self.criterion_function = getattr(self, f"_{self._criterion}")
        self.decision_criteria = getattr(self, f"_{self._criteria}")

    def partition_impurity(self, y: np.array) -> np.array:
        return self.criterion_function(y)

    @staticmethod
    def _gini(y: np.array) -> float:
        _, count = np.unique(y, return_counts=True)
        return 1 - np.sum(np.square(count / np.sum(count)))

    @staticmethod
    def _entropy(y: np.array) -> float:
        n_labels = len(y)
        if n_labels <= 1:
            return 0
        counts = np.bincount(y)
        proportions = counts / n_labels
        n_classes = np.count_nonzero(proportions)
        if n_classes <= 1:
            return 0
        entropy = 0.0
        # Compute standard entropy.
        for prop in proportions:
            if prop != 0.0:
                entropy -= prop * log(prop, n_classes)
        return entropy

    def information_gain(
        self, labels: np.array, labels_up: np.array, labels_dn: np.array
    ) -> float:
        imp_prev = self.criterion_function(labels)
        card_up = card_dn = imp_up = imp_dn = 0
        if labels_up is not None:
            card_up = labels_up.shape[0]
            imp_up = self.criterion_function(labels_up)
        if labels_dn is not None:
            card_dn = labels_dn.shape[0] if labels_dn is not None else 0
            imp_dn = self.criterion_function(labels_dn)
        samples = card_up + card_dn
        if samples == 0:
            return 0.0
        else:
            result = (
                imp_prev
                - (card_up / samples) * imp_up
                - (card_dn / samples) * imp_dn
            )
            return result

    def _select_best_set(
        self, dataset: np.array, labels: np.array, features_sets: list
    ) -> list:
        max_gain = 0
        selected = None
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for feature_set in features_sets:
            self._clf.fit(dataset[:, feature_set], labels)
            node = Snode(
                self._clf, dataset, labels, feature_set, 0.0, "subset"
            )
            self.partition(dataset, node, train=True)
            y1, y2 = self.part(labels)
            gain = self.information_gain(labels, y1, y2)
            if gain > max_gain:
                max_gain = gain
                selected = feature_set
        return selected if selected is not None else feature_set

    @staticmethod
    def _generate_spaces(features: int, max_features: int) -> list:
        comb = set()
        # Generate at most 3 combinations
        set_length = 1 if max_features == features else 3
        while len(comb) < set_length:
            comb.add(
                tuple(sorted(random.sample(range(features), max_features)))
            )
        return list(comb)

    def _get_subspaces_set(
        self, dataset: np.array, labels: np.array, max_features: int
    ) -> np.array:
        features_sets = self._generate_spaces(dataset.shape[1], max_features)
        if len(features_sets) > 1:
            if self._splitter_type == "random":
                index = random.randint(0, len(features_sets) - 1)
                return features_sets[index]
            else:
                return self._select_best_set(dataset, labels, features_sets)
        else:
            return features_sets[0]

    def get_subspace(
        self, dataset: np.array, labels: np.array, max_features: int
    ) -> tuple:
        """Return the best/random subspace to make a split"""
        indices = self._get_subspaces_set(dataset, labels, max_features)
        return dataset[:, indices], indices

    def _impurity(self, data: np.array, y: np.array) -> np.array:
        """return column of dataset to be taken into account to split dataset

        :param data: distances to hyper plane of every class
        :type data: np.array (m, n_classes)
        :param y: vector of labels (classes)
        :type y: np.array (m,)
        :return: column of dataset to be taken into account to split dataset
        :rtype: int
        """
        max_gain = 0
        selected = -1
        for col in range(data.shape[1]):
            tup = y[data[:, col] > 0]
            tdn = y[data[:, col] <= 0]
            info_gain = self.information_gain(y, tup, tdn)
            if info_gain > max_gain:
                selected = col
                max_gain = info_gain
        return selected

    @staticmethod
    def _max_samples(data: np.array, y: np.array) -> np.array:
        """return column of dataset to be taken into account to split dataset

        :param data: distances to hyper plane of every class
        :type data: np.array (m, n_classes)
        :param y: vector of labels (classes)
        :type y: np.array (m,)
        :return: column of dataset to be taken into account to split dataset
        :rtype: int
        """
        # select the class with max number of samples
        _, samples = np.unique(y, return_counts=True)
        return np.argmax(samples)

    def partition(self, samples: np.array, node: Snode, train: bool):
        """Set the criteria to split arrays. Compute the indices of the samples
        that should go to one side of the tree (down)

        """
        # data contains the distances of every sample to every class hyperplane
        # array of (m, nc) nc = # classes
        data = self._distances(node, samples)
        if data.shape[0] < self._min_samples_split:
            # there aren't enough samples to split
            self._up = np.ones((data.shape[0]), dtype=bool)
            return
        if data.ndim > 1:
            # split criteria for multiclass
            # Convert data to a (m, 1) array selecting values for samples
            if train:
                # in train time we have to compute the column to take into
                # account to split the dataset
                col = self.decision_criteria(data, node._y)
                node.set_partition_column(col)
            else:
                # in predcit time just use the column computed in train time
                # is taking the classifier of class <col>
                col = node.get_partition_column()
                if col == -1:
                    # No partition is producing information gain
                    data = np.ones(data.shape)
            data = data[:, col]
        self._up = data > 0

    def part(self, origin: np.array) -> list:
        """Split an array in two based on indices (down) and its complement
        partition has to be called first to establish down indices

        :param origin: dataset to split
        :type origin: np.array
        :param down: indices to use to split array
        :type down: np.array
        :return: list with two splits of the array
        :rtype: list
        """
        down = ~self._up
        return [
            origin[self._up] if any(self._up) else None,
            origin[down] if any(down) else None,
        ]

    @staticmethod
    def _distances(node: Snode, data: np.ndarray) -> np.array:
        """Compute distances of the samples to the hyperplane of the node

        :param node: node containing the svm classifier
        :type node: Snode
        :param data: samples to find out distance to hyperplane
        :type data: np.ndarray
        :return: array of shape (m, nc) with the distances of every sample to
        the hyperplane of every class. nc = # of classes
        :rtype: np.array
        """
        return node._clf.decision_function(data[:, node._features])


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
        max_iter: int = 1e5,
        random_state: int = None,
        max_depth: int = None,
        tol: float = 1e-4,
        degree: int = 3,
        gamma="scale",
        split_criteria: str = "impurity",
        criterion: str = "entropy",
        min_samples_split: int = 0,
        max_features=None,
        splitter: str = "random",
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
        self.splitter = splitter

    def _more_tags(self) -> dict:
        """Required by sklearn to supply features of the classifier

        :return: the tag required
        :rtype: dict
        """
        return {"requires_y": True}

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

        check_classification_targets(y)
        X, y = check_X_y(X, y)
        sample_weight = _check_sample_weight(
            sample_weight, X, dtype=np.float64
        )
        check_classification_targets(y)
        # Initialize computed parameters
        self.splitter_ = Splitter(
            clf=self._build_clf(),
            criterion=self.criterion,
            splitter_type=self.splitter,
            criteria=self.split_criteria,
            random_state=self.random_state,
            min_samples_split=self.min_samples_split,
        )
        if self.random_state is not None:
            random.seed(self.random_state)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]
        self.n_iter_ = self.max_iter
        self.depth_ = 0
        self.n_features_ = X.shape[1]
        self.n_features_in_ = X.shape[1]
        self.max_features_ = self._initialize_max_features()
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
                weight=sample_weight,
            )
        # Train the model
        clf = self._build_clf()
        Xs, features = self.splitter_.get_subspace(X, y, self.max_features_)
        # solve WARNING: class label 0 specified in weight is not found
        # in bagging
        if any(sample_weight == 0):
            indices = sample_weight == 0
            y_next = y[~indices]
            # touch weights if removing any class
            if np.unique(y_next).shape[0] != self.n_classes_:
                sample_weight += 1e-5
        clf.fit(Xs, y, sample_weight=sample_weight)
        impurity = self.splitter_.partition_impurity(y)
        node = Snode(clf, X, y, features, impurity, title, sample_weight)
        self.depth_ = max(depth, self.depth_)
        self.splitter_.partition(X, node, True)
        X_U, X_D = self.splitter_.part(X)
        y_u, y_d = self.splitter_.part(y)
        sw_u, sw_d = self.splitter_.part(sample_weight)
        if X_U is None or X_D is None:
            # didn't part anything
            return Snode(
                clf,
                X,
                y,
                features=X.shape[1],
                impurity=impurity,
                title=title + ", <cgaf>",
                weight=sample_weight,
            )
        node.set_up(self.train(X_U, y_u, sw_u, depth + 1, title + " - Up"))
        node.set_down(self.train(X_D, y_d, sw_d, depth + 1, title + " - Down"))
        return node

    def _build_predictor(self):
        """Process the leaves to make them predictors"""

        def run_tree(node: Snode):
            if node.is_leaf():
                node.make_predictor()
                return
            run_tree(node.get_down())
            run_tree(node.get_up())

        run_tree(self.tree_)

    def _build_clf(self):
        """Build the correct classifier for the node"""
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

    @staticmethod
    def _reorder_results(y: np.array, indices: np.array) -> np.array:
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
            self.splitter_.partition(xp, node, train=False)
            x_u, x_d = self.splitter_.part(xp)
            i_u, i_d = self.splitter_.part(indices)
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
        _, y_true, y_pred = _check_targets(y, y_pred)
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
