"""
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
__version__ = "0.9"
Build an oblique tree classifier based on SVM Trees
"""

from __future__ import annotations
import os
import random
import warnings
from typing import Optional, List, Union, Tuple
from math import log
from itertools import combinations
import numpy as np  # type: ignore
from sklearn.base import BaseEstimator, ClassifierMixin  # type: ignore
from sklearn.svm import SVC, LinearSVC  # type: ignore
from sklearn.utils.multiclass import (  # type: ignore
    check_classification_targets,
)
from sklearn.exceptions import ConvergenceWarning  # type: ignore
from sklearn.utils.validation import (  # type: ignore
    check_is_fitted,
    _check_sample_weight,
)


class Snode:
    """Stores the information of each node of the tree

        :param clf: classifier used in the node to split data
        :type clf: Union[SVC, LinearSVC]
        :param X: dataset passed to the node (only stored in tests)
        :type X: np.ndarray
        :param y: labels of the dataset
        :type y: np.ndarray
        :param features: supsace of features taken by the classifier to split
        :type features: np.array
        :param impurity: the gini/entropy measure
        :type impurity: float
        :param title: description of the node
        :type title: str
        :param weight: weights applied to dataset (only stored in tests),
         defaults to None
        :type weight: np.ndarray, optional
    """

    def __init__(
        self,
        clf: Union[SVC, LinearSVC],
        X: np.ndarray,
        y: np.ndarray,
        features: np.array,
        impurity: float,
        title: str,
        weight: np.ndarray = None,
    ) -> None:
        """constructor method
        """

        self._clf: Union[SVC, LinearSVC] = clf
        self._title: str = title
        self._belief: float = 0.0
        # Only store dataset in Testing
        self._X: Optional[np.array] = X if os.environ.get(
            "TESTING", "NS"
        ) != "NS" else None
        self._y: np.array = y
        self._down: Optional[Snode] = None
        self._up: Optional[Snode] = None
        self._class = None
        self._sample_weight: Optional[np.array] = (
            weight if os.environ.get("TESTING", "NS") != "NS" else None
        )
        self._features: Tuple[int, ...] = features
        self._impurity: float = impurity

    @classmethod
    def copy(cls, node: Snode) -> Snode:
        """Clone the node

        :param node: source node
        :type node: Snode
        :return: cloned node
        :rtype: Snode
        """
        return cls(
            node._clf,
            node._X,
            node._y,
            node._features,
            node._impurity,
            node._title,
        )

    def set_down(self, son: Snode) -> None:
        """Sets the left/down son of the node

        :param son: the son of the node
        :type son: Snode
        """
        self._down = son

    def set_up(self, son: Snode) -> None:
        """Sets the right/up son of the node

        :param son: the son of the node
        :type son: Snode
        """
        self._up = son

    def is_leaf(self) -> bool:
        """Test if the node is a leaf in the tree

        :return: node is leaf or not
        :rtype: bool
        """
        return self._up is None and self._down is None

    def get_down(self) -> Optional[Snode]:
        """Get the left/down son of the node if exists or None

        :return: the son selected
        :rtype: Optional[Snode]
        """
        return self._down

    def get_up(self) -> Optional[Snode]:
        """Get the right/up son of the node if exists or None

        :return: the son selected
        :rtype: Optional[Snode]
        """
        return self._up

    def make_predictor(self) -> None:
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
        """Return information of the node useful to represent the tree

        :return: description of the node
        :rtype: str
        """
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

    :param tree: tree to iterate with
    :type tree: Optional[Snode]
    """

    def __init__(self, tree: Optional[Snode]) -> None:
        """constructor method
        """
        self._stack: List[Snode] = []
        self._push(tree)

    def _push(self, node: Optional[Snode]) -> None:
        """Store the node in the stack of pending nodes

        :param node: the node to store
        :type node: Optional[Snode]
        """
        if node is not None:
            self._stack.append(node)

    def __next__(self) -> Snode:
        """Gets the next node of the tree in preorder

        :raises StopIteration: if no nodes left
        :return: the next node
        :rtype: Snode
        """
        if len(self._stack) == 0:
            raise StopIteration()
        node = self._stack.pop()
        self._push(node.get_up())
        self._push(node.get_down())
        return node


class Splitter:
    def __init__(
        self,
        clf: Union[SVC, LinearSVC],
        criterion: str,
        splitter_type: str,
        criteria: str,
        min_samples_split: int = 0,
        random_state: Optional[int] = None,
    ) -> None:
        """Splits datasets based on different criteria

        :param clf: classifier used to split, defaults to None
        :type clf: Union[SVC, LinearSVC]
        :param criterion: impurity criteria {gini, entropy}
        :type criterion: str
        :param splitter_type: result selection {random, best}
        :type splitter_type: str
        :param criteria: split criteria {min or max distance or max samples}
        :type criteria: str
        :param min_samples_split: min number of samples to split, default 0
        :type min_samples_split: int, optional
        :param random_state: random seed, defaults to None
        :type random_state: Optional[int], optional
        :raises ValueError: if classifier is None
        :raises ValueError: if criterion is not gini or entropy
        :raises ValueError: if criteria is not min or max distance or max samp
        :raises ValueError: if splitter_type is not best or random
        """
        self._clf: Union[SVC, LinearSVC] = clf
        self._random_state: Optional[int] = random_state
        if random_state is not None:
            random.seed(random_state)
        self._criterion: str = criterion
        self._min_samples_split: int = min_samples_split
        self._criteria: str = criteria
        self._splitter_type: str = splitter_type

        if clf is None:
            raise ValueError(f"clf has to be a sklearn estimator, got({clf})")

        if criterion not in ["gini", "entropy"]:
            raise ValueError(
                f"criterion must be gini or entropy got({criterion})"
            )

        if criteria not in ["min_distance", "max_samples", "max_distance"]:
            raise ValueError(
                "split_criteria has to be min_distance "
                f"max_distance or max_samples got ({criteria})"
            )

        if splitter_type not in ["random", "best"]:
            raise ValueError(
                f"splitter must be either random or best got({splitter_type})"
            )
        self.criterion_function = getattr(self, f"_{self._criterion}")
        self.decision_criteria = getattr(self, f"_{self._criteria}")

    def impurity(self, y: np.array) -> float:
        """Apply the criterion function entropy or gini

        :param y: labels to compute impurity
        :type y: np.array
        :return: the impurity w.r.t. the labels
        :rtype: float
        """
        return float(self.criterion_function(y))

    @staticmethod
    def _gini(y: np.array) -> float:
        """compute gini impurity on the labels

        :param y: labels
        :type y: np.array
        :return: impurity
        :rtype: float
        """
        _, count = np.unique(y, return_counts=True)
        return float(1 - np.sum(np.square(count / np.sum(count))))

    @staticmethod
    def _entropy(y: np.array) -> float:
        """compute entropy on the labels

        :param y: labels
        :type y: np.array
        :return: impurity
        :rtype: float
        """
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
        """compute information gain with a proposed split

        :param labels: labels to split
        :type labels: np.array
        :param labels_up: splitted labels up
        :type labels_up: np.array
        :param labels_dn: splitted labels down
        :type labels_dn: np.array
        :return: a measure of the information gained with this split
        :rtype: float
        """
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
            result = float(
                imp_prev
                - (card_up / samples) * imp_up
                - (card_dn / samples) * imp_dn
            )
            return result

    def _select_best_set(
        self,
        dataset: np.array,
        labels: np.array,
        features_sets: List[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        """Select the subset of features that best splits the dataset

        :param dataset: samples dataset
        :type dataset: np.array
        :param labels: labels of the dataset
        :type labels: np.array
        :param features_sets: List with combinations of features
        :type features_sets: List[Tuple[int, ...]]
        :return: selected subset
        :rtype: Tuple[int, ...]
        """
        max_gain: float = 0.0
        selected: Union[Tuple[int, ...], None] = None
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for feature_set in features_sets:
            self._clf.fit(dataset[:, feature_set], labels)
            node = Snode(
                self._clf, dataset, labels, feature_set, 0.0, "subset"
            )
            self.partition(dataset, node)
            y1, y2 = self.part(labels)
            gain = self.information_gain(labels, y1, y2)
            if gain > max_gain:
                max_gain = gain
                selected = feature_set
        return selected if selected is not None else feature_set

    def _get_subspaces_set(
        self, dataset: np.array, labels: np.array, max_features: int
    ) -> Tuple[int, ...]:
        """select the subspace set according to given conditions

        :param dataset: dataset to extract features
        :type dataset: np.array
        :param labels: labels of the dataset
        :type labels: np.array
        :param max_features: number of features to group
        :type max_features: int
        :return: selected subset of features
        :rtype: Tuple[int, ...]
        """
        features = range(dataset.shape[1])
        features_sets = list(combinations(features, max_features))
        if len(features_sets) > 1:
            if self._splitter_type == "random":
                index = random.randint(0, len(features_sets) - 1)
                return features_sets[index]
            else:
                # get only 3 sets at most
                if len(features_sets) > 3:
                    features_sets = random.sample(features_sets, 3)
                return self._select_best_set(dataset, labels, features_sets)
        else:
            return features_sets[0]

    def get_subspace(
        self, dataset: np.array, labels: np.array, max_features: int
    ) -> Tuple[np.array, Tuple[int, ...]]:
        """Return subspace to make a split

        :return: dataset with only selected features and the indices of the
         features
        :rtype: Tuple[int, ...]
        """
        indices = self._get_subspaces_set(dataset, labels, max_features)
        return dataset[:, indices], indices

    @staticmethod
    def _min_distance(data: np.array, _: np.array) -> np.array:
        """Assign class to min distances
        return a vector of classes so partition can separate class 0 from
        the rest of classes, ie. class 0 goes to one splitted node and the
        rest of classes go to the other

        :param data: distances to hyper plane of every class
        :type data: np.array (m, n_classes)
        :param _: enable call compat with other measures
        :type _: None
        :return: vector with the class assigned to each sample
        :rtype: np.array shape (m,)
        """
        return np.argmin(data, axis=1)

    @staticmethod
    def _max_distance(data: np.array, _: np.array) -> np.array:
        """Assign class to max distances
        return a vector of classes so partition can separate class 0 from
        the rest of classes, ie. class 0 goes to one splitted node and the
        rest of classes go to the other

        :param data: distances to hyper plane of every class
        :type data: np.array (m, n_classes)
        :param _: enable call compat with other measures
        :type _: np.array
        :return: vector with the class assigned to each sample values
        (can be 0, 1, ...)
        :rtype: np.array shape (m,)
        """
        return np.argmax(data, axis=1)

    @staticmethod
    def _max_samples(data: np.array, y: np.array) -> np.array:
        """return distances of the class with more samples

        :param data: distances to hyper plane of every class
        :type data: np.array (m, n_classes)
        :param y: vector of labels (classes)
        :type y: np.array shape (m,)
        :return: vector with distances to hyperplane (can be positive or neg.)
        :rtype: np.array shape (m,)
        """
        # select the class with max number of samples
        _, samples = np.unique(y, return_counts=True)
        selected = np.argmax(samples)
        return data[:, selected]

    def partition(self, samples: np.array, node: Snode) -> None:
        """Set the criteria to split arrays. Compute the indices of the samples
        that should go to one side of the tree (down)

        :param samples: dataset
        :type samples: np.array
        :param node: node used to split the dataset
        :type node: Snode
        """
        data = self._distances(node, samples)
        if data.shape[0] < self._min_samples_split:
            self._down = np.ones((data.shape[0]), dtype=bool)
            return
        if data.ndim > 1:
            # split criteria for multiclass
            data = self.decision_criteria(data, node._y)
        self._down = data > 0

    @staticmethod
    def _distances(node: Snode, data: np.ndarray) -> np.array:
        """Compute distances of the samples to each hyperplane of the node (one
         per class if number of classes is greater than 2)

        :param node: node containing the svm classifier
        :type node: Snode
        :param data: samples to find out distance to hyperplane
        :type data: np.ndarray
        :return: array of shape (m, 1) with the distances of every sample to
         the hyperplane of the node
        :rtype: np.array
        """
        return node._clf.decision_function(data[:, node._features])

    def part(self, origin: np.array) -> Tuple[np.array, np.array]:
        """Split an array in two based on indices (down) and its complement

        :param origin: dataset to split
        :type origin: np.array
        :param down: indices to use to split array
        :type down: np.array
        :return: list with two splits of the array
        :rtype: Tuple[np.array, np.array]
        """
        up = ~self._down
        return (
            origin[up] if any(up) else None,
            origin[self._down] if any(self._down) else None,
        )


class Stree(BaseEstimator, ClassifierMixin):  # type: ignore
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
        random_state: Optional[int] = None,
        max_depth: Optional[int] = None,
        tol: float = 1e-4,
        degree: int = 3,
        gamma: Union[float, str] = "scale",
        split_criteria: str = "max_samples",
        criterion: str = "gini",
        min_samples_split: int = 0,
        max_features: Optional[Union[str, int, float]] = None,
        splitter: str = "random",
    ):
        self.max_iter = max_iter
        self.C: float = C
        self.kernel: str = kernel
        self.random_state: Optional[int] = random_state
        self.max_depth: Optional[int] = max_depth
        self.tol: float = tol
        self.gamma: Union[float, str] = gamma
        self.degree: int = degree
        self.min_samples_split: int = min_samples_split
        self.split_criteria: str = split_criteria
        self.max_features: Union[str, int, float, None] = max_features
        self.criterion: str = criterion
        self.splitter: str = splitter

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.array = None
    ) -> Stree:
        """Build the tree based on the dataset of samples and its labels

        :param X: dataset of samples to make predictions
        :type X: np.array
        :param y: samples labels
        :type y: np.array
        :param sample_weight: weights of the samples. Rescale C per sample. \
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
        X, y = self._validate_data(X, y)
        sample_weight = _check_sample_weight(
            sample_weight, X, dtype=np.float64
        )
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
    ) -> Optional[Snode]:
        """Recursive function to split the original dataset into predictor
        nodes (leaves)

        :param X: samples dataset
        :type X: np.ndarray
        :param y: samples labels
        :type y: np.ndarray
        :param sample_weight: weight of samples. Rescale C per sample. \
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
        impurity = self.splitter_.impurity(y)
        node = Snode(clf, X, y, features, impurity, title, sample_weight)
        self.depth_ = max(depth, self.depth_)
        self.splitter_.partition(X, node)
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
        node.set_up(
            self.train(  # type: ignore
                X_U, y_u, sw_u, depth + 1, title + " - Up"
            )
        )
        node.set_down(
            self.train(  # type: ignore
                X_D, y_d, sw_d, depth + 1, title + " - Down"
            )
        )
        return node

    def _build_predictor(self) -> None:
        """Process the leaves to make them predictors
        """

        def run_tree(node: Optional[Snode]) -> None:
            if node is None:
                raise ValueError("Can't build predictors on None")
            if node.is_leaf():
                node.make_predictor()
                return
            run_tree(node.get_down())
            run_tree(node.get_up())

        run_tree(self.tree_)

    def _build_clf(self) -> Union[LinearSVC, SVC]:
        """ Build the selected classifier for the node
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
                random_state=self.random_state,
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
        y_ordered = np.zeros(y.shape, dtype=y.dtype)
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
            xp: np.array, indices: np.array, node: Optional[Snode]
        ) -> Tuple[np.array, np.array]:
            """Recursive function to get predictions for every dataset sample

            :param xp: dataset
            :type xp: np.array
            :param indices: indices of the samples in the original dataset
            :type indices: np.array
            :param node: node to process dataset
            :type node: Optional[Snode]
            :return: the predictions and the indices of the predictions
            :rtype: Tuple[np.array, np.array]
            """
            if xp is None:
                return [], []
            if node.is_leaf():  # type: ignore
                # set a class for every sample in dataset
                prediction = np.full(
                    (xp.shape[0], 1), node._class  # type: ignore
                )
                return prediction, indices
            self.splitter_.partition(xp, node)  # type: ignore
            x_u, x_d = self.splitter_.part(xp)
            i_u, i_d = self.splitter_.part(indices)
            prx_u, prin_u = predict_class(
                x_u, i_u, node.get_up()  # type: ignore
            )
            prx_d, prin_d = predict_class(
                x_d, i_d, node.get_down()  # type: ignore
            )
            return np.append(prx_u, prx_d), np.append(prin_u, prin_d)

        check_is_fitted(self, "n_features_in_")
        # Input validation
        X = self._validate_data(X, reset=False)
        # setup prediction & make it happen
        indices = np.arange(X.shape[0])
        result = (
            self._reorder_results(*predict_class(X, indices, self.tree_))
            .astype(int)
            .ravel()
        )
        return self.classes_[result]

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
        """Initializa the value of the max_features atribute used in splits

        :raises ValueError: if not a valid value supplied
        :raises ValueError: if an invalid float value supplied
        :return: Number of features to use in each split
        :rtype: int
        """
        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_in_)))
            else:
                raise ValueError(
                    "Invalid value for max_features. "
                    "Allowed string values are 'auto', "
                    "'sqrt' or 'log2'."
                )
        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(
                    1, int(self.max_features * self.n_features_in_)
                )
            else:
                raise ValueError(
                    "Invalid value for max_features."
                    "Allowed float must be in range (0, 1] "
                    f"got ({self.max_features})"
                )
        return max_features
