"""
Oblique decision tree classifier based on SVM nodes
"""

import os
import numbers
import random
import warnings
from math import log, factorial
from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import check_classification_targets
from sklearn.exceptions import ConvergenceWarning
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

    def __init__(
        self,
        clf: SVC,
        X: np.ndarray,
        y: np.ndarray,
        features: np.array,
        impurity: float,
        title: str,
        weight: np.ndarray = None,
        scaler: StandardScaler = None,
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
        self._scaler = scaler

    @classmethod
    def copy(cls, node: "Snode") -> "Snode":
        return cls(
            node._clf,
            node._X,
            node._y,
            node._features,
            node._impurity,
            node._title,
            node._sample_weight,
            node._scaler,
        )

    def set_partition_column(self, col: int):
        self._partition_column = col

    def get_partition_column(self) -> int:
        return self._partition_column

    def set_down(self, son):
        self._down = son

    def set_title(self, title):
        self._title = title

    def set_classifier(self, clf):
        self._clf = clf

    def set_features(self, features):
        self._features = features

    def set_impurity(self, impurity):
        self._impurity = impurity

    def get_title(self) -> str:
        return self._title

    def get_classifier(self) -> SVC:
        return self._clf

    def get_impurity(self) -> float:
        return self._impurity

    def get_features(self) -> np.array:
        return self._features

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

    def __iter__(self):
        # To complete the iterator interface
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


class Splitter:
    def __init__(
        self,
        clf: SVC = None,
        criterion: str = None,
        feature_select: str = None,
        criteria: str = None,
        min_samples_split: int = None,
        random_state=None,
        normalize=False,
    ):
        self._clf = clf
        self._random_state = random_state
        if random_state is not None:
            random.seed(random_state)
        self._criterion = criterion
        self._min_samples_split = min_samples_split
        self._criteria = criteria
        self._feature_select = feature_select
        self._normalize = normalize

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

        if feature_select not in ["random", "best", "mutual"]:
            raise ValueError(
                "splitter must be in {random, best, mutual} got "
                f"({feature_select})"
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
        """Compute entropy of a labels set

        Parameters
        ----------
        y : np.array
            set of labels

        Returns
        -------
        float
            entropy
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
        """Compute information gain of a split candidate

        Parameters
        ----------
        labels : np.array
            labels of the dataset
        labels_up : np.array
            labels of one side
        labels_dn : np.array
            labels on the other side

        Returns
        -------
        float
            information gain
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
            result = (
                imp_prev
                - (card_up / samples) * imp_up
                - (card_dn / samples) * imp_dn
            )
            return result

    def _select_best_set(
        self, dataset: np.array, labels: np.array, features_sets: list
    ) -> list:
        """Return the best set of features among feature_sets, the criterion is
        the information gain

        Parameters
        ----------
        dataset : np.array
            array of samples (# samples, # features)
        labels : np.array
            array of labels
        features_sets : list
            list of features sets to check

        Returns
        -------
        list
            best feature set
        """
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
        """Generate at most 5 feature random combinations

        Parameters
        ----------
        features : int
            number of features in each combination
        max_features : int
            number of features in dataset

        Returns
        -------
        list
            list with up to 5 combination of features randomly selected
        """
        comb = set()
        # Generate at most 5 combinations
        number = factorial(features) / (
            factorial(max_features) * factorial(features - max_features)
        )
        set_length = min(5, number)
        while len(comb) < set_length:
            comb.add(
                tuple(sorted(random.sample(range(features), max_features)))
            )
        return list(comb)

    def _get_subspaces_set(
        self, dataset: np.array, labels: np.array, max_features: int
    ) -> tuple:
        """Compute the indices of the features selected by splitter depending
        on the self._feature_select hyper parameter

        Parameters
        ----------
        dataset : np.array
            array of samples
        labels : np.array
            labels of the dataset
        max_features : int
            number of features of the subspace
            (<= number of features in dataset)

        Returns
        -------
        tuple
            indices of the features selected
        """
        # No feature reduction
        if dataset.shape[1] == max_features:
            return tuple(range(dataset.shape[1]))
        # Random feature reduction
        if self._feature_select == "random":
            features_sets = self._generate_spaces(
                dataset.shape[1], max_features
            )
            return self._select_best_set(dataset, labels, features_sets)
        # return the KBest features
        if self._feature_select == "best":
            return (
                SelectKBest(k=max_features)
                .fit(dataset, labels)
                .get_support(indices=True)
            )
        # return best features with mutual info with the label
        feature_list = mutual_info_classif(dataset, labels)
        return tuple(
            sorted(
                range(len(feature_list)), key=lambda sub: feature_list[sub]
            )[-max_features:]
        )

    def get_subspace(
        self, dataset: np.array, labels: np.array, max_features: int
    ) -> tuple:
        """Re3turn a subspace of the selected dataset of max_features length.
        Depending on hyperparmeter

        Parameters
        ----------
        dataset : np.array
            array of samples (# samples, # features)
        labels : np.array
            labels of the dataset
        max_features : int
            number of features to form the subspace

        Returns
        -------
        tuple
            tuple with the dataset with only the features selected  and the
            indices of the features selected
        """
        indices = self._get_subspaces_set(dataset, labels, max_features)
        return dataset[:, indices], indices

    def _impurity(self, data: np.array, y: np.array) -> np.array:
        """return column of dataset to be taken into account to split dataset

        Parameters
        ----------
        data : np.array
            distances to hyper plane of every class
        y : np.array
            vector of labels (classes)

        Returns
        -------
        np.array
            column of dataset to be taken into account to split dataset
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

        Parameters
        ----------
        data : np.array
            distances to hyper plane of every class
        y : np.array
            column of dataset to be taken into account to split dataset

        Returns
        -------
        np.array
            column of dataset to be taken into account to split dataset
        """
        # select the class with max number of samples
        _, samples = np.unique(y, return_counts=True)
        return np.argmax(samples)

    def partition(self, samples: np.array, node: Snode, train: bool):
        """Set the criteria to split arrays. Compute the indices of the samples
        that should go to one side of the tree (up)

        Parameters
        ----------
        samples : np.array
            array of samples (# samples, # features)
        node : Snode
            Node of the tree where partition is going to be made
        train : bool
            Train time - True / Test time - False
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
        """Split an array in two based on indices (self._up) and its complement
        partition has to be called first to establish up indices

        Parameters
        ----------
        origin : np.array
            dataset to split

        Returns
        -------
        list
            list with two splits of the array
        """
        down = ~self._up
        return [
            origin[self._up] if any(self._up) else None,
            origin[down] if any(down) else None,
        ]

    def _distances(self, node: Snode, data: np.ndarray) -> np.array:
        """Compute distances of the samples to the hyperplane of the node

        Parameters
        ----------
        node : Snode
            node containing the svm classifier
        data : np.ndarray
            samples to compute distance to hyperplane

        Returns
        -------
        np.array
            array of shape (m, nc) with the distances of every sample to
            the hyperplane of every class. nc = # of classes
        """
        X_transformed = data[:, node._features]
        if self._normalize:
            X_transformed = node._scaler.transform(X_transformed)
        return node._clf.decision_function(X_transformed)


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
        multiclass_strategy: str = "ovo",
        normalize: bool = False,
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
        self.normalize = normalize
        self.multiclass_strategy = multiclass_strategy

    def _more_tags(self) -> dict:
        """Required by sklearn to supply features of the classifier
        make mandatory the labels array

        :return: the tag required
        :rtype: dict
        """
        return {"requires_y": True}

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.array = None
    ) -> "Stree":
        """Build the tree based on the dataset of samples and its labels

        Returns
        -------
        Stree
            itself to be able to chain actions: fit().predict() ...

        Raises
        ------
        ValueError
            if C < 0
        ValueError
            if max_depth < 1
        ValueError
            if all samples have 0 or negative weights
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
        if self.multiclass_strategy not in ["ovr", "ovo"]:
            raise ValueError(
                "mutliclass_strategy has to be either ovr or ovo"
                f" but got {self.multiclass_strategy}"
            )
        if self.multiclass_strategy == "ovo":
            if self.kernel == "liblinear":
                raise ValueError(
                    "The kernel liblinear is incompatible with ovo "
                    "multiclass_strategy"
                )
            if self.split_criteria == "max_samples":
                raise ValueError(
                    "The multiclass_strategy 'ovo' is incompatible with "
                    "split_criteria 'max_samples'"
                )
        kernels = ["liblinear", "linear", "rbf", "poly", "sigmoid"]
        if self.kernel not in kernels:
            raise ValueError(f"Kernel {self.kernel} not in {kernels}")
        check_classification_targets(y)
        X, y = check_X_y(X, y)
        sample_weight = _check_sample_weight(
            sample_weight, X, dtype=np.float64
        )
        if not any(sample_weight):
            raise ValueError(
                "Invalid input - all samples have zero or negative weights."
            )
        check_classification_targets(y)
        # Initialize computed parameters
        self.splitter_ = Splitter(
            clf=self._build_clf(),
            criterion=self.criterion,
            feature_select=self.splitter,
            criteria=self.split_criteria,
            random_state=self.random_state,
            min_samples_split=self.min_samples_split,
            normalize=self.normalize,
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
        self.tree_ = self._train(X, y, sample_weight, 1, "root")
        self.X_ = X
        self.y_ = y
        return self

    def _train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        depth: int,
        title: str,
    ) -> Optional[Snode]:
        """Recursive function to split the original dataset into predictor
        nodes (leaves)

        Parameters
        ----------
        X : np.ndarray
            samples dataset
        y : np.ndarray
            samples labels
        sample_weight : np.ndarray
            weight of samples. Rescale C per sample.
        depth : int
            actual depth in the tree
        title : str
            description of the node

        Returns
        -------
        Optional[Snode]
            binary tree
        """
        if depth > self.__max_depth:
            return None
        # Mask samples with 0 weight
        if any(sample_weight == 0):
            indices_zero = sample_weight == 0
            X = X[~indices_zero, :]
            y = y[~indices_zero]
            sample_weight = sample_weight[~indices_zero]
        self.depth_ = max(depth, self.depth_)
        scaler = StandardScaler()
        node = Snode(None, X, y, X.shape[1], 0.0, title, sample_weight, scaler)
        if np.unique(y).shape[0] == 1:
            # only 1 class => pure dataset
            node.set_title(title + ", <pure>")
            node.make_predictor()
            return node
        # Train the model
        clf = self._build_clf()
        Xs, features = self.splitter_.get_subspace(X, y, self.max_features_)
        if self.normalize:
            scaler.fit(Xs)
            Xs = scaler.transform(Xs)
        clf.fit(Xs, y, sample_weight=sample_weight)
        node.set_impurity(self.splitter_.partition_impurity(y))
        node.set_classifier(clf)
        node.set_features(features)
        self.splitter_.partition(X, node, True)
        X_U, X_D = self.splitter_.part(X)
        y_u, y_d = self.splitter_.part(y)
        sw_u, sw_d = self.splitter_.part(sample_weight)
        if X_U is None or X_D is None:
            # didn't part anything
            node.set_title(title + ", <cgaf>")
            node.make_predictor()
            return node
        node.set_up(
            self._train(X_U, y_u, sw_u, depth + 1, title + f" - Up({depth+1})")
        )
        node.set_down(
            self._train(
                X_D, y_d, sw_d, depth + 1, title + f" - Down({depth+1})"
            )
        )
        return node

    def _build_clf(self):
        """Build the right classifier for the node"""
        return (
            LinearSVC(
                max_iter=self.max_iter,
                random_state=self.random_state,
                C=self.C,
                tol=self.tol,
            )
            if self.kernel == "liblinear"
            else SVC(
                kernel=self.kernel,
                max_iter=self.max_iter,
                tol=self.tol,
                C=self.C,
                gamma=self.gamma,
                degree=self.degree,
                random_state=self.random_state,
                decision_function_shape=self.multiclass_strategy,
            )
        )

    @staticmethod
    def _reorder_results(y: np.array, indices: np.array) -> np.array:
        """Reorder an array based on the array of indices passed

        Parameters
        ----------
        y : np.array
            data untidy
        indices : np.array
            indices used to set order

        Returns
        -------
        np.array
            array y ordered
        """
        # return array of same type given in y
        y_ordered = y.copy()
        indices = indices.astype(int)
        for i, index in enumerate(indices):
            y_ordered[index] = y[i]
        return y_ordered

    def predict(self, X: np.array) -> np.array:
        """Predict labels for each sample in dataset passed

        Parameters
        ----------
        X : np.array
            dataset of samples

        Returns
        -------
        np.array
            array of labels

        Raises
        ------
        ValueError
            if dataset with inconsistent number of features
        NotFittedError
            if model is not fitted
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

    def nodes_leaves(self) -> tuple:
        """Compute the number of nodes and leaves in the built tree

        Returns
        -------
        [tuple]
            tuple with the number of nodes and the number of leaves
        """
        nodes = 0
        leaves = 0
        for node in self:
            nodes += 1
            if node.is_leaf():
                leaves += 1
        return nodes, leaves

    def __iter__(self) -> Siterator:
        """Create an iterator to be able to visit the nodes of the tree in
        preorder, can make a list with all the nodes in preorder

        Returns
        -------
        Siterator
            an iterator, can for i in... and list(...)
        """
        try:
            tree = self.tree_
        except AttributeError:
            tree = None
        return Siterator(tree)

    def __str__(self) -> str:
        """String representation of the tree

        Returns
        -------
        str
            description of nodes in the tree in preorder
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
            if self.max_features > self.n_features_:
                raise ValueError(
                    "Invalid value for max_features. "
                    "It can not be greater than number of features "
                    f"({self.n_features_})"
                )
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
