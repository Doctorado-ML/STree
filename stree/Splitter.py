"""
Oblique decision tree classifier based on SVM nodes
Splitter class
"""

import os
import warnings
import random
from math import log, factorial
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning
from mufs import MUFS


class Snode:
    """
    Nodes of the tree that keeps the svm classifier and if testing the
    dataset assigned to it

    Parameters
    ----------
    clf : SVC
        Classifier used
    X : np.ndarray
        input dataset in train time (only in testing)
    y : np.ndarray
        input labes in train time
    features : np.array
        features used to compute hyperplane
    impurity : float
        impurity of the node
    title : str
        label describing the route to the node
    weight : np.ndarray, optional
        weights applied to input dataset in train time, by default None
    scaler : StandardScaler, optional
        scaler used if any, by default None
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

    def graph(self):
        """
        Return a string representing the node in graphviz format
        """
        output = ""
        count_values = np.unique(self._y, return_counts=True)
        if self.is_leaf():
            output += (
                f'N{id(self)} [shape=box style=filled label="'
                f"class={self._class} impurity={self._impurity:.3f} "
                f'classes={count_values[0]} samples={count_values[1]}"];\n'
            )
        else:
            output += (
                f'N{id(self)} [label="#features={len(self._features)} '
                f"classes={count_values[0]} samples={count_values[1]} "
                f'({sum(count_values[1])})" fontcolor=black];\n'
            )
            output += f"N{id(self)} -> N{id(self.get_up())} [color=black];\n"
            output += f"N{id(self)} -> N{id(self.get_down())} [color=black];\n"
        return output

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
    """
    Splits a dataset in two based on different criteria

    Parameters
    ----------
    clf : SVC, optional
        classifier, by default None
    criterion : str, optional
        The function to measure the quality of a split (only used if
        max_features != num_features). Supported criteria are “gini” for the
        Gini impurity and “entropy” for the information gain., by default
        "entropy", by default None
    feature_select : str, optional
        The strategy used to choose the feature set at each node (only used if
        max_features < num_features). Supported strategies are: “best”: sklearn
        SelectKBest algorithm is used in every node to choose the max_features
        best features. “random”: The algorithm generates 5 candidates and
        choose the best (max. info. gain) of them. “trandom”: The algorithm
        generates only one random combination. "mutual": Chooses the best
        features w.r.t. their mutual info with the label. "cfs": Apply
        Correlation-based Feature Selection. "fcbf": Apply Fast Correlation-
        Based, by default None
    criteria : str, optional
        ecides (just in case of a multi class classification) which column
        (class) use to split the dataset in a node. max_samples is
        incompatible with 'ovo' multiclass_strategy, by default None
    min_samples_split : int, optional
        The minimum number of samples required to split an internal node. 0
        (default) for any, by default None
    random_state : optional
        Controls the pseudo random number generation for shuffling the data for
        probability estimates. Ignored when probability is False.Pass an int
        for reproducible output across multiple function calls, by
        default None
    normalize : bool, optional
        If standardization of features should be applied on each node with the
        samples that reach it , by default False

    Raises
    ------
    ValueError
        clf has to be a sklearn estimator
    ValueError
        criterion must be gini or entropy
    ValueError
        criteria has to be max_samples or impurity
    ValueError
        splitter must be in {random, best, mutual, cfs, fcbf}
    """

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

        if feature_select not in [
            "random",
            "trandom",
            "best",
            "mutual",
            "cfs",
            "fcbf",
            "iwss",
        ]:
            raise ValueError(
                "splitter must be in {random, trandom, best, mutual, cfs, "
                "fcbf, iwss} "
                f"got ({feature_select})"
            )
        self.criterion_function = getattr(self, f"_{self._criterion}")
        self.decision_criteria = getattr(self, f"_{self._criteria}")
        self.fs_function = getattr(self, f"_fs_{self._feature_select}")

    def _fs_random(
        self, dataset: np.array, labels: np.array, max_features: int
    ) -> tuple:
        """Return the best of five random feature set combinations

        Parameters
        ----------
        dataset : np.array
            array of samples
        labels : np.array
            labels of the dataset
        max_features : int
            number of features of the subspace
            (< number of features in dataset)

        Returns
        -------
        tuple
            indices of the features selected
        """
        # Random feature reduction
        n_features = dataset.shape[1]
        features_sets = self._generate_spaces(n_features, max_features)
        return self._select_best_set(dataset, labels, features_sets)

    @staticmethod
    def _fs_trandom(
        dataset: np.array, labels: np.array, max_features: int
    ) -> tuple:
        """Return the a random feature set combination

        Parameters
        ----------
        dataset : np.array
            array of samples
        labels : np.array
            labels of the dataset
        max_features : int
            number of features of the subspace
            (< number of features in dataset)

        Returns
        -------
        tuple
            indices of the features selected
        """
        # Random feature reduction
        n_features = dataset.shape[1]
        return tuple(sorted(random.sample(range(n_features), max_features)))

    @staticmethod
    def _fs_best(
        dataset: np.array, labels: np.array, max_features: int
    ) -> tuple:
        """Return the variabes with higher f-score

        Parameters
        ----------
        dataset : np.array
            array of samples
        labels : np.array
            labels of the dataset
        max_features : int
            number of features of the subspace
            (< number of features in dataset)

        Returns
        -------
        tuple
            indices of the features selected
        """
        return (
            SelectKBest(k=max_features)
            .fit(dataset, labels)
            .get_support(indices=True)
        )

    def _fs_mutual(
        self, dataset: np.array, labels: np.array, max_features: int
    ) -> tuple:
        """Return the best features with mutual information with labels

        Parameters
        ----------
        dataset : np.array
            array of samples
        labels : np.array
            labels of the dataset
        max_features : int
            number of features of the subspace
            (< number of features in dataset)

        Returns
        -------
        tuple
            indices of the features selected
        """
        # return best features with mutual info with the label
        feature_list = mutual_info_classif(
            dataset, labels, random_state=self._random_state
        )
        return tuple(
            sorted(
                range(len(feature_list)), key=lambda sub: feature_list[sub]
            )[-max_features:]
        )

    @staticmethod
    def _fs_cfs(
        dataset: np.array, labels: np.array, max_features: int
    ) -> tuple:
        """Correlattion-based feature selection with max_features limit

        Parameters
        ----------
        dataset : np.array
            array of samples
        labels : np.array
            labels of the dataset
        max_features : int
            number of features of the subspace
            (< number of features in dataset)

        Returns
        -------
        tuple
            indices of the features selected
        """
        mufs = MUFS(max_features=max_features, discrete=False)
        return mufs.cfs(dataset, labels).get_results()

    @staticmethod
    def _fs_fcbf(
        dataset: np.array, labels: np.array, max_features: int
    ) -> tuple:
        """Fast Correlation-based Filter algorithm with max_features limit

        Parameters
        ----------
        dataset : np.array
            array of samples
        labels : np.array
            labels of the dataset
        max_features : int
            number of features of the subspace
            (< number of features in dataset)

        Returns
        -------
        tuple
            indices of the features selected
        """
        mufs = MUFS(max_features=max_features, discrete=False)
        return mufs.fcbf(dataset, labels, 5e-4).get_results()

    @staticmethod
    def _fs_iwss(
        dataset: np.array, labels: np.array, max_features: int
    ) -> tuple:
        """Correlattion-based feature selection based on iwss with max_features
        limit

        Parameters
        ----------
        dataset : np.array
            array of samples
        labels : np.array
            labels of the dataset
        max_features : int
            number of features of the subspace
            (< number of features in dataset)

        Returns
        -------
        tuple
            indices of the features selected
        """
        mufs = MUFS(max_features=max_features, discrete=False)
        return mufs.iwss(dataset, labels, 0.25).get_results()

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
        n_features = dataset.shape[1]
        if n_features == max_features:
            return tuple(range(n_features))
        # select features as selected in constructor
        return self.fs_function(dataset, labels, max_features)

    def get_subspace(
        self, dataset: np.array, labels: np.array, max_features: int
    ) -> tuple:
        """Re3turn a subspace of the selected dataset of max_features length.
        Depending on hyperparameter

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
