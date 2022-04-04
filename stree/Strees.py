"""
Oblique decision tree classifier based on SVM nodes
"""

import numbers
import random
from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    _check_sample_weight,
)
from .Splitter import Splitter, Snode, Siterator
from ._version import __version__


class Stree(BaseEstimator, ClassifierMixin):
    """
    Estimator that is based on binary trees of svm nodes
    can deal with sample_weights in predict, used in boosting sklearn methods
    inheriting from BaseEstimator implements get_params and set_params methods
    inheriting from ClassifierMixin implement the attribute _estimator_type
    with "classifier" as value

    Parameters
    ----------
    C : float, optional
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive., by default 1.0
    kernel : str, optional
        Specifies the kernel type to be used in the algorithm. It must be one
        of ‘liblinear’, ‘linear’, ‘poly’ or ‘rbf’. liblinear uses
        [liblinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) library and
        the rest uses [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
        library through scikit-learn library, by default "linear"
    max_iter : int, optional
        Hard limit on iterations within solver, or -1 for no limit., by default
        1e5
    random_state : int, optional
        Controls the pseudo random number generation for shuffling the data for
        probability estimates. Ignored when probability is False.Pass an int
        for reproducible output across multiple function calls, by
        default None
    max_depth : int, optional
        Specifies the maximum depth of the tree, by default None
    tol : float, optional
        Tolerance for stopping, by default 1e-4
    degree : int, optional
        Degree of the polynomial kernel function (‘poly’). Ignored by all other
        kernels., by default 3
    gamma : str, optional
        Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.if gamma='scale'
        (default) is passed then it uses 1 / (n_features * X.var()) as value
        of gamma,if ‘auto’, uses 1 / n_features., by default "scale"
    split_criteria : str, optional
        Decides (just in case of a multi class classification) which column
        (class) use to split the dataset in a node. max_samples is
        incompatible with 'ovo' multiclass_strategy, by default "impurity"
    criterion : str, optional
        The function to measure the quality of a split (only used if
        max_features != num_features). Supported criteria are “gini” for the
        Gini impurity and “entropy” for the information gain., by default
        "entropy"
    min_samples_split : int, optional
        The minimum number of samples required to split an internal node. 0
        (default) for any, by default 0
    max_features : optional
        The number of features to consider when looking for the split: If int,
        then consider max_features features at each split. If float, then
        max_features is a fraction and int(max_features * n_features) features
        are considered at each split. If “auto”, then max_features=
        sqrt(n_features). If “sqrt”, then max_features=sqrt(n_features). If
        “log2”, then max_features=log2(n_features). If None, then max_features=
        n_features., by default None
    splitter : str, optional
        The strategy used to choose the feature set at each node (only used if
        max_features < num_features). Supported strategies are: “best”: sklearn
        SelectKBest algorithm is used in every node to choose the max_features
        best features. “random”: The algorithm generates 5 candidates and
        choose the best (max. info. gain) of them. “trandom”: The algorithm
        generates only one random combination. "mutual": Chooses the best
        features w.r.t. their mutual info with the label. "cfs": Apply
        Correlation-based Feature Selection. "fcbf": Apply Fast Correlation-
        Based , by default "random"
    multiclass_strategy : str, optional
        Strategy to use with multiclass datasets, "ovo": one versus one. "ovr":
        one versus rest, by default "ovo"
    normalize : bool, optional
        If standardization of features should be applied on each node with the
        samples that reach it , by default False

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes

    n_iter_ : int
        Max number of iterations in classifier

    depth_ : int
        Max depht of the tree

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    max_features_ : int
        Number of features to use in hyperplane computation

    tree_ : Node
        root of the tree

    X_ : ndarray
        points to the input dataset

    y_ : ndarray
        points to the input labels

    References
    ----------
    R. Montañana, J. A. Gámez, J. M. Puerta, "STree: a single multi-class
    oblique decision tree based on support vector machines.", 2021 LNAI 12882


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

    @staticmethod
    def version() -> str:
        """Return the version of the package."""
        return __version__

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

    def graph(self, title="") -> str:
        """Graphviz code representing the tree

        Returns
        -------
        str
            graphviz code
        """
        output = (
            "digraph STree {\nlabel=<STree "
            f"{title}>\nfontsize=30\nfontcolor=blue\nlabelloc=t\n"
        )
        for node in self:
            output += node.graph()
        output += "}\n"
        return output

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
