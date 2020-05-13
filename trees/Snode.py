'''
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
__version__ = "0.9"
Node of the Stree (binary tree)
'''

import numpy as np
from sklearn.svm import LinearSVC


class Snode:
    def __init__(self, clf: LinearSVC, X: np.ndarray, y: np.ndarray, title: str):
        self._clf = clf
        self._vector = None if clf is None else clf.coef_
        self._interceptor = 0 if clf is None else clf.intercept_
        self._title = title
        self._belief = 0  # belief of the prediction in a leaf node based on samples
        self._X = X
        self._y = y
        self._down = None
        self._up = None
        self._class = None  # really needed?

    def set_down(self, son):
        self._down = son

    def set_up(self, son):
        self._up = son

    def is_leaf(self,) -> bool:
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
        max_card = max(card)
        min_card = min(card)
        try:
            self._belief = max_card / min_card
        except:
            self._belief = 0
        self._class = classes[card == max_card]

    def __str__(self) -> str:
        if self.is_leaf():
            num = 0
            for i in np.unique(self._y):
                num = max(num, self._y[self._y == i].shape[0])
            den = self._y.shape[0]
            accuracy = num / den if den != 0 else 1
            return f"{self._title} LEAF accuracy={accuracy:.2f}, belief={self._belief:.2f} class={self._class}\n"
        else:
            return f"{self._title}\n"
