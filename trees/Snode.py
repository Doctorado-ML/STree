'''
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
__version__ = "1.0"
Node of the Stree
'''

import numpy as np

class Snode:
    def __init__(self, vector: np.ndarray, interceptor: float, X: np.ndarray, y: np.ndarray, title: str):
        self._vector = vector
        self._interceptor = interceptor
        self._title = title
        self._X = X
        self._y = y
        self._down = None
        self._up = None
        self._class = None
    
    def set_down(self, son):
        self._down = son
    
    def set_up(self, son):
        self._up = son

    def is_leaf(self,) -> bool:
        return self._up is None and self._down is None
    
    def get_down(self):
        return self._down

    def get_up(self):
        return self._up

    def __str__(self):
        if self.is_leaf():
            num = 0
            for i in np.unique(self._y):
                num = max(num, self._y[self._y == i].shape[0])
            den = self._y.shape[0]
            accuracy = num / den if den != 0 else 1       
            return f"{self._title} LEAF accuracy={accuracy:.2f}"
        else:
            return self._title

    
    