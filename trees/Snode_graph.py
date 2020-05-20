'''
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
__version__ = "0.9"
Plot 3D views of nodes in Stree
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from trees.Snode import Snode
from trees.Stree import Stree


class Snode_graph(Snode):

    def __init__(self, node: Stree):
        self._plot_size = (8, 8)
        n = Snode.copy(node)
        super().__init__(n._clf, n._X, n._y, n._title)

    def set_plot_size(self, size):
        self._plot_size = size

    def _is_pure(self) -> bool:
        """is considered pure a leaf node with one label
        """
        if self.is_leaf():
            return self._belief == 1.
        return False

    def plot_hyperplane(self):
        # get the splitting hyperplane
        def hyperplane(x, y): return (-self._interceptor - self._vector[0][0] * x
                                      - self._vector[0][1] * y) / self._vector[0][2]
        fig = plt.figure(figsize=self._plot_size)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        if not self._is_pure():
            # Can't plot hyperplane of leaves with one label because it hasn't classiffier
            tmpx = np.linspace(self._X[:, 0].min(), self._X[:, 0].max())
            tmpy = np.linspace(self._X[:, 1].min(), self._X[:, 1].max())
            xx, yy = np.meshgrid(tmpx, tmpy)
            ax.plot_surface(xx, yy, hyperplane(xx, yy), alpha=.5, antialiased=True,
                            rstride=1, cstride=1, cmap='seismic')
        plt.title(self._title)
        self.plot_distribution(ax)
        return ax

    def plot_distribution(self, ax: Axes3D = None):
        if ax is None:
            fig = plt.figure(figsize=self._plot_size)
            ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(self._X[:, 0], self._X[:, 1], self._X[:, 2], c=self._y)
        ax.set_xlabel('X0')
        ax.set_ylabel('X1')
        ax.set_zlabel('X2')
        plt.show()
