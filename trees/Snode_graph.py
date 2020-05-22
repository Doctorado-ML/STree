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
        self._xlimits = (None, None)
        self._ylimits = (None, None)
        self._zlimits = (None, None)
        n = Snode.copy(node)
        super().__init__(n._clf, n._X, n._y, n._title)

    def set_plot_size(self, size: tuple):
        self._plot_size = size

    def _is_pure(self) -> bool:
        """is considered pure a leaf node with one label
        """
        if self.is_leaf():
            return self._belief == 1.
        return False

    def set_axis_limits(self, limits: tuple):
        self._xlimits = limits[0]
        self._ylimits = limits[1]
        self._zlimits = limits[2]

    def _set_graphics_axis(self, ax: Axes3D):
        ax.set_xlim(self._xlimits)
        ax.set_ylim(self._ylimits)
        ax.set_zlim(self._zlimits)

    def save_hyperplane(self, save_folder: str = './', save_prefix: str = '', save_seq: int = 1):
        _, fig = self.plot_hyperplane()
        name = f"{save_folder}{save_prefix}STnode{save_seq}.png"
        fig.savefig(name, bbox_inches='tight')
        plt.close(fig)

    def _get_cmap(self):
        cmap = 'jet'
        if self._is_pure():
            if self._class == 1:
                cmap = 'jet_r'
        return cmap

    def _graph_title(self):
        n_class, card = np.unique(self._y, return_counts=True)
        return f"{self._title} {n_class} {card}"

    def plot_hyperplane(self, plot_distribution: bool = True):
        fig = plt.figure(figsize=self._plot_size)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        if not self._is_pure():
            # Can't plot hyperplane of leaves with one label because it hasn't classiffier
            # get the splitting hyperplane
            def hyperplane(x, y): return (-self._interceptor - self._vector[0][0] * x
                                          - self._vector[0][1] * y) / self._vector[0][2]
            tmpx = np.linspace(self._X[:, 0].min(), self._X[:, 0].max())
            tmpy = np.linspace(self._X[:, 1].min(), self._X[:, 1].max())
            xx, yy = np.meshgrid(tmpx, tmpy)
            ax.plot_surface(xx, yy, hyperplane(xx, yy), alpha=.5, antialiased=True,
                            rstride=1, cstride=1, cmap='seismic')
            self._set_graphics_axis(ax)
        if plot_distribution:
            self.plot_distribution(ax)
        else:
            plt.title(self._graph_title())
            plt.show()
        return ax, fig

    def plot_distribution(self, ax: Axes3D = None):
        if ax is None:
            fig = plt.figure(figsize=self._plot_size)
            ax = fig.add_subplot(1, 1, 1, projection='3d')
        plt.title(self._graph_title())
        cmap = self._get_cmap()
        ax.scatter(self._X[:, 0], self._X[:, 1],
                   self._X[:, 2], c=self._y, cmap=cmap)
        ax.set_xlabel('X0')
        ax.set_ylabel('X1')
        ax.set_zlabel('X2')
        plt.show()
