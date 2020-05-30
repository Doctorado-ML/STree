'''
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
__version__ = "0.9"
Plot 3D views of nodes in Stree
'''

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from .Strees import Stree, Snode, Siterator


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

    def save_hyperplane(self, save_folder: str = './', save_prefix: str = '',
                        save_seq: int = 1):
        _, fig = self.plot_hyperplane()
        name = f"{save_folder}{save_prefix}STnode{save_seq}.png"
        fig.savefig(name, bbox_inches='tight')
        plt.close(fig)

    def _get_cmap(self):
        cmap = 'jet'
        if self._is_pure() and self._class == 1:
            cmap = 'jet_r'
        return cmap

    def _graph_title(self):
        n_class, card = np.unique(self._y, return_counts=True)
        return f"{self._title} {n_class} {card}"

    def plot_hyperplane(self, plot_distribution: bool = True):
        fig = plt.figure(figsize=self._plot_size)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        if not self._is_pure():
            # Can't plot hyperplane of leaves with one label because it hasn't
            # classiffier
            # get the splitting hyperplane
            def hyperplane(x, y): return (-self._interceptor
                                          - self._vector[0][0] * x
                                          - self._vector[0][1] * y) \
                / self._vector[0][2]

            tmpx = np.linspace(self._X[:, 0].min(), self._X[:, 0].max())
            tmpy = np.linspace(self._X[:, 1].min(), self._X[:, 1].max())
            xx, yy = np.meshgrid(tmpx, tmpy)
            ax.plot_surface(xx, yy, hyperplane(xx, yy), alpha=.5,
                            antialiased=True, rstride=1, cstride=1,
                            cmap='seismic')
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


class Stree_grapher(Stree):
    """Build 3d graphs of any dataset, if it's more than 3 features PCA shall
    make its magic
    """

    def __init__(self, params: dict):
        self._plot_size = (8, 8)
        self._tree_gr = None
        # make Snode store X's
        os.environ['TESTING'] = '1'
        self._fitted = False
        self._pca = None
        super().__init__(**params)

    def __del__(self):
        try:
            os.environ.pop('TESTING')
        except KeyError:
            pass
        plt.close('all')

    def _copy_tree(self, node: Snode) -> Snode_graph:
        mirror = Snode_graph(node)
        # clone node
        mirror._class = node._class
        mirror._belief = node._belief
        if node.get_down() is not None:
            mirror.set_down(self._copy_tree(node.get_down()))
        if node.get_up() is not None:
            mirror.set_up(self._copy_tree(node.get_up()))
        return mirror

    def fit(self, X: np.array, y: np.array) -> Stree:
        """Fit the Stree and copy the tree in a Snode_graph tree

        :param X: Dataset
        :type X: np.array
        :param y: Labels
        :type y: np.array
        :return: Stree model
        :rtype: Stree
        """
        if X.shape[1] != 3:
            self._pca = PCA(n_components=3)
            X = self._pca.fit_transform(X)
        res = super().fit(X, y)
        self._tree_gr = self._copy_tree(self.tree_)
        self._fitted = True
        return res

    def score(self, X: np.array, y: np.array) -> float:
        self._check_fitted()
        if X.shape[1] != 3:
            X = self._pca.transform(X)
        return super().score(X, y)

    def _check_fitted(self):
        if not self._fitted:
            raise Exception('Have to fit the grapher first!')

    def save_all(self, save_folder: str = './', save_prefix: str = ''):
        """Save all the node plots in png format, each with a sequence number

        :param save_folder: folder where the plots are saved, defaults to './'
        :type save_folder: str, optional
        """
        self._check_fitted()
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        seq = 1
        for node in self:
            node.save_hyperplane(save_folder=save_folder,
                                 save_prefix=save_prefix, save_seq=seq)
            seq += 1

    def plot_all(self):
        """Plots all the nodes
        """
        self._check_fitted()
        for node in self:
            node.plot_hyperplane()

    def __iter__(self):
        return Siterator(self._tree_gr)
