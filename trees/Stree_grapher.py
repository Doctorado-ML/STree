'''
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
__version__ = "0.9"
Plot 3D views of nodes in Stree
'''

import os
import numpy as np
from sklearn.decomposition import PCA
import trees
import matplotlib.pyplot as plt
from trees.Snode import Snode
from trees.Snode_graph import Snode_graph
from trees.Stree import Stree
from trees.Siterator import Siterator


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
        except:
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
        self._tree_gr = self._copy_tree(self._tree)
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
