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
    def __init__(self, params: dict):
        self._plot_size = (8, 8)
        self._tree_gr = None
        # make Snode store X's
        os.environ['TESTING'] = '1'
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
        if X.shape[1] != 3:
            pca = PCA(n_components=3)
            X = pca.fit_transform(X)
        res = super().fit(X, y)
        self._tree_gr = self._copy_tree(self._tree)
        return res

    def __iter__(self):
        return Siterator(self._tree_gr)
