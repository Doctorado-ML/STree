'''
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
__version__ = "0.9"
Inorder iterator for the binary tree of Snodes
Uses LinearSVC
'''

from trees.Snode import Snode


class Siterator:
    """Inorder iterator
    """

    def __init__(self, tree: Snode):
        self._stack = []
        self._push(tree)

    def __iter__(self):
        return self

    def _push(self, node: Snode):
        while (node is not None):
            self._stack.insert(0, node)
            node = node.get_down()

    def __next__(self) -> Snode:
        if len(self._stack) == 0:
            raise StopIteration()
        node = self._stack.pop()
        self._push(node.get_up())
        return node
