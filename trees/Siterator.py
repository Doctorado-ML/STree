'''
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
__version__ = "0.9"
Inorder iterator for the binary tree of Snodes
'''

from trees.Snode import Snode


class Siterator:
    """Stree preorder iterator
    """

    def __init__(self, tree: Snode):
        self._stack = []
        self._push(tree)

    def __iter__(self):
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
