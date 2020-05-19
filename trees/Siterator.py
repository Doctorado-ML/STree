
from trees.Snode import Snode

class Siterator:
    """Implements an inorder iterator
    """
    def __init__(self, tree: Snode):
        self._stack = []
        self._push(tree)
    
    def hasNext(self) -> bool: 
        return len(self._stack) > 0

    def _push(self, node: Snode):
        while (node is not None):
            self._stack.insert(0, node)
            node = node.get_down()

    def next(self) -> Snode:
        node = self._stack.pop()
        self._push(node.get_up())
        return node
