import unittest

from trees.Stree import Stree

class Stree_test(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        self.random_state = 17
        self._model = Stree(random_state=self.random_state)
        super(Stree_test, self).__init__(*args, **kwargs)
    
    def test_split_data(self):
        self.assertTrue(True)

