import os
import imghdr
import unittest

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import make_classification

from stree import Stree_grapher, Snode_graph


def get_dataset(random_state=0, n_features=3):
    X, y = make_classification(
        n_samples=1500,
        n_features=n_features,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        class_sep=1.5,
        flip_y=0,
        weights=[0.5, 0.5],
        random_state=random_state,
    )
    return X, y


class Stree_grapher_test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        os.environ["TESTING"] = "1"
        self._random_state = 1
        self._clf = Stree_grapher(
            dict(random_state=self._random_state, use_predictions=False)
        )
        self._clf.fit(*get_dataset(self._random_state, n_features=4))
        super().__init__(*args, **kwargs)

    @classmethod
    def tearDownClass(cls):
        try:
            os.environ.pop("TESTING")
        except KeyError:
            pass

    def test_iterator(self):
        """Check preorder iterator
        """
        expected = [
            "root",
            "root - Down",
            "root - Down - Down, <cgaf> - Leaf class=1 belief= 0.976023 counts"
            "=(array([0, 1]), array([ 17, 692]))",
            "root - Down - Up",
            "root - Down - Up - Down, <cgaf> - Leaf class=0 belief= 0.500000 "
            "counts=(array([0, 1]), array([1, 1]))",
            "root - Down - Up - Up, <cgaf> - Leaf class=0 belief= 0.888889 "
            "counts=(array([0, 1]), array([8, 1]))",
            "root - Up, <cgaf> - Leaf class=0 belief= 0.928205 counts=(array("
            "[0, 1]), array([724,  56]))",
        ]
        computed = []
        for node in self._clf:
            computed.append(str(node))
        self.assertListEqual(expected, computed)

    def test_score(self):
        X, y = get_dataset(self._random_state)
        accuracy_score = self._clf.score(X, y)
        yp = self._clf.predict(X)
        accuracy_computed = np.mean(yp == y)
        self.assertEqual(accuracy_score, accuracy_computed)
        self.assertGreater(accuracy_score, 0.86)

    def test_save_all(self):
        folder_name = "/tmp/"
        file_names = [
            os.path.join(folder_name, f"STnode{i}.png") for i in range(1, 8)
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matplotlib.use("Agg")
            self._clf.save_all(save_folder=folder_name)
        for file_name in file_names:
            self.assertTrue(os.path.exists(file_name))
            self.assertEqual("png", imghdr.what(file_name))
            os.remove(file_name)

    def test_plot_all(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matplotlib.use("Agg")
            num_figures_before = plt.gcf().number
            self._clf.plot_all()
            num_figures_after = plt.gcf().number
        self.assertEqual(7, num_figures_after - num_figures_before)


class Snode_graph_test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        os.environ["TESTING"] = "1"
        self._random_state = 1
        self._clf = Stree_grapher(
            dict(random_state=self._random_state, use_predictions=False)
        )
        self._clf.fit(*get_dataset(self._random_state))
        super().__init__(*args, **kwargs)

    @classmethod
    def tearDownClass(cls):
        """Remove the testing environ variable
        """
        try:
            os.environ.pop("TESTING")
        except KeyError:
            pass

    def test_plot_size(self):
        default = self._clf._tree_gr.get_plot_size()
        expected = (17, 3)
        self._clf._tree_gr.set_plot_size(expected)
        self.assertEqual(expected, self._clf._tree_gr.get_plot_size())
        self._clf._tree_gr.set_plot_size(default)
        self.assertEqual(default, self._clf._tree_gr.get_plot_size())

    def test_attributes_in_leaves_graph(self):
        """Check if the attributes in leaves have correct values so they form a
        predictor
        """

        def check_leave(node: Snode_graph):
            if not node.is_leaf():
                check_leave(node.get_down())
                check_leave(node.get_up())
                return
            # Check Belief in leave
            classes, card = np.unique(node._y, return_counts=True)
            max_card = max(card)
            min_card = min(card)
            if len(classes) > 1:
                try:
                    belief = max_card / (max_card + min_card)
                except ZeroDivisionError:
                    belief = 0.0
            else:
                belief = 1
            self.assertEqual(belief, node._belief)
            # Check Class
            class_computed = classes[card == max_card]
            self.assertEqual(class_computed, node._class)

        check_leave(self._clf._tree_gr)

    def test_nodes_graph_coefs(self):
        """Check if the nodes of the tree have the right attributes filled
        """

        def run_tree(node: Snode_graph):
            if node._belief < 1:
                # only exclude pure leaves
                self.assertIsNotNone(node._clf)
                self.assertIsNotNone(node._clf.coef_)
            if node.is_leaf():
                return
            run_tree(node.get_down())
            run_tree(node.get_up())

        run_tree(self._clf._tree_gr)

    def test_save_hyperplane(self):
        folder_name = "/tmp/"
        file_name = os.path.join(folder_name, "STnode1.png")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matplotlib.use("Agg")
            self._clf._tree_gr.save_hyperplane(folder_name)
        self.assertTrue(os.path.exists(file_name))
        self.assertEqual("png", imghdr.what(file_name))
        os.remove(file_name)

    def test_plot_hyperplane_with_distribution(self):
        plt.close()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matplotlib.use("Agg")
            num_figures_before = plt.gcf().number
            self._clf._tree_gr.plot_hyperplane(plot_distribution=True)
            num_figures_after = plt.gcf().number
        self.assertEqual(1, num_figures_after - num_figures_before)

    def test_plot_hyperplane_without_distribution(self):
        plt.close()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matplotlib.use("Agg")
            num_figures_before = plt.gcf().number
            self._clf._tree_gr.plot_hyperplane(plot_distribution=False)
            num_figures_after = plt.gcf().number
        self.assertEqual(1, num_figures_after - num_figures_before)

    def test_plot_distribution(self):
        plt.close()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matplotlib.use("Agg")
            num_figures_before = plt.gcf().number
            self._clf._tree_gr.plot_distribution()
            num_figures_after = plt.gcf().number
        self.assertEqual(1, num_figures_after - num_figures_before)
