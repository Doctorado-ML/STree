from trees.Stree import Stree
from sklearn.datasets import make_classification

random_state = 1
X, y = make_classification(n_samples=1500, n_features=3, n_informative=3,
                           n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2,
                           class_sep=1.5, flip_y=0, weights=[0.5, 0.5], random_state=random_state)
model = Stree(random_state=random_state)
model.fit(X, y)
print(model)
model.save_sub_datasets()
