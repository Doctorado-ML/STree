from sklearn.datasets import make_classification
import numpy as np


def load_dataset(
    random_state=0, n_classes=2, n_features=3, n_samples=1500, n_informative=3
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=2,
        class_sep=1.5,
        flip_y=0,
        random_state=random_state,
    )
    return X, y


def load_disc_dataset(
    random_state=0, n_classes=2, n_features=3, n_samples=1500
):
    np.random.seed(random_state)
    X = np.random.randint(1, 17, size=(n_samples, n_features)).astype(float)
    y = np.random.randint(low=0, high=n_classes, size=(n_samples), dtype=int)
    return X, y
