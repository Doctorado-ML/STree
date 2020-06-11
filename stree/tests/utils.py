from sklearn.datasets import make_classification


def get_dataset(random_state=0, n_classes=2):
    X, y = make_classification(
        n_samples=1500,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=2,
        class_sep=1.5,
        flip_y=0,
        random_state=random_state,
    )
    return X, y
