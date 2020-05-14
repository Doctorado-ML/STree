from trees.Stree import Stree
from sklearn.datasets import make_classification

random_state = 1
X, y = make_classification(n_samples=1500, n_features=3, n_informative=3,
                           n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2,
                           class_sep=1.5, flip_y=0, weights=[0.5, 0.5], random_state=random_state)

def load_creditcard(n_examples=0):
    import pandas as pd
    import numpy as np
    import random
    df = pd.read_csv('data/creditcard.csv')
    print("Fraud: {0:.3f}% {1}".format(df.Class[df.Class == 1].count()*100/df.shape[0], df.Class[df.Class == 1].count()))
    print("Valid: {0:.3f}% {1}".format(df.Class[df.Class == 0].count()*100/df.shape[0], df.Class[df.Class == 0].count()))
    y = np.expand_dims(df.Class.values, axis=1)
    X = df.drop(['Class', 'Time', 'Amount'], axis=1).values
    #Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=random_state, stratify=y)
    #return Xtrain, Xtest, ytrain, ytest
    if n_examples > 0:
        # Take first n_examples samples
        X = X[:n_examples, :]
        y = y[:n_examples, :]
    else:
        # Take all the positive samples with a number of random negatives
        if n_examples < 0:
            Xt = X[(y == 1).ravel()]
            yt = y[(y == 1).ravel()]
            indices = random.sample(range(X.shape[0]), -1 * n_examples)
            X = np.append(Xt, X[indices], axis=0)
            y = np.append(yt, y[indices], axis=0)
    print("X.shape", X.shape, " y.shape", y.shape)
    print("Fraud: {0:.3f}% {1}".format(len(y[y == 1])*100/X.shape[0], len(y[y == 1])))
    print("Valid: {0:.3f}% {1}".format(len(y[y == 0])*100/X.shape[0], len(y[y == 0])))
    return X, y
X, y = load_creditcard(-5000)
#X, y = load_creditcard()

clf = Stree(C=.01, max_iter=100, random_state=random_state)
clf.fit(X, y)
print(clf)
#clf.show_tree()
#clf.save_sub_datasets()
yp = clf.predict_proba(X[0, :].reshape(-1, X.shape[1]))
print(f"Predicting {y[0]} we have {yp[0, 0]} with {yp[0, 1]} of belief")
print(f"Classifier's accuracy: {clf.score(X, y, print_out=False):.4f}")
clf.show_tree(only_leaves=True)
