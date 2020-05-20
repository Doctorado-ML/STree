import time
from sklearn.model_selection import train_test_split
from trees.Stree import Stree

random_state=1

def load_creditcard(n_examples=0):
    import pandas as pd
    import numpy as np
    import random
    df = pd.read_csv('data/creditcard.csv')
    print("Fraud: {0:.3f}% {1}".format(df.Class[df.Class == 1].count()*100/df.shape[0], df.Class[df.Class == 1].count()))
    print("Valid: {0:.3f}% {1}".format(df.Class[df.Class == 0].count()*100/df.shape[0], df.Class[df.Class == 0].count()))
    y = np.expand_dims(df.Class.values, axis=1)
    X = df.drop(['Class', 'Time', 'Amount'], axis=1).values
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
    print("Valid: {0:.3f}% {1}".format(len(y[y == 0]) * 100 / X.shape[0], len(y[y == 0])))
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=random_state, stratify=y)
    return Xtrain, Xtest, ytrain, ytest

# data = load_creditcard(-5000) # Take all true samples + 5000 of the others
# data = load_creditcard(5000)  # Take the first 5000 samples
data = load_creditcard() # Take all the samples

Xtrain = data[0]
Xtest = data[1]
ytrain = data[2]
ytest = data[3]

now = time.time()
clf = Stree(C=.01, random_state=random_state)
clf.fit(Xtrain, ytrain)
print(f"Took {time.time() - now:.2f} seconds to train")
print(clf)
print(f"Classifier's accuracy (train): {clf.score(Xtrain, ytrain):.4f}")
print(f"Classifier's accuracy (test) : {clf.score(Xtest, ytest):.4f}")
proba = clf.predict_proba(Xtest)
print("Checking that we have correct probabilities, these are probabilities of sample belonging to class 1")
res0 = proba[proba[:, 0] == 0]
res1 = proba[proba[:, 0] == 1]
print("++++++++++res0 > .8++++++++++++")
print(res0[res0[:, 1] > .8])
print("**********res1 < .4************")
print(res1[res1[:, 1] < .4])