Examples
=============

Installation/Usage:
*******************
As the package has not been published on PyPi yet, it CANNOT be install using pip.

For now, the suggested method is to put the file `simpleble.py` in the same directory as your source files and call ``from simpleble import SimpleBleClient, SimpleBleDevice``.

``bluepy`` must also be installed and imported as shown in the example below.
For instructions about how to install, as well as the full documentation of, ``bluepy`` please refer `here <https://github.com/IanHarvey/bluepy/>`_

Search for device, connect and read characteristic
**************************************************
.. code-block:: python

    import time
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    from stree import Stree

    random_state = 1

    X, y = load_iris(return_X_y=True)

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    now = time.time()
    print("Predicting with max_features=sqrt(n_features)")
    clf = Stree(C=0.01, random_state=random_state, max_features="auto")
    clf.fit(Xtrain, ytrain)
    print(f"Took {time.time() - now:.2f} seconds to train")
    print(clf)
    print(f"Classifier's accuracy (train): {clf.score(Xtrain, ytrain):.4f}")
    print(f"Classifier's accuracy (test) : {clf.score(Xtest, ytest):.4f}")
    print("=" * 40)
    print("Predicting with max_features=n_features")
    clf = Stree(C=0.01, random_state=random_state)
    clf.fit(Xtrain, ytrain)
    print(f"Took {time.time() - now:.2f} seconds to train")
    print(clf)
    print(f"Classifier's accuracy (train): {clf.score(Xtrain, ytrain):.4f}")
    print(f"Classifier's accuracy (test) : {clf.score(Xtest, ytest):.4f}")
