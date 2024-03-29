# Examples

## Notebooks

- [![benchmark](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Doctorado-ML/STree/blob/master/notebooks/benchmark.ipynb) Benchmark

- [![features](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Doctorado-ML/STree/blob/master/notebooks/features.ipynb) Some features

- [![Gridsearch](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Doctorado-ML/STree/blob/master/notebooks/gridsearch.ipynb) Gridsearch

- [![Ensemble](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Doctorado-ML/STree/blob/master/notebooks/ensemble.ipynb) Ensembles

## Sample Code

```python
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
clf = Stree(random_state=random_state, max_features="auto")
clf.fit(Xtrain, ytrain)
print(f"Took {time.time() - now:.2f} seconds to train")
print(clf)
print(f"Classifier's accuracy (train): {clf.score(Xtrain, ytrain):.4f}")
print(f"Classifier's accuracy (test) : {clf.score(Xtest, ytest):.4f}")
print("=" * 40)
print("Predicting with max_features=n_features")
clf = Stree(random_state=random_state)
clf.fit(Xtrain, ytrain)
print(f"Took {time.time() - now:.2f} seconds to train")
print(clf)
print(f"Classifier's accuracy (train): {clf.score(Xtrain, ytrain):.4f}")
print(f"Classifier's accuracy (test) : {clf.score(Xtest, ytest):.4f}")
```
