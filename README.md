![CI](https://github.com/Doctorado-ML/STree/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/doctorado-ml/stree/branch/master/graph/badge.svg)](https://codecov.io/gh/doctorado-ml/stree)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/35fa3dfd53a24a339344b33d9f9f2f3d)](https://www.codacy.com/gh/Doctorado-ML/STree?utm_source=github.com&utm_medium=referral&utm_content=Doctorado-ML/STree&utm_campaign=Badge_Grade)

# Stree

Oblique Tree classifier based on SVM nodes. The nodes are built and splitted with sklearn SVC models. Stree is a sklearn estimator and can be integrated in pipelines, grid searches, etc.

![Stree](https://raw.github.com/doctorado-ml/stree/master/example.png)

## Installation

```bash
pip install git+https://github.com/doctorado-ml/stree
```

## Examples

### Jupyter notebooks

- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Doctorado-ML/STree/master?urlpath=lab/tree/notebooks/benchmark.ipynb) Benchmark

- [![Test](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Doctorado-ML/STree/blob/master/notebooks/benchmark.ipynb) Benchmark

- [![Test2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Doctorado-ML/STree/blob/master/notebooks/features.ipynb) Test features

- [![Adaboost](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Doctorado-ML/STree/blob/master/notebooks/adaboost.ipynb) Adaboost

- [![Gridsearch](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Doctorado-ML/STree/blob/master/notebooks/gridsearch.ipynb) Gridsearch

- [![Test Graphics](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Doctorado-ML/STree/blob/master/notebooks/test_graphs.ipynb) Test Graphics

## Hyperparameters

| **Hyperparameter** | **used<br>in<br>scikit** | **Values**                                     | **Default** | **Meaning**                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------ | ------------------------ | ---------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C                  | Yes                      | <float>                                        | 1.0         | Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.                                                                                                                                                                                                                                                                                                                              |
| kernel             | Yes                      | {"linear", "poly", "rbf"}                      | linear      | Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’ or ‘rbf’.                                                                                                                                                                                                                                                                                                                                                  |
| max_iter           | Yes                      | <int>                                          | 1e5         | Hard limit on iterations within solver, or -1 for no limit.                                                                                                                                                                                                                                                                                                                                                                                          |
| random_state       | Yes                      | <int>                                          | None        | Controls the pseudo random number generation for shuffling the data for probability estimates. Ignored when probability is False.<br>Pass an int for reproducible output across multiple function calls                                                                                                                                                                                                                                              |
| max_depth          | No                       | <int>                                          | None        | Specifies the maximum depth of the tree                                                                                                                                                                                                                                                                                                                                                                                                              |
| tol                | Yes                      | <float>                                        | 1e-4        | Tolerance for stopping criterion.                                                                                                                                                                                                                                                                                                                                                                                                                    |
| degree             | Yes                      | <int>                                          | 3           | Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.                                                                                                                                                                                                                                                                                                                                                                     |
| gamma              | Yes                      | {"scale", "auto"} or <float>                   | scale       | Kernel coefficient for ‘rbf’ and ‘poly’.<br>if gamma='scale' (default) is passed then it uses 1 / (n_features \* X.var()) as value of gamma,<br>if ‘auto’, uses 1 / n_features.                                                                                                                                                                                                                                                                      |
| split_criteria     | No                       | {"impurity", "max_samples"}                    | impurity    | Decides (just in case of a multi class classification) which column (class) use to split the dataset in a node\*\*                                                                                                                                                                                                                                                                                                                                   |
| criterion          | No                       | {“gini”, “entropy”}                            | entropy     | The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.                                                                                                                                                                                                                                                                                                          |
| min_samples_split  | No                       | <int>                                          | 0           | The minimum number of samples required to split an internal node. 0 (default) for any                                                                                                                                                                                                                                                                                                                                                                |
| max_features       | No                       | <int>, <float> <br>or {“auto”, “sqrt”, “log2”} | None        | The number of features to consider when looking for the split:<br>If int, then consider max_features features at each split.<br>If float, then max_features is a fraction and int(max_features \* n_features) features are considered at each split.<br>If “auto”, then max_features=sqrt(n_features).<br>If “sqrt”, then max_features=sqrt(n_features).<br>If “log2”, then max_features=log2(n_features).<br>If None, then max_features=n_features. |
| splitter           | No                       | {"best", "random"}                             |             | The strategy used to choose the feature set at each node (only used if max_features != num_features). Supported strategies are “best” to choose the best feature set and “random” to choose the a random combination. The algorithm generates 5 candidates at most to choose from.                                                                                                                                                                   |

\*\* **Splitting in a STree node**

The decision function is applied to the dataset and distances from samples to hyperplanes are computed in a matrix. This matrix haas as many columns as classes the samples belongs to (if more than two, i.e. multiclass classification) or 1 column if it's a binary class dataset. In binary classification only one hyperplane is computed and therefore only one column is needed to store the distances of the samples to it. If three or more classes are present in the dataset we need as many hyperplanes as classes are there, and therefore one column per hyperplane is needed.

In case of multiclass classification we have to decide which column take into account to make the split, that depends on hyperparameter _split_criteria_, if "impurity" is chosen then STree computes information gain of every split candidate using each column and chooses the one that maximize the information gain, otherwise STree choses the column with more samples with a predicted class (the column with more positive numbers in it).

Once we have the column to take into account for the split, the algorithm splits samples with a positive distance to hyperplane from the rest.

## Tests

```bash
python -m unittest -v stree.tests
```
