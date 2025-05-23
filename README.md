# STree

![CI](https://github.com/Doctorado-ML/STree/workflows/CI/badge.svg)
[![CodeQL](https://github.com/Doctorado-ML/STree/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/Doctorado-ML/STree/actions/workflows/codeql-analysis.yml)
[![codecov](https://codecov.io/gh/doctorado-ml/stree/branch/master/graph/badge.svg)](https://codecov.io/gh/doctorado-ml/stree)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/35fa3dfd53a24a339344b33d9f9f2f3d)](https://www.codacy.com/gh/Doctorado-ML/STree?utm_source=github.com&utm_medium=referral&utm_content=Doctorado-ML/STree&utm_campaign=Badge_Grade)
[![PyPI version](https://badge.fury.io/py/STree.svg)](https://badge.fury.io/py/STree)
![https://img.shields.io/badge/python-3.11%2B-blue](https://img.shields.io/badge/python-3.11%2B-brightgreen)
[![DOI](https://zenodo.org/badge/262658230.svg)](https://zenodo.org/badge/latestdoi/262658230)

![Stree](https://raw.github.com/doctorado-ml/stree/master/example.png)

Oblique Tree classifier based on SVM nodes. The nodes are built and splitted with sklearn SVC models. Stree is a sklearn estimator and can be integrated in pipelines, grid searches, etc.


## Installation

```bash
pip install Stree
```

## Documentation

Can be found in [stree.readthedocs.io](https://stree.readthedocs.io/en/stable/)

## Examples

### Jupyter notebooks

- [![benchmark](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Doctorado-ML/STree/blob/master/notebooks/benchmark.ipynb) Benchmark

- [![features](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Doctorado-ML/STree/blob/master/notebooks/features.ipynb) Some features

- [![Gridsearch](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Doctorado-ML/STree/blob/master/notebooks/gridsearch.ipynb) Gridsearch

- [![Ensemble](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Doctorado-ML/STree/blob/master/notebooks/ensemble.ipynb) Ensembles

## Hyperparameters

|     | **Hyperparameter**  | **Type/Values**                                                | **Default** | **Meaning**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| --- | ------------------- | -------------------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| \*  | C                   | \<float\>                                                      | 1.0         | Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| \*  | kernel              | {"liblinear", "linear", "poly", "rbf", "sigmoid"}              | linear      | Specifies the kernel type to be used in the algorithm. It must be one of ‘liblinear’, ‘linear’, ‘poly’ or ‘rbf’. liblinear uses [liblinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) library and the rest uses [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) library through scikit-learn library                                                                                                                                                                                                                                                                                                                          |
| \*  | max_iter            | \<int\>                                                        | 1e5         | Hard limit on iterations within solver, or -1 for no limit.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| \*  | random_state        | \<int\>                                                        | None        | Controls the pseudo random number generation for shuffling the data for probability estimates. Ignored when probability is False.<br>Pass an int for reproducible output across multiple function calls                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|     | max_depth           | \<int\>                                                        | None        | Specifies the maximum depth of the tree                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| \*  | tol                 | \<float\>                                                      | 1e-4        | Tolerance for stopping criterion.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| \*  | degree              | \<int\>                                                        | 3           | Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| \*  | gamma               | {"scale", "auto"} or \<float\>                                 | scale       | Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.<br>if gamma='scale' (default) is passed then it uses 1 / (n_features \* X.var()) as value of gamma,<br>if ‘auto’, uses 1 / n_features.                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|     | split_criteria      | {"impurity", "max_samples"}                                    | impurity    | Decides (just in case of a multi class classification) which column (class) use to split the dataset in a node\*\*. max_samples is incompatible with 'ovo' multiclass_strategy                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|     | criterion           | {“gini”, “entropy”}                                            | entropy     | The function to measure the quality of a split (only used if max_features != num_features). <br>Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|     | min_samples_split   | \<int\>                                                        | 0           | The minimum number of samples required to split an internal node. 0 (default) for any                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|     | max_features        | \<int\>, \<float\> <br><br>or {“auto”, “sqrt”, “log2”}         | None        | The number of features to consider when looking for the split:<br>If int, then consider max_features features at each split.<br>If float, then max_features is a fraction and int(max_features \* n_features) features are considered at each split.<br>If “auto”, then max_features=sqrt(n_features).<br>If “sqrt”, then max_features=sqrt(n_features).<br>If “log2”, then max_features=log2(n_features).<br>If None, then max_features=n_features.                                                                                                                                                                                    |
|     | splitter            | {"best", "random", "trandom", "mutual", "cfs", "fcbf", "iwss"} | "random"    | The strategy used to choose the feature set at each node (only used if max_features < num_features).
Supported strategies are: **“best”**: sklearn SelectKBest algorithm is used in every node to choose the max_features best features. **“random”**: The algorithm generates 5 candidates and choose the best (max. info. gain) of them. **“trandom”**: The algorithm generates only one random combination. **"mutual"**: Chooses the best features w.r.t. their mutual info with the label. **"cfs"**: Apply Correlation-based Feature Selection. **"fcbf"**: Apply Fast Correlation-Based Filter. **"iwss"**: IWSS based algorithm |
|     | normalize           | \<bool\>                                                       | False       | If standardization of features should be applied on each node with the samples that reach it                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| \*  | multiclass_strategy | {"ovo", "ovr"}                                                 | "ovo"       | Strategy to use with multiclass datasets, **"ovo"**: one versus one. **"ovr"**: one versus rest                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

\* Hyperparameter used by the support vector classifier of every node

\*\* **Splitting in a STree node**

The decision function is applied to the dataset and distances from samples to hyperplanes are computed in a matrix. This matrix has as many columns as classes the samples belongs to (if more than two, i.e. multiclass classification) or 1 column if it's a binary class dataset. In binary classification only one hyperplane is computed and therefore only one column is needed to store the distances of the samples to it. If three or more classes are present in the dataset we need as many hyperplanes as classes are there, and therefore one column per hyperplane is needed.

In case of multiclass classification we have to decide which column take into account to make the split, that depends on hyperparameter _split_criteria_, if "impurity" is chosen then STree computes information gain of every split candidate using each column and chooses the one that maximize the information gain, otherwise STree choses the column with more samples with a predicted class (the column with more positive numbers in it).

Once we have the column to take into account for the split, the algorithm splits samples with positive distances to hyperplane from the rest.

## Tests

```bash
python -m unittest -v stree.tests
```

## License

STree is [MIT](https://github.com/doctorado-ml/stree/blob/master/LICENSE) licensed

## Reference

R. Montañana, J. A. Gámez, J. M. Puerta, "STree: a single multi-class oblique decision tree based on support vector machines.", 2021 LNAI 12882, pg. 54-64
