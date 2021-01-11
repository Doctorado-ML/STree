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

### Command line

```bash
python main.py
```

## Tests

```bash
python -m unittest -v stree.tests
```
