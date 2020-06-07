[![Codeship Status for Doctorado-ML/STree](https://app.codeship.com/projects/8b2bd350-8a1b-0138-5f2c-3ad36f3eb318/status?branch=master)](https://app.codeship.com/projects/399170)
[![codecov](https://codecov.io/gh/doctorado-ml/stree/branch/master/graph/badge.svg)](https://codecov.io/gh/doctorado-ml/stree)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/35fa3dfd53a24a339344b33d9f9f2f3d)](https://www.codacy.com/gh/Doctorado-ML/STree?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Doctorado-ML/STree&amp;utm_campaign=Badge_Grade)

# Stree

Oblique Tree classifier based on SVM nodes. The nodes are built and splitted with sklearn SVC models.Stree is a sklearn estimator and can be integrated in pipelines, grid searches, etc.

![Stree](https://raw.github.com/doctorado-ml/stree/master/example.png)

## Installation

```bash
pip install git+https://github.com/doctorado-ml/stree
```

## Examples

### Jupyter notebooks

##### Slow launch but better integration

* [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Doctorado-ML/STree/master?urlpath=lab/tree/notebooks/test.ipynb) Test notebook

##### Fast launch but have to run first commented out cell for setup

* [![Test](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Doctorado-ML/STree/blob/master/notebooks/test.ipynb) Test notebook

* [![Test2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Doctorado-ML/STree/blob/master/notebooks/test2.ipynb) Another Test notebook

* [![Test Graphics](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Doctorado-ML/STree/blob/master/notebooks/test_graphs.ipynb) Test Graphics notebook

### Command line

```bash
python main.py
```

## Tests

```bash
python -m unittest -v stree.tests
```
