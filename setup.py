import setuptools
import stree


def readme():
    with open("README.md") as f:
        return f.read()


VERSION = stree.__version__
setuptools.setup(
    name="STree",
    version=stree.__version__,
    license=stree.__license__,
    description="Oblique decision tree with svm nodes",
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url="https://github.com/Doctorado-ML/STree#stree",
    project_urls={
        "Code": "https://github.com/Doctorado-ML/STree",
        "Documentation": "https://stree.readthedocs.io/en/latest/index.html",
    },
    author=stree.__author__,
    author_email=stree.__author_email__,
    keywords="scikit-learn oblique-classifier oblique-decision-tree decision-\
    tree svm svc",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: " + stree.__license__,
        "Programming Language :: Python :: 3.8",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    install_requires=["scikit-learn", "numpy", "ipympl"],
    test_suite="stree.tests",
    zip_safe=False,
)
