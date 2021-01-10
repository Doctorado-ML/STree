import setuptools

__version__ = "0.9rc6"
__author__ = "Ricardo Montañana Gómez"


def readme():
    with open("README.md") as f:
        return f.read()


setuptools.setup(
    name="STree",
    version=__version__,
    license="MIT License",
    description="Oblique decision tree with svm nodes",
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url="https://github.com/doctorado-ml/stree",
    author=__author__,
    author_email="ricardo.montanana@alu.uclm.es",
    keywords="scikit-learn oblique-classifier oblique-decision-tree decision-\
    tree svm svc",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    install_requires=["scikit-learn==0.23.2", "numpy", "ipympl"],
    test_suite="stree.tests",
    zip_safe=False,
)
