import setuptools
import os


def readme():
    with open("README.md") as f:
        return f.read()


def get_data(field):
    item = ""
    file_name = "_version.py" if field == "version" else "__init__.py"
    with open(os.path.join("stree", file_name)) as f:
        for line in f.readlines():
            if line.startswith(f"__{field}__"):
                delim = '"' if '"' in line else "'"
                item = line.split(delim)[1]
                break
        else:
            raise RuntimeError(f"Unable to find {field} string.")
    return item


setuptools.setup(
    name="STree",
    version=get_data("version"),
    license=get_data("license"),
    description="Oblique decision tree with svm nodes",
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url="https://github.com/Doctorado-ML/STree#stree",
    project_urls={
        "Code": "https://github.com/Doctorado-ML/STree",
        "Documentation": "https://stree.readthedocs.io/en/latest/index.html",
    },
    author=get_data("author"),
    author_email=get_data("author_email"),
    keywords="scikit-learn oblique-classifier oblique-decision-tree decision-\
    tree svm svc",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: " + get_data("license"),
        "Programming Language :: Python :: 3.8",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    install_requires=["scikit-learn", "mufs"],
    test_suite="stree.tests",
    zip_safe=False,
)
