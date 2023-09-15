# conda-libmamba-solver

The fast mamba solver, now in conda!

## What is this exactly?

conda-libmamba-solver is a new solver for the
[conda package manager](https://docs.conda.io/) which uses the solver from the
[mamba project](https://mamba.readthedocs.io/) behind the scenes, while
carefully implementing conda's functionality and expected behaviors on top.
The library used by mamba to do the heavy-lifting is called [libsolv](https://github.com/openSUSE/libsolv).

Additional information about the project can be found in the blog post on Anaconda's weblog:
[A Faster Solver for Conda: Libmamba](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community).

## Documentation

Check the [documentation](https://conda.github.io/conda-libmamba-solver/) for
instructions on how to install, use and make the most out the new conda solver!


## Build status

| [![Build status](https://github.com/conda/conda-libmamba-solver/actions/workflows/tests.yml/badge.svg)](https://github.com/conda/conda-libmamba-solver/actions/workflows/tests.yml?query=branch%3Amain) [![Docs status](https://github.com/conda/conda-libmamba-solver/actions/workflows/docs.yml/badge.svg)](https://github.com/conda/conda-libmamba-solver/actions/workflows/docs.yml?query=branch%3Amain) [![codecov](https://codecov.io/gh/conda/conda-libmamba-solver/branch/main/graph/badge.svg)](https://codecov.io/gh/conda/conda-libmamba-solver) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/conda/conda-libmamba-solver/main.svg)](https://results.pre-commit.ci/latest/github/conda/conda-libmamba-solver/main) [![CalVer version used YY.MM.MICRO](https://img.shields.io/badge/calver-YY.MM.MICRO-22bfda.svg?style=flat-square)](https://calver.org) | [![Anaconda-Server Badge](https://anaconda.org/conda-canary/conda-libmamba-solver/badges/latest_release_date.svg)](https://anaconda.org/conda-canary/conda-libmamba-solver) |
| --- | :-: |
| [`conda install defaults::conda-libmamba-solver`](https://anaconda.org/anaconda/conda-libmamba-solver) | [![Anaconda-Server Badge](https://anaconda.org/anaconda/conda-libmamba-solver/badges/version.svg)](https://anaconda.org/anaconda/conda-libmamba-solver) |
| [`conda install conda-forge::conda-libmamba-solver`](https://anaconda.org/conda-forge/conda-libmamba-solver) | [![Anaconda-Server Badge](https://anaconda.org/conda-forge/conda-libmamba-solver/badges/version.svg)](https://anaconda.org/conda-forge/conda-libmamba-solver) |
| [`conda install conda-canary/label/dev::conda-libmamba-solver`](https://anaconda.org/conda-canary/conda-libmamba-solver) | [![Anaconda-Server Badge](https://anaconda.org/conda-canary/conda-libmamba-solver/badges/version.svg)](https://anaconda.org/conda-canary/conda-libmamba-solver) |
