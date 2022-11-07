# conda-libmamba-solver

[![Anaconda-Server Badge](https://anaconda.org/main/conda-libmamba-solver/badges/version.svg)](https://anaconda.org/main/conda-libmamba-solver)
[![Anaconda-Server Badge](https://anaconda.org/main/conda-libmamba-solver/badges/latest_release_date.svg)](https://anaconda.org/main/conda-libmamba-solver)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/conda-incubator/conda-libmamba-solver/CI?label=CI&logo=github&style=flat-square)](https://github.com/conda-incubator/conda-libmamba-solver/actions/workflows/ci.yml)
[![CalVer version used YY.MM.MICRO](https://img.shields.io/badge/calver-YY.MM.MICRO-22bfda.svg?style=flat-square)](https://calver.org)

The fast mamba solver, now in conda!

## What is this exactly?

conda-libmamba-solver is a new (experimental) solver for the
[conda package manager](https://docs.conda.io/) which uses the solver from the
[mamba project](https://mamba.readthedocs.io/) behind the scenes, while
carefully implementing conda's functionality and expected behaviors on top.
The library used by mamba to do the heavy-lifting is called [libsolv](https://github.com/openSUSE/libsolv).

Additional information about the project can be found in the blog post on Anaconda's weblog:
[A Faster Solver for Conda: Libmamba](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community).

## Documentation

Check the [documentation](./docs/README.md) for instructions on how to install, use and make the most out our new solver!
