# Implementation details

This document provides an overview on how the `conda-libmamba-solver` integrations are implemented,
both within the `conda_libmamba_solver` package itself, and as a `conda` plugin.

## Repository structure

* `.github/workflows/`: CI pipelines to run unit and upstream tests, as well as linting and performance benchmarks.
  Some extra workflows might be added by the `conda/infra` settings.
* `conda_libmamba_solver/`: The Python package. Check sections below for details.
* `recipe/`: The conda-build recipe used for the PR build previews. It should be kept in sync with `conda-forge` and `defaults`.
* `dev/`: Supporting scripts and configuration files to set up development environments.
* `docs/`: Documentation sources.
* `tests/`: Pytest testing infrastructure.
* `pyproject.toml`: Project metadata. See below for details.

## Project metadata

The `pyproject.toml` file stores the required packaging metadata, 
as well as some the configuration for some tools (`black`, `pytest`, etc).

Some peculiarities:

* `flit_core` is the chosen backend for the packaging (as opposed to `setuptools`).
* The `version` is dynamically taken from `conda_libmamba_solver/__init__.py`.
* `black` uses a line length of 99 characters.
* PyTest configurations are extensive but mainly to deal with upstream testing. 
  Check "Development workflows" for details.

## `conda_libmamba_solver` package

The package is a flat namespace:

* `conda_libmamba_solver.__init__`: Defines `__version__` and the old-style plugin API.
* `conda_libmamba_solver.exceptions`: Subclasses of `conda.exceptions`.
* `conda_libmamba_solver.index`: Helper objects to deal with repodata fetching and loading, interfacing with `libmamba` helpers.
* `conda_libmamba_solver.mamba_utils`: Utility functions to help set up the `libmamba` objects.
* `conda_libmamba_solver.models`: Application-agnostic objects to assist in the metadata collection phases.
* `conda_libmamba_solver.solver`: The `conda.core.solve.Solver` subclass with all the libmamba-specific logic.
* `conda_libmamba_solver.state`: Solver-agnostic objects to assist in the solver state specification and collection.
* `conda_libmamba_solver.utils`: Other application-agnostic utility functions.

Refer to each module docstrings for further details!

## Integrations with `conda`

### First iterations

> This is just here as a historical trivia item. Please check the Plugin implementation section for current details!

The first experimental releases of `conda_libmamba_solver` used an ad-hoc mechanism based on `try/except` hooks.

On the `conda/conda` side, we had [`conda.core.solve._get_solver_class()`](https://github.com/conda/conda/blob/22.9.0/conda/core/solve.py#L57-L78):

```python
def _get_solver_class(key=None):
    key = key or conda.base.context.Context.experimental_solver
    if key == "classic":
        return conda.core.solve.Solver  # Classic
    if key.startswith("libmamba"):
        try:
            from conda_libmamba_solver import get_solver_class
            return get_solver_class(key)
        except ImportError as exc:
            raise CondaImportError(...)
    raise ValueError(...)
```

The `key` values were hardcoded in `conda.base.constants`. Not very extensible! 
This was only meant to be temporary as we iterated on the `conda-libmamba-solver` side.
We had one more `get_solver_class()` function in `conda_libmamba_solver` so we could easily change the `Solver` object import path without changing `conda` itself.

The default value for the `key` was set by the `Context` object, which was populated by either:

* The environment variable, `CONDA_EXPERIMENTAL_SOLVER`.
* The command-line flag, `--experimental-solver`.
* A configuration file (e.g. `~/.condarc`).

### With the plugin system

With the plugin system, the `Context` object still provides the solver type to use. 
The attribute is now called simply `solver` (instead of `experimental_solver`), 
and it is populated by the `pluggy` registry.

TODO: WIP.
