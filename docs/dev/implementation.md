# Implementation details

This document provides an overview on how the `conda-libmamba-solver` integrations are implemented,
both within the `conda_libmamba_solver` package itself, and as a `conda` plugin.

## Repository structure

* `.github/workflows/`: CI pipelines to run unit and upstream tests, as well as linting and performance benchmarks. Some extra workflows might be added by the `conda/infra` settings.
* `conda_libmamba_solver/`: The Python package. Check sections below for details.
* `recipe/`: The conda-build recipe used for the PR build previews. It should be kept in sync with `conda-forge` and `defaults`.
* `dev/`: Supporting scripts and configuration files to set up development environments.
* `docs/`: Documentation sources.
* `tests/`: pytest testing infrastructure.
* `pyproject.toml`: Project metadata. See below for details.

## Project metadata

The `pyproject.toml` file stores the required packaging metadata,
as well as some the configuration for some tools (`black`, `pytest`, etc.).

Some peculiarities:

* `flit_core` is the chosen backend for the packaging (as opposed to `setuptools`).
* The `version` is dynamically taken from `conda_libmamba_solver/__init__.py`.
* `black` uses a line length of 99 characters.
* Pytest configurations are extensive but are only necessary when dealing with upstream testing.
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

```{note}
Refer to each module docstrings for further details!
```

### Solver-agnostic parts

The idea behind the module separation is to have better logic reusability and separation between the `libmamba` library and the preparation logic `conda` uses.
The following paragraphs assume you have read the [Deep Dive guides](https://docs.conda.io/projects/conda/en/stable/dev-guide/deep-dive-install.html) in the `conda` documentation, but as a refresher:

* The `Solver` class will take a list of `MatchSpec` objects coming from diverse sources, like:
  * The packages the user requested in the command line or an environment file
  * The installed packages in the target environment
  * The explicitly requested packages in previous commands (the history)
  * ... and some more coming from configured settings
* All these `MatchSpecs` are then merged and sorted depending on certain preparation logic.
* The final set of `MatchSpec` objects is used to _query_ the so-called _index_ of packages.
* The index is a long list of `PackageRecord` objects, loaded from the `repodata.json` files obtained from the configured channels.
* The job of the solver is to find the set of `PackageRecord` objects that better satisfies the optimization criteria for the given `MatchSpec` objects.

All the preparation steps before the actual SAT solver starts working are conda-specific, and solver-agnostic!
In `conda` classic, this logic is spread across the different layers, but in `conda-libmamba-solver` we tried to synthesize it in a single module.
This is the `conda_libmamba_solver.state` module, which contains the `SolverInputState` and `SolverOutputState` classes.

* `SolverInputState` deals with the collection and management of the `MatchSpec` objects.
* `SolverOutputState` will assist the `Solver` class maintain its state through the different solving attempts,
  and will finally export a list of `PackageRecords`.
  The early exit and post-solve logics are also expressed here.

Both `SolverInputState` and `SolverOutputState` classes are supported by the `TrackedMap` dictionary subclass,
which logs its own changes for better debugging and developer experience while analyzing solver problems.

### libmamba-specific parts

`conda_libmamba_solver` interfaces with `libmamba` objects through three modules only:

* `.solver`, which contains the `conda.core.solve.Solver` subclass.
  It relies heavily on `conda_libmamba_solver.state` in an effort to only contain the logic necessary to interface with `libmamba`.
* `.index`, which deals with the repodata fetching and loading.
  Initially, it invoked the necessary `libmamba` objects to download and load the repodata JSON files.
  In later releases, downloading is done with `conda` objects, and we then pass the JSON files to the `libmamba` loaders.
* `.mamba_utils`, which contains utility functions borrowed and adapted from `mamba` itself.
  Its main usage is the initialization of the `libmamba.Context` options from `conda`'s `Context`.

## Integrations with `conda`

### First iterations

```{note}
This is just here as a historical trivia item.
Please check the Plugin implementation section for current details!
```

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

The `key` values were hard-coded in `conda.base.constants`. Not very extensible!
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
