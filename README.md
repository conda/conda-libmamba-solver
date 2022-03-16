# conda-libmamba-solver

[![Anaconda-Server Badge](https://anaconda.org/conda-canary/conda-libmamba-solver/badges/version.svg)](https://anaconda.org/conda-canary/conda-libmamba-solver)
[![Anaconda-Server Badge](https://anaconda.org/conda-canary/conda-libmamba-solver/badges/latest_release_date.svg)](https://anaconda.org/conda-canary/conda-libmamba-solver)
[![CalVer version used YY.MM.MICRO](https://img.shields.io/badge/calver-YY.MM.MICRO-22bfda.svg?style=flat-square)](https://calver.org)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/conda-incubator/conda-libmamba-solver/CI?label=CI&logo=github&style=flat-square)](https://github.com/conda-incubator/conda-libmamba-solver/actions/workflows/ci.yml)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/conda-incubator/conda-libmamba-solver/Canary%20builds?label=Canary%20builds&logo=github&style=flat-square)](https://github.com/conda-incubator/conda-libmamba-solver/actions/workflows/builds-canary.yaml)

The fast mamba solver, now in conda!

## What is this exactly?

conda-libmamba-solver is a new (experimental) solver for the
[conda package manager](https://docs.conda.io/) which uses the solver from the
[mamba project](https://mamba.readthedocs.io/) behind the scenes, while
carefully implementing conda's functionality and expected behaviors on top.
The library used by mamba to do the heavy-lifting is called [libsolv](https://github.com/openSUSE/libsolv).

## Trying it out

The new libmamba integrations are experimental, but you can get a taste of how they are working
so far by following these instructions.

Before we start: to use the libmamba integrations you need to update the `conda` installation
in your `base` environment to a canary release. This is potentially a destructive action, so
make sure you are not testing this in a production environment. We recommend using a VM, a Docker
image or something similar.

1. `conda-libmamba-solver` needs to be present in your `base` environment.

First make sure you are running conda 4.12.0 or higher:
```bash
conda update -n base conda
```

Then install conda-libmamba-solver:
```bash
conda install -n base conda-libmamba-solver
```

2. Now you can experiment with different things. `--dry-run` is specially useful to check how
different solvers interact. The main switch you need to take care of is the _experimental solver_
option:

```bash
# Using default (classic) solver
$ conda create -n demo scipy --dry-run
# This is equivalent
$ conda create -n demo scipy --dry-run --experimental-solver=classic
# Using libmamba integrations
$ conda create -n demo scipy --dry-run --experimental-solver=libmamba
# Using old proof-of-concept, debugging-only libmamba integrations
$ conda create -n demo scipy --dry-run --experimental-solver=libmamba-draft
```

> Hint: You can also enable the experimental solver with the `CONDA_EXPERIMENTAL_SOLVER`
> environment variable: `CONDA_EXPERIMENTAL_SOLVER=libmamba conda install ...`

3. Use `time` to measure how different solvers perform. Take into account that repodata
retrieval is cached across attempts, so only consider timings after warming that up:

```bash
# Warm up the repodata cache
$ conda create -n demo scipy --dry-run
# Timings for original solver
$ time conda create -n demo scipy --dry-run --experimental-solver=classic
# Timings for libmamba integrations
$ time conda create -n demo scipy --dry-run --experimental-solver=libmamba
```

> `conda create` commands will have similar performance because it's a very simple action! However,
> things change once you factor in existing environments. Simple commands like `conda install scipy`
> show ~2x speedups already.

4. If you need extra details on _why_ solvers are working in that way, increase verbosity. Output
might get too long for your terminal buffer, so consider using a pager like `less`:

```bash
# Verbosity can be expressed with 1, 2 or 3 `v`
$ conda create -n demo scipy --dry-run -vvv --experimental-solver=libmamba  2>&1 | less
```

## FAQ

### How do I uninstall it?

If you don't want to use the experimental solver anymore, you can uninstall it with:

```
$ conda remove conda-libmamba-solver
```

To revert `conda` to the stable version:

```
$ conda install -c defaults conda
```

### Why can't I use the experimental solver on the `base` environment?

This decision has been made to protect your `base` installation from unexpected changes. This
package is still in a experimental phase and, as a result, you can only use it in non-base
environments for now.

### How do I configure conda to use the experimental solver permanently?

Use the following command:

```
$ conda config --set experimental_solver libmamba --env
```

Note that we are using the `--env` flag so the setting is only applied to the active
environment. Otherwise it will have a global effect on all your environments, including `base`,
which is now protected. As such, we strongly recommend to enable this setting in a case by case
basis or, even better, on a command by command basis by setting the corresponding command line flags
or environment variables when needed.
