# conda-libmamba-solver

The fast mamba solver, now in conda!

## Trying it out

The new libmamba integrations are experimental, but you can get a taste of how they are working
so far by following these instructions.

Before we start: to use the libmamba integrations you need to update the `conda` installation
in your `base` environment to a canary release. This is potentially a destructive action, so
make sure you are not testing this in a production environment. We recommend using a VM, a Docker
image or something similar.

1. Install the canary builds of both `conda` and `conda-libmamba-solver` in your `base` environment.
The following channels are required for now:

```bash
$ conda activate base
$ conda install \
    --override-channels \
    -c conda-canary/label/conda-conda-pr-11193 \
    -c conda-canary/label/conda-libmamba-solver-pr-3 \
    -c conda-forge \
    -c defaults \
    conda \
    conda-libmamba-solver
```

2. Now you can experiment with different things. `--dry-run` is specially useful to check how
different solvers interact. The main switch you need to take care of is the _experimental solver_
option:

```bash
# Using default (classic) solver
$ conda create -n demo scipy --dry-run
# This is equivalent
$ CONDA_EXPERIMENTAL_SOLVER=classic conda create -n demo scipy --dry-run
# Using libmamba integrations
$ CONDA_EXPERIMENTAL_SOLVER=libmamba conda create -n demo scipy --dry-run
# Using old proof-of-concept, debugging-only libmamba integrations
$ CONDA_EXPERIMENTAL_SOLVER=libmamba-draft conda create -n demo scipy --dry-run
```

3. Use `time` to measure how different solvers perform. Take into account that repodata
retrieval is cached across attempts, so only consider timings after warming that up:

```bash
# Warm up the repodata cache
$ conda create -n demo scipy --dry-run
# Timings for original solver
$ time env CONDA_EXPERIMENTAL_SOLVER=classic conda create -n demo scipy --dry-run
# Timings for libmamba integration
$ time env CONDA_EXPERIMENTAL_SOLVER=libmamba conda create -n demo scipy --dry-run
```

> `conda create` commands will have similar performance because it's a very simple action! However,
> things change once you factor in existing environments. Simple commands like `conda install scipy`
> show ~2x speedups already.

4. If you need extra details on _why_ solvers are working in that way, increase verbosity. Output
might get too long for your terminal buffer, so consider using a pager like `less`:

```bash
# Verbosity can be 0, 1, 2 or 3
$ CONDA_VERBOSITY=1 CONDA_EXPERIMENTAL_SOLVER=libmamba conda create -n demo scipy --dry-run  2>&1 | less
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
basis or, even better, on a command by command basis by setting the `CONDA_EXPERIMENTAL_SOLVER`
environment variable when needed.
