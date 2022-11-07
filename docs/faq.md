# Frequently asked questions

### What's the difference between the available solvers in `conda`?

Please refer to the section "Differences between `libmamba` and `classic`" in
the [`libmamba-vs-classic`](./libmamba-vs-classic.md) docs.


### How do I uninstall it?

If you don't want to use the solver anymore, follow these instructions:

1. If you configured it as the default solver, make sure you revert it with:

```bash
$ conda config --remove-key solver
# You might also need this:
$ conda config --remove-key experimental_solver
```

2. Then, remove the package from `base` with:

```bash
$ conda remove -n base conda-libmamba-solver
```

### How do I configure conda to use the experimental solver permanently?

Use the following command to always use `libmamba` as your default solver:

```bash
$ conda config --set solver libmamba
```

To undo this change permanently, run:

```bash
$ conda config --remove-key solver
# You might also need this:
$ conda config --remove-key experimental_solver
```
