# Frequently Asked Questions

## What's the difference between the available solvers in `conda`?

Please refer to the section "Differences between `libmamba` and `classic`" in
the [`libmamba-vs-classic`](./libmamba-vs-classic.md#differences-between-libmamba-and-classic) docs.

## How do I uninstall it?

If you don't want to use the solver anymore, follow these instructions:

```{warning}
Please make sure you __follow all steps below__ to uninstall the solver!
```

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

## How do I configure conda to use the solver permanently?

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

## I get an error when I try to use the `--solver` flag

If you are seeing this error:

```
CondaValueError: Key 'solver' is not a known primitive parameter.
```

It might mean you are using an old version of the conda-libmamba-solver package.
You can check which version is installed with `conda list -n base conda-libmamba-solver`.

Before version 22.12, the CLI flag was `--experimental-solver`.
We recommend you upgrade to `conda` 22.11 or above, and then `conda-libmamba-solver` 22.12 or above.

See the [22.12.0 announcement post](https://github.com/conda/conda-libmamba-solver/releases/tag/22.12.0) for more details on how to upgrade.
