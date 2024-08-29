# Frequently Asked Questions

## What's the difference between the available solvers in `conda`?

Please refer to the section "Technical differences between `libmamba` and `classic`" in
the [`libmamba-vs-classic`](./libmamba-vs-classic.md#technical-differences-between-libmamba-and-classic) docs.

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

(install-older-conda)=

## I have an older `conda` and I can't install `conda-libmamba-solver`

Since older `conda` versions only supported the `classic` solver, you might run into solver conflicts or too long installations if your `base` environment is too constrained. This becomes a "chicken-and-egg" problem where you'd need `conda-libmamba-solver` to update to a more recent `conda` with `conda-libmamba-solver`.

Fortunately, there's a workaround thanks to the `conda-standalone` project. This is a single binary that bundles recent `conda` versions, with `conda-libmamba-solver` included. It's not a substitute for the full `conda` user experience but it can help bootstrap and rescue conda installations.

1. Download the most recent `conda-standalone` from its [Github Releases page](https://github.com/conda/conda-standalone/releases/latest). Make sure to pick the one for your operating system and platform. Once downloaded, rename it as `conda.exe` on Windows and `conda` on Linux / macOS.
2. Write down the location of your `base` environment: `conda info --root`.
3. Write down the main preconfigured channel in your installation: `conda config --show channels`. This is usually `defaults` or `conda-forge`.
4. Go to the Downloads directory and run this command from your terminal:

On Windows:

```console
.\conda.exe install -p "[path given by step 2]" -c [channel from step 3] "conda>=23.10" conda-libmamba-solver
```

On Linux or macOS:

```console
./conda install -p "[path given by step 2]" -c [channel from step 3] "conda>=23.10" conda-libmamba-solver
```

Once the command succeeds, you'll have `conda-libmamba-solver` installed in your base environment and will be ready to use it as normal. You can delete the conda-standalone binaries.
