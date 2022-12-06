# Getting started

The `conda-libmamba-solver` plugin allows you to use `libmamba`, the same `libsolv`-powered solver used by `mamba` and `micromamba`, directly in `conda`.

## How to install

`conda-libmamba-solver` is distributed as a separate package, available on both conda-forge and defaults. The plugin needs to be present in the same environment you use `conda` from; most of the time, this is your `base` environment. Run this command:

```bash
$ conda install -n base conda-libmamba-solver
```

### Update from the experimental versions (22.9 and below)

The instructions in this page assume you are using conda-libmamba-solver 22.12 or above.
Please refer to the [v22.12.0 release notes](https://github.com/conda/conda-libmamba-solver/releases/tag/22.12.0) for more details on how to update from a previous version if you were already using the experimental builds.

## Usage

Even if installed, `conda` won't use `conda-libmamba-solver` by default. It will still rely on the `classic` solver.

### Try it once

To enable it for one operation, you can use the `--solver` flag, available for `conda create|install|remove|update` commands.

```
$ conda install tensorflow --solver=libmamba
```

```{note}
The `--solver` flag is also exposed as an environment variable, `CONDA_SOLVER`,
in case you need that.
```

### Set as default

To enable it permanently, you can add `solver: libmamba` to your `.condarc` file, either manually, or with this command:

```
$ conda config --set solver libmamba
```

### Revert to `classic`

If you ever need to use the classic solver temporarily, you can again rely on the `--solver` flag:

```
$ conda install numpy --solver=classic
```

Finally, if you need to revert the default configuration back to `classic`, you can:

* Run `conda config --set solver classic` (to do your choice explicit).
* Run `conda config --remove-key solver` to delete the `solver: libmamba` line from your `.condarc` file.

```{admonition} Tip
If you are unsure what configuration is being used by conda, you can inspect
it with `conda config --show-sources`.
```
