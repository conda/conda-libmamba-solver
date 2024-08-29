# User Guide

The `conda-libmamba-solver` plugin allows you to use `libmamba`, the same `libsolv`-powered solver used by `mamba` and `micromamba`, directly in `conda`.

## How to install

If you have a recent `conda` (23.10 or later), you don't have to do anything. `conda-libmamba-solver` is already preconfigured as default.
For older versions `conda`, we simply recommend updating `conda` to a more recent version:

```console
$ conda update -n base conda
```

If this command fails, check this entry in the FAQ section: {ref}`install-older-conda`.

```{admonition} Update from the experimental versions
:class: note

Please refer to the [v22.12.0 release notes](https://github.com/conda/conda-libmamba-solver/releases/tag/22.12.0) for more details on how to update from a previous version if you were already using the experimental builds (conda-libmamba-solver 22.9 and below).
```

## Usage

From `conda` 23.10, `conda-libmamba-solver` is the default solver. You don't have to do anything else. It will just work.


````{admonition} Usage with conda 23.9 and below
`conda <23.10` won't use `conda-libmamba-solver` by default.
It will still rely on the `classic` solver.

<details>

<summary>Sporadic use</summary>

To enable it for one operation, you can use the `--solver` flag, available for `conda create|install|remove|update` commands.

```
$ conda install tensorflow --solver=libmamba
```

Note: The `--solver` flag is also exposed as an environment variable, `CONDA_SOLVER`,
in case you need that.

</details>

<details>

<summary>Set as default</summary>

To enable it permanently, you can add `solver: libmamba` to your `.condarc` file, either manually, or with this command:

```
$ conda config --set solver libmamba
```

</details>
````

## Revert to `classic`

If you ever need to use the classic solver temporarily, use `--solver` flag:

```
$ conda install numpy --solver=classic
```

Finally, if you need to revert the default configuration back to `classic`, you can:

* Run `conda config --set solver classic` (to make your choice explicit).

```{admonition} Tip
If you are unsure what configuration is being used by conda, you can inspect
it with `conda config --show-sources`.
```

```{toctree}
:hidden:

subcommands
configuration
faq
libmamba-vs-classic
performance
more-resources
```
