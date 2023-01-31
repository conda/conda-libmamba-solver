# Changelog

All notable changes to this project will be documented in this file.

> The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
> and this project adheres to [calendar versioning](https://calver.org/) in the `YY.M.MICRO`format.

<!--
Populate these categories as PRs are merged to `main`. When a release is cut,
copy to its corresponding section, deleting empty sections if any.
Remember to update the hyperlinks at the bottom.
--->

[//]: # (current developments)

## [22.12.0] - 2022-12-01

### Upgrade notice
To upgrade to `conda-libmamba-solver 22.12.0` please update to `conda 22.11.0` using the "classic" solver first:

```
$ CONDA_EXPERIMENTAL_SOLVER=classic conda install -n base conda=22.11.0
```

and then install a new version of conda-libmamba-solver:

```
$ CONDA_EXPERIMENTAL_SOLVER=classic conda install -n base conda-libmamba-solver=22.12.0
```

Afterwards, please use the new `CONDA_SOLVER` environment variable and ``--solver`` as mentioned below.

### Added

* Added a new documentation site: https://conda.github.io/conda-libmamba-solver/ (#58)
* Added [CEP 4](https://github.com/conda-incubator/ceps/blob/main/cep-4.md) compatible plugin for conda's `solvers` plugin hook. (#63)

### Changed

* The `conda-libmamba-solver` package is now generally available, removes the `experimental` label. (#53)
* The index will also load channels only listed as part the installed packages in the active prefix. (#52)
* Updated compatibility to [mamba 1.0.0](https://github.com/mamba-org/mamba/releases/tag/2022.11.01) and [conda 22.11.0](https://github.com/conda/conda/releases/tag/22.11.0). (#78)

### Deprecated

* Deprecate support for Python 3.6.x.

### Fixed

* Fixed a wrong dependency on libmambapy. (#90)

* If missing or empty, package records will have their `subdir` field populated by the channel platform. (#53)

## [22.8.1] - 2022-08-25

### Fixed

* Amend packaging metadata (#51)

## [22.8.0] - 2022-08-24

### Added

* Check if conda is outdated with `libmamba` instead of relying on conda's implementation (#46)

### Changed

* Rely on conda's `SubdirData` as a fallback for channel protocols not supported by `libmamba` (#49)

### Deprecated

* Deprecate `libmamba-draft` solver variant (#45)

### Removed

* Remove legacy debugging code and file-logging based on stream capture (#48)

## [22.6.0] - 2022-06-28

### Added

* Custom user agent (#29)
* Compatibility with conda-build (#30)

### Changed

* Enable support for user-defined `repodata_fn` while ignoring `current_repodata.json` (#34)
* Faster Python version changes (#33)
* Remove base environment protection (#43)

### Fixed

* Fix libmamba 0.23 compatibility (#35)
* Fix handling of `*`-enabled build strings (#36)
* Fix `escape_channel_url` problems (#32)
* Fix error reporting if S3-backed channels are used (#41)

## [22.3.1] - 2022-03-23

### Fixed

* Make sure `noarch` packages get reinstalled if Python version changed (#26)
* Accept star-only version specs (e.g. `libblas=*=*mkl`) and fix support for `channel::package` syntax (#25)
* Enable support for authenticated channels (#23)

## [22.3.0] - 2022-03-09

_First public release_

## [22.2.0] - 2022-02-01

_Internal pre-release as a separate repository._

<!-- Hyperlinks --->

[Unreleased]: https://github.com/conda/conda-libmamba-solver/compare/22.8.1..main
[22.3.1]: https://github.com/conda/conda-libmamba-solver/releases/tag/22.3.1
[22.3.0]: https://github.com/conda/conda-libmamba-solver/releases/tag/22.3.0
[22.2.0]: https://github.com/conda/conda-libmamba-solver/releases/tag/22.2.0
[22.6.0]: https://github.com/conda/conda-libmamba-solver/releases/tag/22.6.0
[22.8.0]: https://github.com/conda/conda-libmamba-solver/releases/tag/22.8.0
[22.8.1]: https://github.com/conda/conda-libmamba-solver/releases/tag/22.8.1
[22.12.0]: https://github.com/conda/conda-libmamba-solver/releases/tag/22.12.0
