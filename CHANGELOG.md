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

## 23.9.0 (2023-09-28)

### Enhancements

* Expose libmamba's `repoquery` search features as a conda subcommand plugin. (#258)
* Rewrite how we create tasks for `libsolv`, making use of `libmamba`'s `add_pin` features. (#270, #288)
* Name-only pins will lock the corresponding package if installed. ([conda#13031](https://github.com/conda/conda/pull/13031) via #289)
* Use the `.solv` cache for repodata if available and recent. (#295)

### Bug fixes

* Handle commands with no channels passed gracefully. (#256)
* Workaround for missing `noarch` field in returned `PackageRecord` payload. (#257)
* Port logic from [conda/conda#9614](https://github.com/conda/conda/pull/9614), which fixes
  a bug where the `--prune` flag was not working correctly in `conda env update` commands.
  (#270)
* Ensure environments are not aggressively updated to higher priority channels under some conditions. (#240 via #270, #285)
* Do not inject those channels from installed packages that do not exist or are unavailable. (#262 via #274)
* Correctly print all configured channels in `PackagesNotFoundError` exceptions. (#284)
* Do not crash if a `MatchSpec` with a build string is specified in the CLI and there's a pinned spec for the same package name. (#286 via #289)
* Only apply `defaults::pkg` workarounds for the default value `default_channels`. (#292)

### Deprecations

* Users won't be able to override pinned specs with incompatible CLI specs anymore. Instead they must modify their pinned specs explicitly. ([conda#9016](https://github.com/conda/conda/issues/9016) via #289, #294)

### Docs

* Document intentional deviations from conda's `classic` solver behavior. (#289)

### Other

* Explain why `defaults::pkg_name` is broken libmamba 1.5.0 ([details](https://github.com/mamba-org/mamba/issues/2431)). (#266)



## 23.7.0 (2023-07-31)

### Enhancements

* Add basic integrations for local channels in conda-build. (#194)

### Bug fixes

* Add backwards-compatible support for the new namespaced settings in `libmamba.Context`. (#192 via #193)
* Fixes an error where a user-supplied package in the CLI would get dropped if the argument happened to match a larger string in the argument list. (#221 via #222)
* Fix authentication in package downloads. (#216 via #226)
* Fix unhandled channels coming from injected virtual packages in `conda-lock`. (#230 via #234)
* Handle URL-encoded channels correctly with recent `libmamba` versions. (#247 via #248)

### Docs

* Added notes about development workflows with libmamba from source. (#191)

### Other

* Adding libmamba/libmambapy as development dependencies within docker images. (#191)

### Contributors

* @costrouc
* @jaimergp
* @conda-bot
* @pre-commit-ci[bot]



## 23.5.0 (2023-05-25)

### Enhancements

* Provide a `CONDA_LIBMAMBA_SOLVER_NO_CHANNELS_FROM_INSTALLED` environment variable to prevent
  channels from being injected from installed packages. This is useful for air-gapped environments
  where outside channels are not available. (#108 via #184)
* Simplify `libmambapy.Context` initialization so we only set the bits that we use. (#209)
* Use the new `RepoInterface` and remove the `SubdirData` subclass workarounds, which requires `conda 23.5.0`.
  (#210)

### Bug fixes

* Fix an issue where running `conda update <package>` would result in the package being _downgraded_ if no newer versions were available. (#71 via #158)
* Ensure unauthenticated channels are not re-injected in the channel lists from installed packages
  if an authenticated equivalent is already present. (#108 via #184)
* Honor `context.repodata_threads`. (#200)
* Do not set `quiet` manually when `context.json` is true. (#187)

### Deprecations

* Remove unneeded user-agent tests. (#183)

### Docs

* Document known solver behavior differences. (#115, #131 via #197)
* Update development docs to reflect changes in build system and other inaccuracies. (#208)

### Other

* Add tests reproducing the known solver differences. (#115, #131 via #197)
* Skip tests on libmamba 1.4.2 temporarily to workaround some test failures. Tracked by #186. (#187)

### Contributors

* @jakirkham made their first contribution in https://github.com/conda/conda-libmamba-solver/pull/189
* @costrouc
* @jaimergp
* @jezdez
* @kenodegard
* @conda-bot
* @pre-commit-ci[bot]



## 23.3.0 (2023-03-31)

### Enhancements

* Simplify exception parsing and enable new (experimental) conflict reports in `libmamba`. (#102 via #103, #160)
* Use `conda`'s `SubdirData` for all repodata fetching and caching. (#59, #68 via #65, #171)

### Bug fixes

* Disable lockfiles within libmambapy to conform with conda's behavior of not using them. (#120)
* Fix JSON serialization errors in some exceptions. (#140 via #142)
* Fix API breakage upstream: `SubdirData.cache_path_json` property changed from `str` to `PrefixPath`. Depend directly on `boltons.setutils.IndexedSet`. (#151)
* Updated bundled conda recipe and corresponding CI workflow. (#166)
* Bumped minimum conda version from 22.11.1 -> 23.3.0 due to change in boltons IndexedSet. (#170)
* Add workaround for `defaults::<pkg_name>` specs. (#173 via #172)

### Deprecations

* Python 3.7 is no longer supported. The minimum version is now 3.8. (#174)

### Other

* Change the build-system to `hatchling` + `hatch-cvs` for a `setuptools-scm`-like versioning setup. (#128 via #127)
* Add conda-forge based CI environments. (#133)
* Fix cache directory in flaky test. (#157)
* CI: Pin `minio` to `2023-03-13T19-46-17Z` to avoid breaking changes. (#159)
* Require `libmamba 1.4.1` or greater and remove unused code paths. (#165)

### Contributors

* @AlbertDeFusco made their first contribution in https://github.com/conda/conda-libmamba-solver/pull/142
* @costrouc
* @jaimergp
* @jezdez
* @conda-bot
* @pre-commit-ci[bot]



## [23.1.0] - 2023-01-31

### Bug fixes

* Fix "Packages Not Found" error messages to be more accurate and informative. (#96 via #101)
* Ensure solves are deterministic and input order independent. (#75 via #111)
* Fix compatibility errors with newer conda versions >=23.1.0 since we are using an internal API SubdirData. (#118 via #119)

### Docs

* Mention expected versions and how to upgrade from experimental builds. (#89 via #93)

### Other

* CI: Add scheduled runs with self-reported issues. (#60 via #106)
* Fix typo in workflow documentation so it is consistent with the setup page. (#110)

### Contributors

* @costrouc made their first contribution in #110
* @jaimergp
* @jezdez
* @conda-bot
* @pre-commit-ci[bot]


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
[23.1.0]: https://github.com/conda/conda-libmamba-solver/releases/tag/23.1.0
