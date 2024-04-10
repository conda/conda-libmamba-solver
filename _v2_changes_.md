# Libmamba v2 integration changes

- Deprecate CONDA_LIBMAMBA_SOLVER_NO_CHANNELS_FROM_INSTALLED. No longer needed with v2.
- Typing in all methods
- time_recorder for metadata collection and solving loop
- removed state.BaseIndexHelper
- The libmamba v1 "pool" (collection of "repos"; repo = loaded repodata.json) is now a "database" of "RepoInfo" objects.
- Do use current_repodata.json if explicitly set in CLI
- Move to Ruff for pre-commit linting & formatting
- Logging from libsolv has a big overhead now because it goes from C to C++ to Python to logging to stdout instead of C -> stdout.
- Allow uninstall was previously set to false (only true for conda remove), and we set it up for individual solver jobs involving updates or conflicts. With v2, we have individual control over what to "Keep" instead of drop. This required marking important updates as keepers instead. Otherwise they would be uninstalled.
- Other changes in the test suite discussed in https://github.com/conda/conda/pull/13784
