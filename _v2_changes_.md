# Libmamba v2 integration changes

- Deprecate CONDA_LIBMAMBA_SOLVER_NO_CHANNELS_FROM_INSTALLED. No longer needed with v2.
- Typing in all methods
- time_recorder for metadata collection and solving loop
- removed state.BaseIndexHelper
- The libmamba v1 "pool" (collection of "repos"; repo = loaded repodata.json) is now a "database" of "RepoInfo" objects.
- Do use current_repodata.json if explicitly set in CLI
