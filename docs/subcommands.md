# Subcommands

The conda-libmamba-solver package also provides conda subcommand plugins in addition to the solver plugin.

## `conda repoquery`

A conda subcommand plugin that offers the same functionality as `mamba repoquery` and
`micromamba repoquery`. It provides three actions:

- `conda repoquery search`: Query repodata for packages matching a pattern.
- `conda repoquery depends`: Show the dependencies of the requested package.
- `conda repoquery whoneeds`: Show the packages that depend on the requested package.

Check the `--help` messages for each task for more information.
