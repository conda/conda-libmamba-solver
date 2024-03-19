# Typical development workflows

```{note}
Check out ["How to set up your development environment"](setup.md) if you haven't yet!
```

## Testing

The solver is a critical part of `conda` as a tool.
In addition to unit tests for `conda_libmamba_solver`,
our CI also runs the full `conda/conda` integration suite.

### conda-libmamba-solver tests

From the properly mounted devcontainer (see ["Development environment setup"](setup.md)):

```bash
$ cd /workspaces/conda-libmamba-solver
$ pytest
```

Or just use the PyTest integrations in VS Code (flask icon).

### Upstream tests

From the properly mounted devcontainer (see ["Development environment setup"](setup.md)):

```bash
$ cd /workspaces/conda
$ CONDA_SOLVER=libmamba pytest
```
