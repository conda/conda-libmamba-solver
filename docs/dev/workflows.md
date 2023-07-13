# Typical development workflows

```{note}
Check out ["How to set up your development environment"](setup.md) if you haven't yet!
```

## Testing

The solver is a critical part of `conda` as a tool.
In addition to unit tests for `conda_libmamba_solver`,
our CI also runs the full `conda/conda` integration suite.

### Unit tests

From the properly mounted `conda/conda` Docker container (see ["Development environment setup"](setup.md)):

```bash
$ cd /opt/conda-libmamba-solver-src
$ pytest
```

### Integration tests

From the properly mounted `conda/conda` Docker container (see ["Development environment setup"](setup.md)):

```bash
$ cd /opt/conda-src
$ CONDA_SOLVER=libmamba pytest
```

Note we [deselect some upstream tests in our `pyproject.toml`](../../dev/collect_upstream_conda_tests/collect_upstream_conda_tests.py) for a number of reasons.
For this to work we need to ensure that `pytest` loads that plugin by installing it in the same environment.
