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
$ cd /opt/conda-libmamba-src
$ pytest
```

### Integration tests

From the properly mounted `conda/conda` Docker container (see ["Development environment setup"](setup.md)):

```bash
$ cd /opt/conda-src
$ CONDA_SOLVER=libmamba pytest
```

Note we [deselect some upstream tests in our `pyproject.toml`](../../pyproject.toml) for a number of reasons.
The CI workflows override `conda`'s pytest settings in `setup.cfg` with the ones present in`conda-libmamba-solver`'s `pyproject.toml`.
This allows us to apply the ignored filters automatically.
You can replace the files as well in your debugging sessions, but remember to revert once you are done!
