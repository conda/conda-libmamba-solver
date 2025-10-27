# Profiling

Profiling for this project can be found on codspeed here:

- https://codspeed.io/conda/conda-libmamba-solver

To run profiling tests locally, run the following command in the root of the project:

```shell
pytest --codspeed
```

To profile tests, add the following decorator above it:

```python
import pytest


@pytest.mark.benchmark
def test_new_feature():
    """Ensure feature performs well"""
```
