# Performance testing

## Codspeed

Performance testing for this project can be found on codspeed here:

- https://codspeed.io/conda/conda-libmamba-solver

To run performance tests locally, run the following command in the root of the project:

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

See the official [codspeed documentation](https://codspeed.io/docs) for more information.
