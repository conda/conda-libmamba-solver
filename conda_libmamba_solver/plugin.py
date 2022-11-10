from conda import plugins

from .solver import LibMambaSolver


@plugins.hookimpl
def conda_solvers():
    """
    The conda plugin hook implementation to load the solver into conda.
    """
    yield plugins.CondaSolver(
        name="libmamba",
        backend=LibMambaSolver,
    )
