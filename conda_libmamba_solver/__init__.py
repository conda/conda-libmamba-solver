
from .libmamba import LibMambaSolver
from .libmamba2 import LibMambaSolver2


def get_solver_logic(key=None):
    return {
        "libmamba": LibMambaSolver,
        "libmamba2": LibMambaSolver2,
    }[key]
