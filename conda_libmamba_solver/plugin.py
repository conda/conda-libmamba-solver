# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
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
