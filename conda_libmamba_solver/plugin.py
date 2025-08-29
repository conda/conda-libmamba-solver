# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Entry points for the conda plugin system
"""

from conda import plugins

try:
    from conda.plugins import CondaSolver
except ImportError:
    # FUTURE: Use this import only in conda 26.3+
    from conda.plugins.types import CondaSolver

from .repoquery import configure_parser, repoquery
from .solver import LibMambaSolver


@plugins.hookimpl
def conda_solvers():
    """
    The conda plugin hook implementation to load the solver into conda.
    """
    yield CondaSolver(
        name="libmamba",
        backend=LibMambaSolver,
    )


@plugins.hookimpl
def conda_subcommands():
    yield plugins.CondaSubcommand(
        name="repoquery",
        summary="Advanced search for repodata.",
        action=repoquery,
        configure_parser=configure_parser,
    )
