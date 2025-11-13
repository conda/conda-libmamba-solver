# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Entry points for the conda plugin system
"""

from conda.common.configuration import PrimitiveParameter
from conda.plugins import hookimpl
from conda.plugins.types import CondaSetting, CondaSolver, CondaSubcommand

from .repoquery import configure_parser, repoquery
from .solver import LibMambaSolver


@hookimpl
def conda_solvers():
    """
    The conda plugin hook implementation to load the solver into conda.
    """
    yield CondaSolver(
        name="libmamba",
        backend=LibMambaSolver,
    )


@hookimpl
def conda_subcommands():
    yield CondaSubcommand(
        name="repoquery",
        summary="Advanced search for repodata.",
        action=repoquery,
        configure_parser=configure_parser,
    )


@hookimpl
def conda_settings():
    """
    Define all settings specific to the conda-libmamba-solver plugin.
    """
    yield CondaSetting(
        name="use_sharded_repodata",
        description="Enable use of sharded repodata when available.",
        parameter=PrimitiveParameter(False, element_type=bool),
    )
