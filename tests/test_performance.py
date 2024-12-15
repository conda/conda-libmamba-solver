# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Measure the speed and memory usage of the different backend solvers
"""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

import pytest
from conda.base.context import context
from conda.common.io import env_var
from conda.exceptions import DryRunExit
from conda.testing.integration import Commands, make_temp_env, run_command

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Iterable

    from conda.testing.fixtures import PathFactoryFixture
    from pytest import FixtureRequest

platform = context.subdir

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _get_channels_from_lockfile(path):
    """Parse `# channels: conda-forge,defaults` comments"""
    with open(path) as f:
        for line in f:
            if line.startswith("# channels:"):
                return line.split(":")[1].strip().split(",")


def _channels_as_args(channels):
    if not channels:
        return ()
    args = ["--override-channels"]
    for channel in channels:
        args += ["-c", channel]
    return tuple(args)


@pytest.fixture(
    scope="module",
    params=[f for f in os.listdir(TEST_DATA_DIR) if f.endswith(".lock")],
)
def prefix_and_channels(
    request: FixtureRequest, session_path_factory: PathFactoryFixture
) -> Iterable[tuple[Path, list[str]]]:
    lockfile = os.path.join(TEST_DATA_DIR, request.param)
    lock_platform = lockfile.split(".")[-2]
    if lock_platform != platform:
        pytest.skip(f"Running platform {platform} does not match file platform {lock_platform}")
    with env_var("CONDA_TEST_SAVE_TEMPS", "1"):
        prefix = session_path_factory()
        with make_temp_env("--file", lockfile, prefix=prefix) as prefix:
            channels = _get_channels_from_lockfile(lockfile)
            yield prefix, channels
    shutil.rmtree(prefix)


@pytest.fixture(scope="function", params=["libmamba", "classic"])
def solver_args(request):
    yield ("--dry-run", "--solver", request.param)


@pytest.mark.slow
def test_a_warmup(prefix_and_channels, solver_args):
    """Dummy test to install envs and warm up caches"""
    prefix_and_channels, solver_args = prefix_and_channels, solver_args


@pytest.mark.slow
def test_update_python(prefix_and_channels, solver_args):
    prefix, channels = prefix_and_channels
    try:
        run_command(
            Commands.UPDATE,
            prefix,
            *_channels_as_args(channels),
            *solver_args,
            "python",
            no_capture=True,
        )
    except DryRunExit:
        assert True
    else:
        # this can happen if "all requirements are satisfied"
        assert True


@pytest.mark.slow
def test_install_python_update_deps(prefix_and_channels, solver_args):
    prefix, channels = prefix_and_channels
    try:
        run_command(
            Commands.INSTALL,
            prefix,
            *_channels_as_args(channels),
            *solver_args,
            "python",
            "--update-deps",
            no_capture=True,
        )
    except DryRunExit:
        assert True
    else:
        # this can happen if "all requirements are satisfied"
        assert True


@pytest.mark.slow
def test_update_all(prefix_and_channels, solver_args):
    prefix, channels = prefix_and_channels
    try:
        run_command(
            Commands.UPDATE,
            prefix,
            *_channels_as_args(channels),
            *solver_args,
            "--all",
            no_capture=True,
        )
    except DryRunExit:
        assert True
    else:
        # this can happen if "all requirements are satisfied"
        assert True


@pytest.mark.slow
def test_install_vaex_from_conda_forge_and_defaults(
    solver_args: tuple[str, ...], path_factory: PathFactoryFixture
) -> None:
    try:
        run_command(
            Commands.CREATE,
            str(path_factory()),
            *solver_args,
            "--override-channels",
            "-c",
            "conda-forge",
            "-c",
            "defaults",
            "python=3.9",
            "vaex",
            no_capture=True,
        )
    except DryRunExit:
        assert True
