# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Measure the speed and memory usage of the different backend solvers
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from conda.base.context import context
from conda.exceptions import DryRunExit

if TYPE_CHECKING:
    from collections.abc import Iterable

    from conda.testing.fixtures import CondaCLIFixture, TmpEnvFixture
    from pytest import FixtureRequest

pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("parametrized_solver_fixture")]

TEST_DATA_DIR = Path(__file__).parent / "data"
PLATFORM = context.subdir


def _get_channels_from_lockfile(path: Path) -> tuple[str, ...]:
    """Parse `# channels: conda-forge,defaults` comments"""
    for line in path.read_text().splitlines():
        if line.startswith("# channels:"):
            return tuple(line.split(":")[1].strip().split(","))
    return ()


def _channels_as_args(channels: Iterable[str]) -> tuple[str, ...]:
    if not channels:
        return ()
    return ("--override-channels", *(f"--channel={channel}" for channel in channels))


@pytest.fixture(
    scope="session",
    params=TEST_DATA_DIR.glob("*.lock"),
)
def prefix_and_channels(
    request: FixtureRequest,
    session_tmp_env: TmpEnvFixture,
) -> Iterable[tuple[Path, tuple[str, ...]]]:
    lockfile = Path(request.param)
    lock_platform = lockfile.suffixes[-2]
    if lock_platform != PLATFORM:
        pytest.skip(f"Running platform {PLATFORM} does not match file platform {lock_platform}")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("CONDA_TEST_SAVE_TEMPS", "1")

        with session_tmp_env("--file", lockfile) as prefix:
            channels = _get_channels_from_lockfile(lockfile)
            yield prefix, channels


def test_update_python(
    prefix_and_channels: tuple[Path, tuple[str, ...]],
    conda_cli: CondaCLIFixture,
) -> None:
    prefix, channels = prefix_and_channels
    try:
        conda_cli(
            "update",
            f"--prefix={prefix}",
            "--dry-run",
            *_channels_as_args(channels),
            "python",
        )
    except DryRunExit:
        assert True
    else:
        # this can happen if "all requirements are satisfied"
        assert True


def test_install_python_update_deps(
    prefix_and_channels: tuple[Path, tuple[str, ...]],
    conda_cli: CondaCLIFixture,
) -> None:
    prefix, channels = prefix_and_channels
    conda_cli(
        "install",
        f"--prefix={prefix}",
        "--dry-run",
        *_channels_as_args(channels),
        "python",
        "--update-deps",
        raises=DryRunExit,
    )


def test_update_all(
    prefix_and_channels: tuple[Path, tuple[str, ...]],
    conda_cli: CondaCLIFixture,
) -> None:
    prefix, channels = prefix_and_channels
    conda_cli(
        "update",
        f"--prefix={prefix}",
        "--dry-run",
        *_channels_as_args(channels),
        "--all",
        raises=DryRunExit,
    )


def test_install_vaex_from_conda_forge_and_defaults(conda_cli: CondaCLIFixture) -> None:
    conda_cli(
        "create",
        "--dry-run",
        *_channels_as_args(["conda-forge", "defaults"]),
        "python=3.9",
        "vaex",
        raises=DryRunExit,
    )
