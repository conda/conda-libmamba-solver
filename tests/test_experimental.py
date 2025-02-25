# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Ensure experimental features work accordingly.
"""

from __future__ import annotations

import os
import sys
from subprocess import run
from typing import TYPE_CHECKING

import pytest
from conda.base.context import context, fresh_context
from conda.exceptions import CondaEnvironmentError
from conda.testing.integration import _get_temp_prefix

if TYPE_CHECKING:
    from conda.testing.fixtures import CondaCLIFixture
    from pytest import MonkeyPatch


def _get_temp_prefix_safe():
    return _get_temp_prefix(use_restricted_unicode=True).replace(" ", "")


def print_and_check_output(*args, **kwargs):
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("universal_newlines", True)
    process = run(*args, **kwargs)
    print("stdout", process.stdout, "---", "stderr", process.stderr, sep="\n")
    process.check_returncode()
    return process


@pytest.mark.xfail(reason="base protections not enabled anymore")
def test_protection_for_base_env(monkeypatch: MonkeyPatch, conda_cli: CondaCLIFixture) -> None:
    with pytest.raises(CondaEnvironmentError), fresh_context(CONDA_SOLVER="libmamba"):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        conda_cli(
            "install",
            f"--prefix={context.root_prefix}",
            "--dry-run",
            "scipy",
            "--solver=libmamba",
        )


def test_cli_flag_in_help():
    commands_with_flag = (
        ["install"],
        ["update"],
        ["remove"],
        ["create"],
        ["env", "create"],
        ["env", "update"],
        ["env", "remove"],
    )
    for command in commands_with_flag:
        process = print_and_check_output([sys.executable, "-m", "conda"] + command + ["--help"])
        assert "--solver" in process.stdout

    commands_without_flag = (
        ["config"],
        ["list"],
        ["info"],
        ["run"],
        ["env", "list"],
    )
    for command in commands_without_flag:
        process = print_and_check_output([sys.executable, "-m", "conda"] + command + ["--help"])
        assert "--solver" not in process.stdout


def cli_flag_and_env_var_settings():
    env_no_var = os.environ.copy()
    env_no_var.pop("CONDA_SOLVER", None)
    env_libmamba = env_no_var.copy()
    env_classic = env_no_var.copy()
    env_libmamba["CONDA_SOLVER"] = "libmamba"
    env_classic["CONDA_SOLVER"] = "classic"
    command = [
        sys.executable,
        "-m",
        "conda",
        "create",
        "-y",
        "-p",
        _get_temp_prefix_safe(),
        "--dry-run",
        "xz",
    ]
    cli_libmamba = ["--solver=libmamba"]
    cli_classic = ["--solver=classic"]
    tests = [
        ["no flag, no env", command, env_no_var, "classic"],
        ["no flag, env classic", command, env_classic, "classic"],
        ["no flag, env libmamba", command, env_libmamba, "libmamba"],
        ["flag libmamba, no env", command + cli_libmamba, env_no_var, "libmamba"],
        ["flag libmamba, env libmamba", command + cli_libmamba, env_libmamba, "libmamba"],
        ["flag libmamba, env classic", command + cli_libmamba, env_classic, "libmamba"],
        ["flag classic, no env", command + cli_classic, env_no_var, "classic"],
        ["flag classic, env libmamba", command + cli_classic, env_libmamba, "classic"],
        ["flag classic, env classic", command + cli_classic, env_classic, "classic"],
    ]
    return tests
