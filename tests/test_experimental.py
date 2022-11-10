# Copyright (C) 2022 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause
"""
Ensure experimental features work accordingly.
"""
import os
import sys
from subprocess import run

import pytest

from conda.base.context import fresh_context, context
from conda.exceptions import CondaEnvironmentError
from conda.testing.integration import run_command, Commands, _get_temp_prefix


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
def test_protection_for_base_env():
    with pytest.raises(CondaEnvironmentError), fresh_context(CONDA_SOLVER="libmamba"):
        current_test = os.environ.pop("PYTEST_CURRENT_TEST", None)
        try:
            run_command(
                Commands.INSTALL,
                context.root_prefix,
                "--dry-run",
                "scipy",
                "--solver=libmamba",
                no_capture=True,
            )
        finally:
            if current_test is not None:
                os.environ["PYTEST_CURRENT_TEST"] = current_test


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
