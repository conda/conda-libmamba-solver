# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Ensure experimental features work accordingly.
"""

import os
import sys
from subprocess import run

import pytest
from conda.base.context import context, fresh_context
from conda.exceptions import CondaEnvironmentError
from conda.testing.integration import Commands, run_command


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
