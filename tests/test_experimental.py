"""
Ensure experimental features work accordingly.
"""
import os
import sys
from subprocess import check_output, STDOUT

import pytest

from conda.base.context import fresh_context, context
from conda.exceptions import CondaEnvironmentError
from conda.testing.integration import run_command, Commands, _get_temp_prefix


@pytest.mark.parametrize("solver", ("libmamba", "libmamba-draft"))
def test_protection_for_base_env(solver):
    with pytest.raises(CondaEnvironmentError), fresh_context(CONDA_EXPERIMENTAL_SOLVER=solver):
        os.environ.pop("PYTEST_CURRENT_TEST", None)
        run_command(Commands.INSTALL, context.root_prefix, "--dry-run", "scipy", no_capture=True)


def test_logging():
    "Check we are indeed writing full logs to disk"
    env = os.environ.copy()
    env["CONDA_EXPERIMENTAL_SOLVER"] = "libmamba"
    stdout = check_output(
        [sys.executable, "-m", "conda", "create", "-y", "-p", _get_temp_prefix(), "--dry-run", "xz"],
        env=env, stderr=STDOUT, universal_newlines=True,
    )

    in_header = False
    for line in stdout.splitlines():
        line = line.strip()
        if "USING EXPERIMENTAL LIBMAMBA INTEGRATIONS" in line:
            in_header = True
        elif line.startswith("----------"):
            in_header = False
        elif in_header and line.endswith(".log"):
            logfile_path = line
            break
    else:
        pytest.fail("Could not find logfile path in outout")

    with open(logfile_path) as f:
        log_contents = f.read()
        assert "conda.conda_libmamba_solver" in log_contents
        assert "solver started" in log_contents
        assert "choice rule creation took" in log_contents


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
        stdout = check_output([sys.executable, "-m", "conda"] + command + ["--help"], stderr=STDOUT, universal_newlines=True)
        assert "--experimental-solver" in stdout

    commands_without_flag = (
       ["config"],
       ["list"],
       ["info"],
       ["run"],
       ["env", "list"],
    )
    for command in commands_without_flag:
        stdout = check_output([sys.executable, "-m", "conda"] + command + ["--help"], stderr=STDOUT, universal_newlines=True)
        assert "--experimental-solver" not in stdout


def test_cli_flag_and_env_var_settings():
    env_no_var = os.environ.copy()
    env_libmamba = os.environ.copy()
    env_classic = os.environ.copy()
    env_libmamba["CONDA_EXPERIMENTAL_SOLVER"] = "libmamba"
    env_classic["CONDA_EXPERIMENTAL_SOLVER"] = "classic"
    command = [sys.executable, "-m", "conda", "create", "-y", "-p", _get_temp_prefix(), "--dry-run", "xz"]
    cli_libmamba = ["--experimental-solver=libmamba"]
    cli_classic = ["--experimental-solver=classic"]
    tests = [
        {"cmd": command, "env": env_no_var, "solver": "classic"},
        {"cmd": command, "env": env_classic, "solver": "classic"},
        {"cmd": command, "env": env_libmamba, "solver": "libmamba"},
        {"cmd": command + cli_libmamba, "env": env_no_var, "solver": "libmamba"},
        {"cmd": command + cli_libmamba, "env": env_libmamba, "solver": "libmamba"},
        {"cmd": command + cli_libmamba, "env": env_classic, "solver": "libmamba"},
        {"cmd": command + cli_classic, "env": env_no_var, "solver": "classic"},
        {"cmd": command + cli_classic, "env": env_libmamba, "solver": "classic"},
        {"cmd": command + cli_classic, "env": env_classic, "solver": "classic"},
    ]
    for test in tests:
        stdout = check_output(test["cmd"], env=test["env"], stderr=STDOUT, universal_newlines=True)
        if test["solver"] == "libmamba":
            assert "USING EXPERIMENTAL LIBMAMBA INTEGRATIONS" in stdout
        else:
            assert "USING EXPERIMENTAL LIBMAMBA INTEGRATIONS" not in stdout
