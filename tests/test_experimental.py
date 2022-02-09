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
    stdout = check_output([sys.executable, "-m", "conda", "create", "-y", "-p", _get_temp_prefix(), "--dry-run", "xz"], stderr=STDOUT, universal_newlines=True, env=env)

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


def test_cli_flag():
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
