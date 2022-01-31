"""
Ensure experimental features work accordingly.
"""
import os
import pytest

from conda.base.context import fresh_context, context
from conda.exceptions import CondaEnvironmentError
from conda.testing.integration import run_command, Commands


@pytest.mark.parametrize("solver", ("libmamba", "libmamba-draft"))
def test_protection_for_base_env(solver):
    with pytest.raises(CondaEnvironmentError), fresh_context(CONDA_SOLVER_LOGIC=solver):
        os.environ.pop("PYTEST_CURRENT_TEST", None)
        run_command(Commands.INSTALL, context.root_prefix, "--dry-run", "scipy", no_capture=True)
