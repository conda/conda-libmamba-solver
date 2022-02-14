"""
Measure the speed and memory usage of the different backend solvers
"""
import os
import shutil

import pytest

from conda.testing.integration import make_temp_env, run_command, Commands
from conda.common.io import env_var
from conda.base.context import context
from conda.base.constants import ExperimentalSolverChoice
from conda.exceptions import DryRunExit

platform = context.subdir

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="module", params=os.listdir(TEST_DATA_DIR))
def prefix(request):
    lock_platform = request.param.split(".")[-2]
    if lock_platform != platform:
        pytest.skip(f"Running platform {platform} does not match file platform {lock_platform}")
    with env_var("CONDA_TEST_SAVE_TEMPS", "1"):
        with make_temp_env("--file", os.path.join(TEST_DATA_DIR, request.param)) as prefix:
            yield prefix
    shutil.rmtree(prefix)


@pytest.fixture(scope="function", params=[ExperimentalSolverChoice.LIBMAMBA, ExperimentalSolverChoice.CLASSIC])
def solver_args(request):
    yield ("--dry-run", "--experimental-solver", request.param.value)


@pytest.mark.slow
def test_a_warmup(prefix, solver_args):
    """Dummy test to install envs and warm up caches"""
    prefix, solver_args = prefix, solver_args


@pytest.mark.slow
def test_update_python(prefix, solver_args):
    with pytest.raises(DryRunExit):
        run_command(Commands.UPDATE, prefix, *solver_args, "python")

