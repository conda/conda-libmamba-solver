"""
Measure the speed and memory usage of the different backend solvers
"""
import os
import shutil

import pytest

from conda.testing.integration import make_temp_env, run_command, Commands, _get_temp_prefix
from conda.common.io import env_var
from conda.base.context import context
from conda.base.constants import ExperimentalSolverChoice
from conda.exceptions import DryRunExit

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


def _tmp_prefix_safe():
    return _get_temp_prefix(use_restricted_unicode=True).replace(" ", "")


@pytest.fixture(scope="module", params=os.listdir(TEST_DATA_DIR))
def prefix_and_channels(request):
    lockfile = os.path.join(TEST_DATA_DIR, request.param)
    lock_platform = lockfile.split(".")[-2]
    if lock_platform != platform:
        pytest.skip(f"Running platform {platform} does not match file platform {lock_platform}")
    with env_var("CONDA_TEST_SAVE_TEMPS", "1"):
        prefix = _tmp_prefix_safe()
        with make_temp_env("--file", lockfile, prefix=prefix) as prefix:
            channels = _get_channels_from_lockfile(lockfile)
            yield prefix, channels
    shutil.rmtree(prefix)


@pytest.fixture(
    scope="function", params=[ExperimentalSolverChoice.LIBMAMBA, ExperimentalSolverChoice.CLASSIC]
)
def solver_args(request):
    yield ("--dry-run", "--experimental-solver", request.param.value)


@pytest.mark.slow
def test_a_warmup(prefix_and_channels, solver_args):
    """Dummy test to install envs and warm up caches"""
    prefix_and_channels, solver_args = prefix_and_channels, solver_args


@pytest.mark.slow
def test_update_python(prefix_and_channels, solver_args):
    prefix, channels = prefix_and_channels
    try:
        run_command(Commands.UPDATE, prefix, *_channels_as_args(channels), *solver_args, "python")
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
        run_command(Commands.UPDATE, prefix, *_channels_as_args(channels), *solver_args, "--all")
    except DryRunExit:
        assert True
    else:
        # this can happen if "all requirements are satisfied"
        assert True


@pytest.mark.slow
def test_install_vaex_from_conda_forge_and_defaults(solver_args):
    try:
        run_command(
            Commands.CREATE,
            _tmp_prefix_safe(),
            *solver_args,
            "--override-channels",
            "-c",
            "conda-forge",
            "-c",
            "defaults",
            "python=3.9",
            "vaex",
        )
    except DryRunExit:
        assert True
