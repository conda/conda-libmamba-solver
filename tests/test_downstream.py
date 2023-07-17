# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
import os
from pathlib import Path
from subprocess import CalledProcessError, check_call

import pytest
from conda.base.context import context


DATA = Path(__file__).parent / "data"


def test_build_recipes():
    """
    Adapted from
    https://github.com/mamba-org/boa/blob/3213180564/tests/test_mambabuild.py#L6

    See /tests/data/conda_build_recipes/LICENSE for more details
    """
    recipes_dir = DATA / "conda_build_recipes"

    recipes = [str(x) for x in recipes_dir.iterdir() if x.is_dir()]
    env = os.environ.copy()
    env["CONDA_SOLVER"] = "libmamba"
    expected_fail_recipes = ["baddeps"]
    for recipe in recipes:
        recipe_name = Path(recipe).name
        print(f"Running {recipe_name}")
        if recipe_name in expected_fail_recipes:
            with pytest.raises(CalledProcessError):
                check_call(["conda", "build", recipe], env=env)
        else:
            check_call(["conda", "build", recipe], env=env)


def test_conda_lock():
    env = os.environ.copy()
    env["CONDA_SOLVER"] = "libmamba"
    check_call(["conda-lock", "lock", "-p", context.subdir, "-f", DATA / "lock_this_env.yml"], env=env)
