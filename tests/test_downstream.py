# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
import os
from pathlib import Path
from subprocess import CalledProcessError, check_call

import pytest
from conda.base.context import context

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    "recipe",
    [
        pytest.param(x, id=x.name)
        for x in sorted((DATA / "conda_build_recipes").iterdir())
        if (x / "meta.yaml").is_file()
    ],
)
def test_build_recipe(recipe):
    """
    Adapted from
    https://github.com/mamba-org/boa/blob/3213180564/tests/test_mambabuild.py#L6

    See /tests/data/conda_build_recipes/LICENSE for more details
    """
    expected_fail_recipes = ["baddeps"]
    env = os.environ.copy()
    env["CONDA_SOLVER"] = "libmamba"
    recipe_name = Path(recipe).name
    if recipe_name in expected_fail_recipes:
        with pytest.raises(CalledProcessError):
            check_call(["conda-build", recipe], env=env)
    else:
        check_call(["conda-build", recipe], env=env)


def test_conda_lock(tmp_path):
    conda_exe_path = "Scripts/conda.exe" if os.name == "nt" else "bin/conda"
    check_call(
        [
            "conda-lock",
            "lock",
            "--platform",
            context.subdir,
            "--file",
            DATA / "lock_this_env.yml",
            "--conda",
            Path(context.conda_prefix) / conda_exe_path,
        ],
        cwd=tmp_path,
    )
