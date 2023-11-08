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
                check_call(["conda-build", recipe], env=env)
        else:
            check_call(["conda-build", recipe], env=env)


def test_conda_build_with_aliased_channels(request):
    "https://github.com/conda/conda-libmamba-solver/issues/363"
    condarc = Path.home() / ".condarc"
    if condarc.exists():
        condarc_contents = condarc.read_text()
        request.addfinalizer(lambda: condarc.write_text(condarc_contents))
    else:
        request.addfinalizer(lambda: condarc.unlink())
    condarc.write_text(
        "channels: ['defaults']\n"
        "default_channels: ['conda-forge']\n"
        "solver: libmamba\n"
    )
    check_call(["conda-build", DATA / "conda_build_recipes" / "jedi"])


def test_conda_lock(tmp_path):
    env = os.environ.copy()
    env["CONDA_SOLVER"] = "libmamba"
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
        env=env,
        cwd=tmp_path,
    )
