# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2022 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import os
import sys
from itertools import chain, permutations, repeat
from pathlib import Path
from subprocess import check_call, run
from uuid import uuid4

from conda.common.compat import on_win
from conda.core.prefix_data import get_python_version_for_prefix
from conda.testing.integration import Commands, make_temp_env, run_command
from conda.testing.solver_helpers import SolverTests

from conda_libmamba_solver import LibMambaSolver


class TestLibMambaSolver(SolverTests):
    @property
    def solver_class(self):
        return LibMambaSolver

    @property
    def tests_to_skip(self):
        return {
            "LibMambaSolver does not support track-features/features": [
                "test_iopro_mkl",
                "test_iopro_nomkl",
                "test_mkl",
                "test_accelerate",
                "test_scipy_mkl",
                "test_pseudo_boolean",
                "test_no_features",
                "test_surplus_features_1",
                "test_surplus_features_2",
                # this one below only fails reliably on windows;
                # it passes Linux on CI, but not locally?
                "test_unintentional_feature_downgrade",
            ],
            "LibMambaSolver installs numpy with mkl while we were expecting no-mkl numpy": [
                "test_remove",
            ],
        }


def test_python_downgrade_reinstalls_noarch_packages():
    """
    Reported in https://github.com/conda/conda/issues/11346

    See also test_create::test_noarch_python_package_reinstall_on_pyver_change
    in conda/conda test suite. Note that we use conda-forge here deliberately;
    defaults at the time of writing (March 2022) packages pip as a non-noarch
    build, which means it has a different name across Python versions. conda-forge
    uses noarch here, so the package is the same across Python versions. Probably
    why upstream didn't catch this error before.
    """
    with make_temp_env(
        "--override-channels",
        "-c",
        "conda-forge",
        "--solver=libmamba",
        "pip",
        "python=3.10",
        name=f"conda_libmamba_solver-{uuid4()}",  # shebangs cannot contain spaces - override!
        no_capture=True,
    ) as prefix:
        py_ver = get_python_version_for_prefix(prefix)
        assert py_ver.startswith("3.10")
        if on_win:
            pip = str(Path(prefix) / "Scripts" / "pip.exe")
        else:
            pip = str(Path(prefix) / "bin" / "pip")
        check_call([pip, "--version"])

        run_command(
            Commands.INSTALL,
            prefix,
            "--solver=libmamba",
            "--override-channels",
            "-c",
            "conda-forge",
            "python=3.9",
            no_capture=True,
        )
        check_call([pip, "--version"])


def test_determinism(tmpdir):
    "Based on https://github.com/conda/conda-libmamba-solver/issues/75"
    env = os.environ.copy()
    env.pop("PYTHONHASHSEED", None)
    env["CONDA_PKGS_DIRS"] = str(tmpdir / "pkgs")
    installed_bokeh_versions = []
    common_args = (
        sys.executable,
        "-mconda",
        "create",
        "--name=unused",
        "--dry-run",
        "--yes",
        "--json",
        "--solver=libmamba",
        "--channel=conda-forge",
        "--override-channels",
    )
    pkgs = ("python=3.8", "bokeh", "hvplot")
    # Two things being tested in the same loop:
    # - Repeated attempts of the same input should give the same result
    # - Input order (from the user side) should not matter, and should give the same result
    for i, pkg_list in enumerate(chain(repeat(pkgs, 10), permutations(pkgs, len(pkgs)))):
        offline = ("--offline",) if i else ()
        process = run([*common_args, *offline, *pkg_list], env=env, text=True, capture_output=True)
        if process.returncode:
            print("Attempt:", i)
            print(process.stdout)
            print(process.stderr, file=sys.stderr)
            process.check_returncode()
        data = json.loads(process.stdout)
        assert data["success"] is True
        for pkg in data["actions"]["LINK"]:
            if pkg["name"] == "bokeh":
                installed_bokeh_versions.append(pkg["version"])
                break
        else:
            raise AssertionError("Didn't find bokeh!")
    assert len(set(installed_bokeh_versions)) == 1
