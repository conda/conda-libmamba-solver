# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
This module collects examples of environments that were hard to solve, required
workarounds or didn't meet users' expectations... specially if compared to conda classic.
"""

import json
import os

import pytest
from conda.common.compat import on_linux

from .repodata_time_machine import repodata_time_machine
from .utils import conda_subprocess


@pytest.mark.skip(reason="Fixed by #381. v1.8.2 is now found.")
def test_pydantic_182_not_on_python_311():
    """
    See daico007's report on https://github.com/conda/conda-libmamba-solver/issues/115

    - The original environment file didn't specify a Python version.
    - conda classic does find that Python 3.10 and pydantic 1.8.2 are compatible
    - libsolv has to try all Python versions, starting with 3.11 as of writing,
      and then tries different pydantic versions. It finds 0.18.2 is compatible, but
      because it's noarch with an open-ended upper bound.
    - If we do specify that we want a Python version for which pydantic 1.8.2 is available,
      libsolv correctly finds it.

    After #381, where stricter specs are ordered first in the solver task, the solver
    does find the correct pydantic as classic does.
    """
    env = os.environ.copy()
    env["CONDA_SUBDIR"] = "linux-64"
    args = (
        "create",
        "-n",
        "unused",
        "--dry-run",
        "--override-channels",
        "-c",
        "conda-forge",
        "--json",
    )
    pkgs = (
        "numpy",
        "sympy",
        "unyt<=2.8",
        "boltons",
        "lxml",
        "pydantic<1.9.0",
        "networkx",
        "ele>=0.2.0",
        "forcefield-utilities",
    )
    p = conda_subprocess(
        *args,
        "--solver=classic",
        *pkgs,
        env=env,
    )
    data = json.loads(p.stdout)
    pydantic = next(pkg for pkg in data["actions"]["LINK"] if pkg["name"] == "pydantic")
    assert pydantic["version"] == "1.8.2"

    p = conda_subprocess(
        *args,
        "--solver=libmamba",
        *pkgs,
    )
    data = json.loads(p.stdout)
    pydantic = next(pkg for pkg in data["actions"]["LINK"] if pkg["name"] == "pydantic")
    assert pydantic["version"] != "1.8.2"  # this was the bug, now fixed

    p = conda_subprocess(
        *args,
        "--solver=libmamba",
        *pkgs,
        "python<3.11",
    )
    data = json.loads(p.stdout)
    pydantic = next(pkg for pkg in data["actions"]["LINK"] if pkg["name"] == "pydantic")
    assert pydantic["version"] == "1.8.2"


@pytest.mark.skipif(not on_linux, reason="Only relevant on Linux")
def test_gpu_cpu_mutexes():
    """
    See:
        - https://github.com/conda/conda-libmamba-solver/issues/115#issuecomment-1399040871
        - https://github.com/conda/conda-libmamba-solver/issues/115#issuecomment-1399040867
        - https://github.com/conda/conda-libmamba-solver/issues/131

    This behaviour difference is known and explained at
    https://github.com/conda/conda-libmamba-solver/issues/131#issuecomment-1440745813.

    If at some point this changes (e.g. libmamba fix), this test will capture it.
    """
    args = (
        "create",
        "-n",
        "unused",
        "--dry-run",
        "--json",
        "--override-channels",
        "-c",
        "conda-forge",
        "-c",
        "pyg",
        "-c",
        "pytorch",
    )
    pkgs = (
        "cpuonly",
        "pyg=2.1.0",
        "python=3.9",
        "pytorch::pytorch=1.12",
    )
    env = os.environ.copy()
    env["CONDA_SUBDIR"] = "linux-64"
    p = conda_subprocess(
        *args,
        "--solver=classic",
        *pkgs,
        env=env,
    )
    data = json.loads(p.stdout)
    found = 0
    target_pkgs = ("pytorch", "pyg")
    for pkg in data["actions"]["LINK"]:
        if pkg["name"] in target_pkgs:
            found += 1
            assert "cpu" in pkg["build_string"]
        elif pkg["name"] == "cudatoolkit":
            raise AssertionError("CUDA shouldn't be installed due to 'cpuonly'")
    assert found == len(target_pkgs)

    p = conda_subprocess(
        *args,
        "--solver=libmamba",
        *pkgs,
        env=env,
    )
    data = json.loads(p.stdout)
    # This should not happen, but it does. See docstring.
    assert next(pkg for pkg in data["actions"]["LINK"] if pkg["name"] == "cudatoolkit")

    p = conda_subprocess(
        *args,
        "--solver=libmamba",
        "cpuonly",
        "pyg=2.1.0",
        "python=3.9",
        "pytorch::pytorch",  # more recent pytorch versions seem to be properly packaged
        env=env,
    )
    data = json.loads(p.stdout)
    # This should not happen, but it does. See docstring.
    assert not next((pkg for pkg in data["actions"]["LINK"] if pkg["name"] == "cudatoolkit"), None)


@pytest.mark.skipif(not on_linux, reason="Slow test, only run on Linux")
def test_old_panel(tmp_path):
    """
    https://github.com/conda/conda-libmamba-solver/issues/64

    We could not reproduce this test until #381. Note this is not a problem
    in the non-time-machine'd repodata (as of 2023-11-16 at least).
    """
    os.chdir(tmp_path)
    print("Patching repodata...")
    old_repodata = os.path.abspath(
        repodata_time_machine(
            channels=["conda-forge", "pyviz/label/dev"],
            timestamp_str="2022-06-16 12:31:00",
            subdirs=("osx-64", "noarch"),
        )
    )
    with open(f"{old_repodata}/conda-forge/osx-64/repodata.json") as f:
        for line in f:
            # Make sure we have patched the repodata correctly
            # Python 3.11 only appeared in October 2022
            assert '"python-3.11.0-' not in line

    channel_prefix = f"file://{old_repodata}/"
    env = os.environ.copy()
    env["CONDA_SUBDIR"] = "osx-64"
    env["CONDA_REPODATA_THREADS"] = "1"
    env["CONDA_DEFAULT_THREADS"] = "1"
    env["CONDA_FETCH_THREADS"] = "1"
    env["CONDA_REMOTE_CONNECT_TIMEOUT_SECS"] = "1"
    env["CONDA_REMOTE_MAX_RETRIES"] = "1"
    env["CONDA_REMOTE_BACKOFF_FACTOR"] = "1"
    env["CONDA_REMOTE_READ_TIMEOUT_SECS"] = "1"
    args = (
        "create",
        "-n",
        "unused",
        "--dry-run",
        "--json",
        "--override-channels",
        "-c",
        f"{channel_prefix}pyviz/label/dev",
        "-c",
        f"{channel_prefix}conda-forge",
        "--repodata-fn=repodata.json",
    )
    pkgs = (
        "python=3.8",
        "lumen",
    )

    for solver in ("classic", "libmamba"):
        print("Solving with", solver)
        p = conda_subprocess(
            *args,
            "--solver",
            solver,
            *pkgs,
            env=env,
        )
        data = json.loads(p.stdout)
        panel = next(pkg for pkg in data["actions"]["LINK"] if pkg["name"] == "panel")
        assert panel["version"] == "0.14.0a2"
