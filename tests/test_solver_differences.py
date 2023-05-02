"""
This module collects examples of environments that were hard to solve, required
workarounds or didn't meet users' expectations... specially if compared to conda classic.
"""
import json
import os

from conda.common.compat import on_linux
import pytest

from .utils import conda_subprocess


@pytest.mark.skipif(not on_linux, reason="Only relevant on Linux")
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
    assert pydantic["version"] != "1.8.2"

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

