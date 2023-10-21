# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
import json
import os
import sys
from itertools import chain, permutations, repeat
from pathlib import Path
from subprocess import check_call, run
from textwrap import dedent
from uuid import uuid4

import pytest
from conda.base.context import context
from conda.common.compat import on_linux, on_win
from conda.common.io import env_var
from conda.core.prefix_data import PrefixData, get_python_version_for_prefix
from conda.testing.integration import Commands, make_temp_env, run_command
from conda.testing.solver_helpers import SolverTests

from conda_libmamba_solver import LibMambaSolver
from conda_libmamba_solver.mamba_utils import mamba_version

from .utils import conda_subprocess


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


@pytest.mark.xfail(
    mamba_version().startswith("1.5."),
    reason="Known bug. See https://github.com/mamba-org/mamba/issues/2431",
    strict=True,
)
def test_defaults_specs_work():
    """
    See https://github.com/conda/conda-libmamba-solver/issues/173

    `conda install defaults::<pkg_name>` fails with libmamba due to a
    mapping issue between conda and libmamba.Repo channel names.
    defaults is secretly (main, r and msys2), and repos are built using those
    actual channels. A bug in libmamba fails to map this relationship.

    We are testing our workaround (https://github.com/conda/conda-libmamba-solver/issues/173)
    works for now, but we should probably help fix this in libmamba.
    """
    out, err, rc = run_command(
        "create",
        "unused",
        "--dry-run",
        "--json",
        "--solver=libmamba",
        "--override-channels",
        "-c",
        "conda-forge",
        "python=3.10",
        "defaults::libarchive",
        use_exception_handler=True,
    )
    data = json.loads(out)
    assert data.get("success") is True
    for link in data["actions"]["LINK"]:
        if link["name"] == "libarchive":
            assert link["channel"] in ("defaults", "pkgs/main")
            break
    else:
        raise AssertionError("libarchive not found in LINK actions")


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


def test_update_from_latest_not_downgrade(tmpdir):
    """Based on two issues where an upgrade caused a downgrade in a given package

    Suppose we have two python versions 3.11.2 and 3.11.3. The bug is when:
    $ conda install python | grep python
    python 3.11.3
    $ conda update python | grep python
    python 3.11.2

    Update should not downgrade the package
     - https://github.com/conda/conda-libmamba-solver/issues/71
     - https://github.com/conda/conda-libmamba-solver/issues/156
    """
    with make_temp_env(
        "--override-channels",
        "-c",
        "conda-forge",
        "--solver=libmamba",
        "python",
        no_capture=True,
    ) as prefix:
        original_python = PrefixData(prefix).get("python")
        run_command(
            Commands.UPDATE,
            prefix,
            "--solver=libmamba",
            "--override-channels",
            "-c",
            "conda-forge",
            "python",
            no_capture=True,
        )
        update_python = PrefixData(prefix).get("python")
        assert original_python.version == update_python.version


@pytest.mark.skipif(not on_linux, reason="Linux only")
def test_too_aggressive_update_to_conda_forge_packages():
    """
    Comes from report in https://github.com/conda/conda-libmamba-solver/issues/240
    We expect a minimum change to the 'base' environment if we only ask for a single package.
    conda classic would just change a few (<5) packages, but libmamba seemed to upgrade
    EVERYTHING it can to conda-forge.
    """
    with make_temp_env("conda", "python", "--override-channels", "--channel=defaults") as prefix:
        cmd = (
            "install",
            "-p",
            prefix,
            "-c",
            "conda-forge",
            "libzlib",
            "--json",
            "--dry-run",
            "-y",
            "-vvv",
        )
        env = os.environ.copy()
        env.pop("CONDA_SOLVER", None)
        p_classic = conda_subprocess(*cmd, "--solver=classic", explain=True, env=env)
        p_libmamba = conda_subprocess(*cmd, "--solver=libmamba", explain=True, env=env)
        data_classic = json.loads(p_classic.stdout)
        data_libmamba = json.loads(p_libmamba.stdout)
        assert len(data_classic["actions"]["LINK"]) < 15
        assert len(data_libmamba["actions"]["LINK"]) <= len(data_classic["actions"]["LINK"])


@pytest.mark.skipif(context.subdir != "linux-64", reason="Linux-64 only")
def test_pinned_with_cli_build_string():
    """ """
    specs = (
        "scipy=1.7.3=py37hf2a6cf1_0",
        "python=3.7.3",
        "pandas=1.2.5=py37h295c915_0",
    )
    channels = (
        "--override-channels",
        "--channel=conda-forge",
        "--channel=defaults",
    )
    with make_temp_env(*specs, *channels) as prefix:
        Path(prefix, "conda-meta").mkdir(exist_ok=True)
        Path(prefix, "conda-meta", "pinned").write_text(
            dedent(
                """
                python ==3.7.3
                pandas ==1.2.5 py37h295c915_0
                scipy ==1.7.3 py37hf2a6cf1_0
                """
            ).lstrip()
        )
        # We ask for the same packages or name-only, it should be compatible
        for valid_specs in (specs, ("python", "pandas", "scipy")):
            p = conda_subprocess(
                "install",
                "-p",
                prefix,
                *valid_specs,
                *channels,
                "--dry-run",
                "--json",
                explain=True,
                check=False,
            )
            data = json.loads(p.stdout)
            assert data.get("success")
            assert data["message"] == "All requested packages already installed."

        # However if we ask for a different version, it should fail
        invalid_specs = ("python=3.8", "pandas=1.2.4", "scipy=1.7.2")
        p = conda_subprocess(
            "install",
            "-p",
            prefix,
            *invalid_specs,
            *channels,
            "--dry-run",
            "--json",
            explain=True,
            check=False,
        )
        data = json.loads(p.stdout)
        assert not data.get("success")
        assert data["exception_name"] == "SpecsConfigurationConflictError"

        non_existing_specs = ("python=0", "pandas=1000", "scipy=24")
        p = conda_subprocess(
            "install",
            "-p",
            prefix,
            *non_existing_specs,
            *channels,
            "--dry-run",
            "--json",
            explain=True,
            check=False,
        )
        data = json.loads(p.stdout)
        assert not data.get("success")
        assert data["exception_name"] == "PackagesNotFoundError"


def test_constraining_pin_and_requested():
    env = os.environ.copy()
    env["CONDA_PINNED_PACKAGES"] = "python=3.9"

    # This should fail because it contradicts the pinned packages
    p = conda_subprocess(
        "create",
        "-n",
        "unused",
        "--dry-run",
        "--json",
        "python=3.10",
        "--override-channels",
        "-c",
        "conda-forge",
        env=env,
        explain=True,
        check=False,
    )
    data = json.loads(p.stdout)
    assert not data.get("success")
    assert data["exception_name"] == "SpecsConfigurationConflictError"

    # This is ok because it's a no-op
    p = conda_subprocess(
        "create",
        "-n",
        "unused",
        "--dry-run",
        "--json",
        "python",
        env=env,
        explain=True,
        check=False,
    )
    data = json.loads(p.stdout)
    assert data.get("success")
    assert data.get("dry_run")


def test_locking_pins():
    with env_var("CONDA_PINNED_PACKAGES", "zlib"), make_temp_env("zlib") as prefix:
        # Should install just fine
        zlib = PrefixData(prefix).get("zlib")
        assert zlib

        # This should fail because it contradicts the lock packages
        out, err, retcode = run_command(
            "install",
            prefix,
            "zlib=1.2.11",
            "--dry-run",
            "--json",
            use_exception_handler=True,
        )
        data = json.loads(out)
        assert retcode
        assert data["exception_name"] == "SpecsConfigurationConflictError"
        assert str(zlib) in data["error"]

        # This is a no-op and ok. It won't involve changes.
        out, err, retcode = run_command(
            "install", prefix, "zlib", "--dry-run", "--json", use_exception_handler=True
        )
        data = json.loads(out)
        assert not retcode
        assert data.get("success")
        assert data["message"] == "All requested packages already installed."


def test_ca_certificates_pins():
    ca_certificates_pin = "ca-certificates=2023"
    with make_temp_env() as prefix:
        Path(prefix, "conda-meta").mkdir(exist_ok=True)
        Path(prefix, "conda-meta", "pinned").write_text(f"{ca_certificates_pin}\n")

        for cli_spec in (
            "ca-certificates",
            "ca-certificates=2023",
            "ca-certificates>0",
            "ca-certificates<2024",
            "ca-certificates!=2022",
        ):
            out, err, retcode = run_command(
                "install",
                prefix,
                cli_spec,
                "--dry-run",
                "--json",
                "--override-channels",
                "-c",
                "conda-forge",
                use_exception_handler=True,
            )
            data = json.loads(out)
            assert not retcode
            assert data.get("success")
            assert data.get("dry_run")

            for pkg in data["actions"]["LINK"]:
                if pkg["name"] == "ca-certificates":
                    assert pkg["version"].startswith("2023.")
                    break
            else:
                raise AssertionError("ca-certificates not found in LINK actions")
