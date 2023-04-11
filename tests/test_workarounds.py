# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
import json
import sys
from subprocess import PIPE, check_call, run


def test_matchspec_star_version():
    """
    Specs like `libblas=*=*mkl` choked on `MatchSpec.conda_build_form()`.
    We work around that with `.utils.safe_conda_build_form()`.
    Reported in https://github.com/conda/conda/issues/11347
    """
    check_call(
        [
            sys.executable,
            "-m",
            "conda",
            "create",
            "-p",
            "UNUSED",
            "--dry-run",
            "--override-channels",
            "-c",
            "conda-test",
            "--solver=libmamba",
            "activate_deactivate_package=*=*0",
        ]
    )


def test_build_string_filters():
    process = run(
        [
            sys.executable,
            "-m",
            "conda",
            "create",
            "-p",
            "UNUSED",
            "--dry-run",
            "--solver=libmamba",
            "numpy=*=*py38*",
            "--json",
        ],
        stdout=PIPE,
        text=True,
    )
    print(process.stdout)
    process.check_returncode()
    data = json.loads(process.stdout)
    assert data["success"]
    for pkg in data["actions"]["LINK"]:
        if pkg["name"] == "python":
            assert pkg["version"].startswith("3.8")
        if pkg["name"] == "numpy":
            assert "py38" in pkg["build_string"]
