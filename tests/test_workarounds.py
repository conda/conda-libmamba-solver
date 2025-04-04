# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
import ctypes
import json
import signal
import subprocess as sp
import sys
import time

import pytest
from conda.common.compat import on_win


def test_matchspec_star_version():
    """
    Specs like `libblas=*=*mkl` choked on `MatchSpec.conda_build_form()`.
    We work around that with `.utils.safe_conda_build_form()`.
    Reported in https://github.com/conda/conda/issues/11347
    """
    sp.check_call(
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
    process = sp.run(
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
        stdout=sp.PIPE,
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


@pytest.mark.parametrize("stage", ["Collecting package metadata", "Solving environment"])
def test_ctrl_c(stage):
    TIMEOUT = 15  # Used twice in total, so account for double the amount
    p = sp.Popen(
        [
            sys.executable,
            "-m",
            "conda",
            "create",
            "-p",
            "UNUSED",
            "--dry-run",
            "--solver=libmamba",
            "--override-channels",
            "--channel=conda-forge",
            "--quiet",
            "vaex",
        ],
        text=True,
        stdout=sp.PIPE,
        stderr=sp.PIPE,
    )
    t0 = time.time()
    while stage not in p.stdout.readline():
        time.sleep(0.1)
        if time.time() - t0 > TIMEOUT:
            raise RuntimeError("Timeout")

    # works around Windows' awkward CTRL-C signal handling
    # https://stackoverflow.com/a/64357453
    if on_win:
        try:
            kernel = ctypes.windll.kernel32
            kernel.FreeConsole()
            kernel.AttachConsole(p.pid)
            kernel.SetConsoleCtrlHandler(None, 1)
            kernel.GenerateConsoleCtrlEvent(0, 0)
            p.wait(timeout=TIMEOUT)
        finally:
            kernel.SetConsoleCtrlHandler(None, 0)
    else:
        p.send_signal(signal.SIGINT)
        p.wait(timeout=TIMEOUT)

    assert p.returncode != 0
    assert "KeyboardInterrupt" in p.stdout.read() + p.stderr.read()
