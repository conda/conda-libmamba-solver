# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
This file tests for the different user-agents conda and conda-libmamba-solver
might present to a server. We mock the server via `_reposerver.py`.

There are two request backend we can end up using:
* libcurl, via libmamba
* requests, via conda

From conda-libmamba-solver 23.3, we only use conda internals (requests).

With libmamba 0.23 and conda 4.13, we can customize the user agent. The user
agent will look slightly different depending on the network stack, so we have
up to three possible user agent strings:

* Conda classic will always have the same user agent:
    conda/4.12.0.post61+7acaba230 requests/2.27.1 CPython/3.8.13
    Linux/5.10.25-linuxkit debian/10 glibc/2.28
* Conda with libmamba enabled:
    conda/4.12.0.post61+7acaba230 requests/2.27.1 CPython/3.8.13
    Linux/5.10.25-linuxkit debian/10 glibc/2.28
    solver/libmamba conda-libmamba-solver/22.3.1 libmambapy/0.24.0

Note how conda classic's user agent is present in all three possibilities. When
the libmamba solver is enabled, we extend it with `solver/libmamba` plus the
versions of conda-libmamba-solver and libmambapy.

When using `mamba` directly in the CLI, its user agent will start with `mamba/<version>`.
"""

import json
import os
import sys
from subprocess import PIPE, check_output, run

import pytest
from importlib_metadata import version

from .channel_testing_utils import (
    create_with_channel,
    http_server_auth_none_debug_packages,
    http_server_auth_none_debug_repodata,
)


@pytest.mark.parametrize("solver", ("classic", "libmamba"))
def test_user_agent_conda_info(solver):
    env = os.environ.copy()
    env["CONDA_SOLVER"] = solver
    out = check_output(["conda", "info", "--json"], env=env)
    info = json.loads(out)
    assert "conda/" in info["user_agent"]
    if solver == "classic":
        assert "solver/" not in info["user_agent"]
    elif solver == "libmamba":
        assert "solver/libmamba" in info["user_agent"]
        assert f"conda-libmamba-solver/{version('conda-libmamba-solver')}" in info["user_agent"]
        assert f"libmambapy/{version('libmambapy')}" in info["user_agent"]


def assert_requests_user_agent(stdout, solver, request_type="repodata"):
    """
    With conda classic, the request handler uses `requests`. An invalid response from the server
    will raise an exception that already contains the User-Agent headers in the payload.
    """
    # the output might contain more than one json dict
    # tip: conda separates json payloads with nul (\x00)
    # we split by nul and keep the last chunk surrounded by {}
    payload = None
    for chunk in stdout.split("\x00"):
        chunk = chunk.strip()
        if chunk.startswith("{") and chunk.endswith("}"):
            payload = chunk
    if payload is None:
        raise AssertionError(f"Could not find a JSON payload in the output: {stdout}")
    data = json.loads(payload)
    if request_type == "repodata":
        assert data["exception_name"] == "UnavailableInvalidChannel"
        assert data["reason"] == "FILE NOT FOUND"
        details = data["response_details"]
    elif request_type == "packages":
        assert data["error"] == "Multiple Errors Encountered."
        assert data["exception_name"] == "CondaMultiError"
        error = data["errors"][0]
        assert error["exception_name"] == "CondaHTTPError"
        assert error["reason"] == "FILE NOT FOUND"
        details = error["response_details"]

    for line in details.splitlines():
        if line.startswith("> User-Agent:"):
            assert "conda/" in line
            if solver == "classic":
                assert "solver/" not in line
            elif solver == "libmamba":
                assert "conda/" in line
                assert "solver/libmamba" in line
                assert f"conda-libmamba-solver/{version('conda-libmamba-solver')}" in line
                assert f"libmambapy/{version('libmambapy')}" in line
                assert "requests/" in line
            break
    else:
        raise AssertionError("Couldn't find the User-Agent info in headers!")


def test_user_agent_libmamba_packages(http_server_auth_none_debug_packages, tmp_path):
    run([sys.executable, "-m", "conda", "clean", "--tarballs", "--packages", "--yes"])
    env = os.environ.copy()
    env["CONDA_PKGS_DIRS"] = str(tmp_path)
    process = create_with_channel(
        http_server_auth_none_debug_packages,
        solver="libmamba",
        check=False,
        stdout=PIPE,
        stderr=PIPE,
        text=True,
        env=env,
    )
    print("-- STDOUT --")
    print(process.stdout)
    print("-- STDERR --")
    print(process.stderr)
    assert_requests_user_agent(process.stdout, solver="libmamba", request_type="packages")


def test_user_agent_classic_repodata(http_server_auth_none_debug_repodata):
    process = create_with_channel(
        http_server_auth_none_debug_repodata,
        solver="classic",
        check=False,
        stdout=PIPE,
        stderr=PIPE,
        text=True,
    )
    assert_requests_user_agent(process.stdout, solver="classic")


def test_user_agent_classic_packages(http_server_auth_none_debug_packages, tmp_path):
    run([sys.executable, "-m", "conda", "clean", "--tarballs", "--packages", "--yes"])

    env = os.environ.copy()
    env["CONDA_PKGS_DIRS"] = str(tmp_path)
    process = create_with_channel(
        http_server_auth_none_debug_packages,
        solver="classic",
        check=False,
        stdout=PIPE,
        stderr=PIPE,
        text=True,
        env=env,
    )
    print("-- STDOUT --")
    print(process.stdout)
    print("-- STDERR --")
    print(process.stderr)
    assert_requests_user_agent(process.stdout, solver="classic", request_type="packages")
