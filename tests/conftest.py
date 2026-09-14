# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
import platform
import shutil
import sys
import sysconfig
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from conda.base.context import context, reset_context
from conda.core.prefix_data import PrefixData
from conda.testing import http_test_server as http_server_module
from conda.testing.fixtures import HttpTestServerFixture
from libmambapy import bindings

import conda_libmamba_solver

from .http_channel_helpers import MAMBA_REPO, TOKEN

if TYPE_CHECKING:
    from collections.abc import Iterator

pytest_plugins = (
    # Add testing fixtures and internal pytest plugins here
    "conda.testing",
    "conda.testing.fixtures",
)

# Shard-related fixtures have been removed (shards module being deprecated)


def pytest_report_header():
    if expected_subdir := os.environ.get("CONDA_TEST_SUBDIR"):
        assert context.subdir == expected_subdir, context.subdir
        prefix_data = PrefixData(sys.prefix)
        for name in ("python", "libmamba", "libmambapy"):
            assert prefix_data.get(name).subdir == expected_subdir, name
        assert {record.subdir for record in prefix_data.iter_records()} <= {
            "noarch",
            expected_subdir,
        }
        assert Path(conda_libmamba_solver.__file__).resolve().parent == (
            Path(__file__).resolve().parents[1] / "conda_libmamba_solver"
        )
        if expected_subdir == "win-arm64":
            assert platform.machine().lower() == "arm64", platform.machine()
            assert sysconfig.get_platform() == "win-arm64", sysconfig.get_platform()
        assert Path(bindings.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())

    return [
        f"Python platform: {sysconfig.get_platform()}",
        f"libmambapy extension: {bindings.__file__}",
        f"conda-libmamba-solver source: {conda_libmamba_solver.__file__}",
    ]


@pytest.fixture
def historical_python_subdir(monkeypatch):
    """Solve historical Python regressions with win-64 packages on Windows ARM64."""
    if context.subdir == "win-arm64":
        monkeypatch.setenv("CONDA_SUBDIR", "win-64")
        reset_context()


@pytest.fixture(scope="module")
def mamba_repo_server(tmp_path_factory) -> Iterator[HttpTestServerFixture]:
    """Module-scoped server pre-populated with mamba_repo + token subpath."""

    server_dir = tmp_path_factory.mktemp("mamba_repo_server")
    shutil.copytree(MAMBA_REPO, server_dir, dirs_exist_ok=True)
    token_dest = server_dir / "t" / TOKEN
    shutil.copytree(MAMBA_REPO, token_dest, dirs_exist_ok=True)
    server = http_server_module.run_test_server(str(server_dir))
    host, port = server.socket.getsockname()[:2]
    url_host = f"[{host}]" if ":" in host else host
    yield HttpTestServerFixture(
        server=server,
        host=host,
        port=port,
        url=f"http://{url_host}:{port}",
        directory=server_dir,
    )
    server.shutdown()
