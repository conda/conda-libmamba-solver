# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest
from conda.testing import http_test_server as http_server_module
from conda.testing.fixtures import HttpTestServerFixture

if TYPE_CHECKING:
    from collections.abc import Iterator

pytest_plugins = (
    # Add testing fixtures and internal pytest plugins here
    "conda.testing",
    "conda.testing.fixtures",
)

# Shard-related fixtures have been removed (shards module being deprecated)


@pytest.fixture(scope="module")
def mamba_repo_server(tmp_path_factory) -> Iterator[HttpTestServerFixture]:
    """Module-scoped server pre-populated with mamba_repo + token subpath."""
    from .http_channel_helpers import MAMBA_REPO, TOKEN

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
