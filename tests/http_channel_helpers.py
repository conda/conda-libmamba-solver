# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conda.testing.fixtures import HttpTestServerFixture

TOKEN = "xy-12345678-1234-1234-1234-123456789012"
MAMBA_REPO = Path(__file__).parent / "data" / "mamba_repo"


def setup_mamba_repo(server: HttpTestServerFixture) -> None:
    shutil.copytree(MAMBA_REPO, server.directory, dirs_exist_ok=True)


def setup_token_channel(server: HttpTestServerFixture, token: str = TOKEN) -> str:
    dest = server.directory / "t" / token
    shutil.copytree(MAMBA_REPO, dest, dirs_exist_ok=True)
    return f"{server.url}/t/{token}"


def basic_auth_url(server: HttpTestServerFixture, user: str, password: str) -> str:
    host = server.url.removeprefix("http://")
    return f"http://{user}:{password}@{host}"
