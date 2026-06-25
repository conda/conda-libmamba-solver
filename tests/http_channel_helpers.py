# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conda.testing.fixtures import HttpTestServerFixture

TOKEN = "xy-12345678-1234-1234-1234-123456789012"
MAMBA_REPO = Path(__file__).parent / "data" / "mamba_repo"


def basic_auth_url(server: HttpTestServerFixture, user: str, password: str) -> str:
    host = server.url.removeprefix("http://")
    return f"http://{user}:{password}@{host}"
