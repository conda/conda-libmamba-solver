# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
HTTP shard server fixtures for libmamba solver tests.

Shard implementation and tests live in conda; this module only builds mock
repos for channel-ordering and similar integration tests.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Callable
from pathlib import Path

import msgpack
import pytest
import zstandard
from conda.models.channel import Channel
from conda.testing import http_test_server

from conda.gateways.repodata.shards.typing import ShardsIndexDict

CONDA_FORGE_WITH_SHARDS = "conda-forge"

FAKE_REPODATA = {
    "info": {"subdir": "noarch", "base_url": "", "shards_base_url": ""},
    "packages": {
        "foo.tar.bz2": {
            "name": "foo",
            "version": "1",
            "build": "0_a",
            "build_number": 0,
            "depends": ["bar", "baz"],
        },
        "bar.tar.bz2": {
            "name": "bar",
            "version": "1",
            "build": "0_a",
            "build_number": 0,
            "depends": ["foo"],
        },
        "no-matching-conda.tar.bz2": {
            "name": "foo",
            "version": "0.1",
            "build": "0_a",
            "build_number": 0,
        },
    },
    "packages.conda": {
        "foo.conda": {
            "name": "foo",
            "version": "1",
            "build": "0_a",
            "build_number": 0,
            "depends": ["bar", "baz"],
            "constrains": ["splat<3"],
            "sha256": hashlib.sha256().digest(),
        },
        "bar.conda": {
            "name": "bar",
            "version": "1",
            "build": "0_a",
            "build_number": 0,
            "depends": ["foo"],
            "constrains": ["splat<3"],
            "sha256": hashlib.sha256().digest(),
        },
        "no-matching-tar-bz2.conda": {
            "name": "foo",
            "version": "2",
            "build": "0_a",
            "build_number": 0,
            "depends": ["quux", "warble"],
            "constrains": ["splat<3"],
            "sha256": hashlib.sha256().digest(),
        },
    },
    "repodata_version": 2,
}


def shard_for_name(repodata, name):
    return {
        group: {k: v for (k, v) in repodata[group].items() if v["name"] == name}
        for group in ("packages", "packages.conda")
    }


FAKE_SHARD = shard_for_name(FAKE_REPODATA, "foo")
FAKE_SHARD_2 = shard_for_name(FAKE_REPODATA, "bar")


class ShardFactory:
    """Create temporary HTTP servers serving minimal sharded repodata."""

    def __init__(self, root: Path = tempfile.gettempdir()):
        self.root = root
        self._http_servers = []

    def clean_up_http_servers(self):
        for http in self._http_servers:
            http.shutdown()
        self._http_servers = []

    def http_server_shards(
        self, dir_name: str, finish_request_action: Callable | None = None
    ) -> str:
        shards_repository = self.root / dir_name / "sharded_repo"
        shards_repository.mkdir(parents=True)
        noarch = shards_repository / "noarch"
        noarch.mkdir()

        foo_shard = zstandard.compress(msgpack.dumps(FAKE_SHARD))  # type: ignore
        foo_shard_digest = hashlib.sha256(foo_shard).digest()
        (noarch / f"{foo_shard_digest.hex()}.msgpack.zst").write_bytes(foo_shard)

        bar_shard = zstandard.compress(msgpack.dumps(FAKE_SHARD_2))  # type: ignore
        bar_shard_digest = hashlib.sha256(bar_shard).digest()
        (noarch / f"{bar_shard_digest.hex()}.msgpack.zst").write_bytes(bar_shard)

        malformed = {"follows_schema": False}
        bad_schema = zstandard.compress(msgpack.dumps(malformed))  # type: ignore
        malformed_digest = hashlib.sha256(bad_schema).digest()

        (noarch / f"{malformed_digest.hex()}.msgpack.zst").write_bytes(bad_schema)
        not_zstd = b"not zstd"
        (noarch / f"{hashlib.sha256(not_zstd).digest().hex()}.msgpack.zst").write_bytes(not_zstd)
        not_msgpack = zstandard.compress(b"not msgpack")
        (noarch / f"{hashlib.sha256(not_msgpack).digest().hex()}.msgpack.zst").write_bytes(
            not_msgpack
        )
        fake_shards: ShardsIndexDict = {
            "info": {"subdir": "noarch", "base_url": "", "shards_base_url": ""},
            "version": 1,
            "shards": {
                "foo": foo_shard_digest,
                "bar": bar_shard_digest,
                "wrong_package_name": foo_shard_digest,
                "fake_package": b"",
                "malformed": hashlib.sha256(bad_schema).digest(),
                "not_zstd": hashlib.sha256(not_zstd).digest(),
                "not_msgpack": hashlib.sha256(not_msgpack).digest(),
            },
        }
        (shards_repository / "noarch" / "repodata_shards.msgpack.zst").write_bytes(
            zstandard.compress(msgpack.dumps(fake_shards))  # type: ignore
        )

        http = http_test_server.run_test_server(
            str(shards_repository), finish_request_action=finish_request_action
        )
        self._http_servers.append(http)

        host, port = http.socket.getsockname()[:2]
        url_host = f"[{host}]" if ":" in host else host
        return f"http://{url_host}:{port}/"


@pytest.fixture(scope="session")
def shard_factory(tmp_path_factory, request: pytest.FixtureRequest) -> ShardFactory:
    shards_repository = tmp_path_factory.mktemp("sharded_repo")
    factory = ShardFactory(shards_repository)

    def close_servers():
        factory.clean_up_http_servers()

    request.addfinalizer(close_servers)
    return factory
