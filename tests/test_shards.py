# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Test sharded repodata.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

import msgpack
import pytest
import zstandard
from conda.base.context import context, reset_context
from conda.core.subdir_data import SubdirData
from conda.models.channel import Channel

from conda_libmamba_solver import shards_cache
from conda_libmamba_solver.shards import (
    RepodataDict,
    ShardLike,
    Shards,
    fetch_shards,
    shard_mentioned_packages,
)
from tests.channel_testing.helpers import _dummy_http_server

if TYPE_CHECKING:
    from conda_libmamba_solver.shards import ShardsIndex

HERE = Path(__file__).parent


def package_names(shard: shards_cache.Shard):
    """
    All package names mentioned in a shard (should be a single package name)
    """
    return set(package["name"] for package in shard["packages"].values()) | set(
        package["name"] for package in shard["packages.conda"].values()
    )


@pytest.fixture
def conda_no_token(monkeypatch: pytest.MonkeyPatch):
    """
    Reset token to avoid being logged in. e.g. the testing channel doesn't understand them.
    """
    monkeypatch.setenv("CONDA_TOKEN", "")
    reset_context()


@pytest.fixture
def http_server_shards(xprocess, tmp_path_factory):
    """
    A shard repository with a difference.
    """
    shards_repository = tmp_path_factory.mktemp("sharded_repo")
    (shards_repository / "noarch").mkdir()
    malformed = {"follows_schema": False}
    malformed_bytes = zstandard.compress(msgpack.dumps(malformed))  # type: ignore
    # XXX not-zstandard; not msgpack
    malformed_digest = hashlib.sha256(malformed_bytes).digest()
    (shards_repository / "noarch" / f"{malformed_digest.hex()}.msgpack.zst").write_bytes(
        malformed_bytes
    )
    fake_shards: ShardsIndex = {
        "info": {"subdir": "noarch", "base_url": "", "shards_base_url": ""},
        "repodata_version": 1,
        "shards": {"fake_package": b"", "malformed": hashlib.sha256(malformed_bytes).digest()},
        "removed": [],
    }
    (shards_repository / "noarch" / "repodata_shards.msgpack.zst").write_bytes(
        zstandard.compress(msgpack.dumps(fake_shards))  # type: ignore
    )
    yield from _dummy_http_server(
        xprocess, name="http_server_auth_none", port=0, auth="none", path=shards_repository
    )


def test_fetch_shards_error(http_server_shards):
    channel = Channel.from_url(f"{http_server_shards}/noarch")
    subdir_data = SubdirData(channel)
    found = fetch_shards(subdir_data)
    assert found

    with pytest.raises(zstandard.ZstdError):
        # XXX we tried to decompress a 404 response
        found.fetch_shard("fake_package")

    malo = found.fetch_shard("malformed")
    assert malo == {"follows_schema": False}  # XXX should we return None or raise


def test_shards(conda_no_token: None):
    """
    Test basic shard fetch.
    """
    channels = [
        Channel.from_url(f"https://conda.anaconda.org/conda-forge-sharded/{subdir}")
        for subdir in context.subdirs
    ]

    subdir_data = SubdirData(channels[0])
    found = fetch_shards(subdir_data)
    assert found, f"Shards not found for {channels[0]}"

    for package in found.packages_index:
        shard_url = found.shard_url(package)
        assert shard_url.startswith("http")  # or channel url or shards_base_url

    # download or fetch-from-cache a random set of shards

    for package in random.choices([*found.packages_index.keys()], k=16):
        shard = found.fetch_shard(package)

        mentioned_in_shard = shard_mentioned_packages(shard)
        assert (
            package in mentioned_in_shard
        )  # the package's own name is mentioned, as well as any dependencies.
        print(package_names(shard), mentioned_in_shard)


def test_fetch_shards(conda_no_token: None):
    """
    Test all channels fetch as Shards or ShardLike, depending on availability.
    """
    channels = list(context.default_channels)
    print(channels)

    channels.append(Channel("conda-forge-sharded"))

    channel_data: dict[str, ShardLike] = {}
    for channel in channels:
        for channel_url in Channel(channel).urls(True, context.subdirs):
            subdir_data = SubdirData(Channel(channel_url))
            found = fetch_shards(subdir_data)
            if not found:
                repodata_json, _ = subdir_data.repo_fetch.fetch_latest_parsed()
                repodata_json = RepodataDict(repodata_json)  # type: ignore
                found = ShardLike(repodata_json, channel_url)
            channel_data[channel_url] = found

    # at least one should be real shards, not repodata.json presented as shards.
    assert any(isinstance(channel, Shards) for channel in channel_data.values())


def test_shard_cache(tmp_path: Path):
    cache = shards_cache.ShardCache(tmp_path)

    fake_shard = {"foo": "bar"}
    annotated_shard = shards_cache.AnnotatedRawShard(
        "https://foo",
        "foo",
        zstandard.compress(msgpack.dumps(fake_shard)),  # type: ignore
    )
    cache.insert(annotated_shard)

    data = cache.retrieve(annotated_shard.url)
    assert data == fake_shard
    assert data is not fake_shard

    data2 = cache.retrieve("notfound")
    assert data2 is None

    assert (tmp_path / shards_cache.SHARD_CACHE_NAME).exists()


def test_shard_cache_multiple(tmp_path: Path):
    """
    Test that retrieve_multiple() is equivalent to several retrieve() calls.
    """
    NUM_FAKE_SHARDS = 64

    cache = shards_cache.ShardCache(tmp_path)
    fake_shards = []

    compressor = zstandard.ZstdCompressor(level=1)
    for i in range(NUM_FAKE_SHARDS):
        fake_shard = {f"foo{i}": "bar"}
        annotated_shard = shards_cache.AnnotatedRawShard(
            f"https://foo{i}",
            f"foo{i}",
            compressor.compress(msgpack.dumps(fake_shard)),  # type: ignore
        )
        cache.insert(annotated_shard)
        fake_shards.append(annotated_shard)

    start_multiple = time.monotonic_ns()
    retrieved = cache.retrieve_multiple([shard.url for shard in fake_shards])
    end_multiple = time.monotonic_ns()

    assert len(retrieved) == NUM_FAKE_SHARDS

    print(
        f"retrieve {len(fake_shards)} shards in a single call: {(end_multiple - start_multiple) / 1e9:0.6f}s"
    )

    start_single = time.monotonic_ns()
    for i, url in enumerate([shard.url for shard in fake_shards]):
        single = cache.retrieve(url)
        assert retrieved[url] == single
    end_single = time.monotonic_ns()
    print(
        f"retrieve {len(fake_shards)} shards with multiple calls: {(end_single - start_single) / 1e9:0.6f}s"
    )

    if (end_single - start_single) != 0:  # avoid ZeroDivisionError
        print(
            f"Multiple API takes {(end_multiple - start_multiple) / (end_single - start_single):.2f} times as long."
        )

        assert (end_multiple - start_multiple) / (end_single - start_single) < 1, (
            "batch API took longer"
        )

    assert (tmp_path / shards_cache.SHARD_CACHE_NAME).exists()


def test_shard_cache_2():
    """
    Check that existing shards cache has been populated with matching shards,
    package names.
    """
    subdir_data = SubdirData(Channel("main/noarch"))
    # accepts directory that cache goes into, not database filename
    cache = shards_cache.ShardCache(Path(subdir_data.cache_path_base).parent)
    extant = {}
    for row in cache.conn.execute("SELECT url, package, shard, timestamp FROM shards"):
        extant[row["url"]] = shards_cache.AnnotatedRawShard(
            url=row["url"], package=row["package"], compressed_shard=row["shard"]
        )

    # check that url's are in the url field, and package names are in the package name field
    for url in extant:
        if shard := cache.retrieve(url):
            assert "://" in extant[url].url
            assert "://" not in extant[url].package
            assert all(
                package["name"] == extant[url].package for package in shard["packages"].values()
            )
            assert all(
                package["name"] == extant[url].package
                for package in shard["packages.conda"].values()
            )


def test_shardlike():
    """
    ShardLike class presents repodata.json as shards in a way that is suitable
    for our subsetting algorithm.
    """
    repodata = json.loads(
        (Path(__file__).parent / "data" / "mamba_repo" / "noarch" / "repodata.json").read_text()
    )

    # make fake packages
    for n in range(10):
        for m in range(n):  # 0 test0's
            repodata["packages"][f"test{n}{m}.tar.bz2"] = {"name": f"test{n}"}
            repodata["packages.conda"][f"test{n}{m}.tar.bz2"] = {"name": f"test{n}"}

    as_shards = ShardLike(repodata)

    assert len(as_shards.repodata_no_packages)
    assert len(as_shards.shards)

    assert sorted(as_shards.shards["test4"]["packages"].keys()) == [
        "test40.tar.bz2",
        "test41.tar.bz2",
        "test42.tar.bz2",
        "test43.tar.bz2",
    ]

    fetched_shard = as_shards.fetch_shard("test1")
    assert fetched_shard["packages"]["test10.tar.bz2"]["name"] == "test1"
    assert as_shards.url in repr(as_shards)
    assert "test1" in as_shards

    fetched_shards = as_shards.fetch_shards(("test1", "test2"))
    assert len(fetched_shards) == 2
    assert fetched_shards["test1"]
    assert fetched_shards["test2"]

    as_shards.visited.update(fetched_shards)
    as_shards.visited["package-that-does-not-exist"] = None
    repodata = as_shards.build_repodata()
    assert len(repodata["packages"]) == 3
    assert len(repodata["packages.conda"]) == 3


def test_shardlike_repr():
    shardlike = ShardLike(
        {
            "packages": {},
            "packages.conda": {},
            "info": {"base_url": "", "shards_base_url": "", "subdir": "noarch"},
        },
        "https://conda.anaconda.org/",
    )
    cls, url, *rest = repr(shardlike).split()
    assert "ShardLike" in cls
    assert shardlike.url == url
