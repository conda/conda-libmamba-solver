# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Test sharded repodata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
import urllib.parse
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import msgpack
import pytest
import zstandard
from conda.base.context import context, reset_context
from conda.core.subdir_data import SubdirData
from conda.models.channel import Channel

from conda_libmamba_solver import shards, shards_cache
from conda_libmamba_solver.index import LibMambaIndexHelper
from conda_libmamba_solver.shards import (
    ShardLike,
    Shards,
    ShardsIndex,
    batch_retrieve_from_cache,
    fetch_shards,
    shard_mentioned_packages,
)
from conda_libmamba_solver.shards_subset import Node, build_repodata_subset, fetch_channels
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
    bad_schema = zstandard.compress(msgpack.dumps(malformed))  # type: ignore
    # XXX not-zstandard; not msgpack
    malformed_digest = hashlib.sha256(bad_schema).digest()
    noarch = shards_repository / "noarch"
    (noarch / f"{malformed_digest.hex()}.msgpack.zst").write_bytes(bad_schema)
    not_zstd = b"not zstd"
    (noarch / f"{sha256(not_zstd).digest().hex()}.msgpack.zst").write_bytes(not_zstd)
    not_msgpack = zstandard.compress(b"not msgpack")
    (noarch / f"{sha256(not_msgpack).digest().hex()}.msgpack.zst").write_bytes(not_msgpack)
    fake_shards: ShardsIndex = {
        "info": {"subdir": "noarch", "base_url": "", "shards_base_url": ""},
        "repodata_version": 1,
        "shards": {
            "fake_package": b"",
            "malformed": hashlib.sha256(bad_schema).digest(),
            "not_zstd": hashlib.sha256(not_zstd).digest(),
            "not_msgpack": hashlib.sha256(not_msgpack).digest(),
        },
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
        # XXX this is currently trying to decompress the server's 404 response, which should be a `requests.Response.raise_for_status()`
        found.fetch_shard("fake_package")

    # currently logs KeyError: 'packages', doesn't cache, returns decoded msgpack
    malo = found.fetch_shard("malformed")
    assert malo == {"follows_schema": False}  # XXX should we return None or raise

    with pytest.raises(zstandard.ZstdError):
        found.fetch_shard("not_zstd")

    # unclear if all possible "bad msgpack" errors inherit from a common class
    # besides ValueError
    with pytest.raises(ValueError):
        found.fetch_shard("not_msgpack")


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

    channel_data = fetch_channels(channels)

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
    """
    Code coverage for ShardLike.__repr__()
    """
    shardlike = ShardLike(
        {
            "packages": {},
            "packages.conda": {},
            "info": {"base_url": "", "shards_base_url": "", "subdir": "noarch"},
        },
        "https://conda.anaconda.org/",
    )
    cls, url, *_ = repr(shardlike).split()
    assert "ShardLike" in cls
    assert shardlike.url == url


ROOT_PACKAGES = [
    "__archspec",
    "__conda",
    "__osx",
    "__unix",
    "bzip2",
    "ca-certificates",
    "expat",
    "icu",
    "libexpat",
    "libffi",
    "liblzma",
    "libmpdec",
    "libsqlite",
    "libzlib",
    "ncurses",
    "openssl",
    "pip",
    "python",
    "python_abi",
    "readline",
    "tk",
    "twine",
    "tzdata",
    "xz",
    "zlib",
]


def test_traverse_shards_3(conda_no_token: None, tmp_path):
    """
    Build repodata subset using the third attempt at a dependency traversal
    algorithm.
    """

    logging.basicConfig(level=logging.INFO)
    shards.log.setLevel(logging.DEBUG)
    shards_cache.log.setLevel(logging.DEBUG)

    # installed, plus what we want to add (twine)
    root_packages = ROOT_PACKAGES[:]

    channels = list(context.default_channels)
    channels.append(Channel("conda-forge-sharded"))

    subset, repodata_size = build_repodata_subset(tmp_path, root_packages, channels)

    print(f"Repodata subset is {repodata_size} bytes")

    # e.g. this for noarch and osx-arm64
    # % curl https://conda.anaconda.org/conda-forge-sharded/noarch/repodata.json.zst | zstd -d | wc
    full_repodata_benchmark = 138186556 + 142680224

    print(
        f"Versus only noarch and osx-arm64 full repodata: {repodata_size / full_repodata_benchmark:.02f} times as large"
    )

    print("Channels:", ",".join(urllib.parse.urlparse(url).path[1:] for url in subset))


def test_shards_indexhelper(conda_no_token):
    """
    Load LibMambaIndexHelper with parameters that will enable sharded repodata.
    """
    channels = [*context.default_channels, Channel("conda-forge-sharded")]

    class fake_in_state:
        installed = {name: object() for name in ROOT_PACKAGES}
        requested = ("twine",)

    # Would eagerly download repodata.json.zst for all channels
    helper = LibMambaIndexHelper(
        channels,
        (),  # subdirs
        "repodata.json",
        (),
        (),  # pkgs_dirs to load packages locally when offline
        in_state=fake_in_state,  # type: ignore
    )

    print(helper.repos)


@contextmanager
def _timer(name: str):
    begin = time.monotonic_ns()
    yield
    end = time.monotonic_ns()
    print(f"{name} took {(end - begin) / 1e9:0.6f}s")


def test_parallel_fetcherator(conda_no_token: None):
    channels = [*context.default_channels, Channel("conda-forge-sharded")]
    roots = [
        Node(distance=0, package="ca-certificates", visited=False),
        Node(distance=0, package="icu", visited=False),
        Node(distance=0, package="expat", visited=False),
        Node(distance=0, package="libexpat", visited=False),
        Node(distance=0, package="libffi", visited=False),
        Node(distance=0, package="libmpdec", visited=False),
        Node(distance=0, package="libzlib", visited=False),
        Node(distance=0, package="openssl", visited=False),
        Node(distance=0, package="python", visited=False),
        Node(distance=0, package="readline", visited=False),
        Node(distance=0, package="liblzma", visited=False),
        Node(distance=0, package="xz", visited=False),
        Node(distance=0, package="libsqlite", visited=False),
        Node(distance=0, package="tk", visited=False),
        Node(distance=0, package="ncurses", visited=False),
        Node(distance=0, package="zlib", visited=False),
        Node(distance=0, package="pip", visited=False),
        Node(distance=0, package="twine", visited=False),
        Node(distance=0, package="python_abi", visited=False),
        Node(distance=0, package="tzdata", visited=False),
    ]

    with _timer("repodata.json/shards index fetch"):
        channel_data = fetch_channels(channels)

    with _timer("Shard fetch"):
        sharded = [channel for channel in channel_data.values() if isinstance(channel, Shards)]
        assert sharded, "No sharded repodata found"
        remaining = batch_retrieve_from_cache(sharded, [node.package for node in roots])
        print(f"{len(remaining)} shards to fetch from network")

    # XXX don't call everything Shard/Shards
