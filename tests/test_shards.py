# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Test sharded repodata.
"""

from __future__ import annotations

import heapq
import json
import logging
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import msgpack
import pytest
import zstandard
from conda.base.context import context, reset_context
from conda.core.subdir_data import SubdirData
from conda.models.channel import Channel

from conda_libmamba_solver import shard_cache, shards
from conda_libmamba_solver.shards import (
    RepodataDict,
    ShardLike,
    Shards,
    fetch_shards,
    shard_mentioned_packages,
)

HERE = Path(__file__).parent


def package_names(shard: shard_cache.Shard):
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


def test_shards(conda_no_token: None):
    """
    Test basic shard fetch etc.
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
    cache = shard_cache.ShardCache(tmp_path)

    fake_shard = {"foo": "bar"}
    annotated_shard = shard_cache.AnnotatedRawShard(
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

    assert (tmp_path / shard_cache.SHARD_CACHE_NAME).exists()


def test_shard_cache_multiple(tmp_path: Path):
    """
    Test that retrieve_multiple() is equivalent to several retrieve() calls.
    """
    NUM_FAKE_SHARDS = 64

    cache = shard_cache.ShardCache(tmp_path)
    fake_shards = []

    compressor = zstandard.ZstdCompressor(level=1)
    for i in range(NUM_FAKE_SHARDS):
        fake_shard = {f"foo{i}": "bar"}
        annotated_shard = shard_cache.AnnotatedRawShard(
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

    assert (tmp_path / shard_cache.SHARD_CACHE_NAME).exists()


def test_shard_cache_2():
    """
    Check that existing shards cache has been populated with matching shards,
    package names.
    """
    subdir_data = SubdirData(Channel("main/noarch"))
    # accepts directory that cache goes into, not database filename
    cache = shard_cache.ShardCache(Path(subdir_data.cache_path_base).parent)
    extant = {}
    for row in cache.conn.execute("SELECT url, package, shard, timestamp FROM shards"):
        extant[row["url"]] = shard_cache.AnnotatedRawShard(
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


@dataclass(order=True)
class Node:
    distance: int = sys.maxsize
    package: str = ""
    visited: bool = False

    def __hash__(self):
        return hash(self.package)

    def reset(self):
        self.visited = False
        self.distance = sys.maxsize


@dataclass
class RepodataSubset:
    nodes: dict[str, Node]
    shardlikes: list[ShardLike]

    def __init__(self, shardlikes):
        self.nodes = {}
        self.shardlikes = shardlikes

    def neighbors(self, node: Node):
        """
        All neighbors for node.
        """
        discovered = set()
        for shardlike in self.shardlikes:
            if node.package in shardlike:
                try:
                    # check that we don't fetch the same shard twice...
                    shard = shardlike.fetch_shard(node.package)
                except:
                    raise
                for package in shard_mentioned_packages(shard):
                    if package not in self.nodes:
                        self.nodes[package] = Node(node.distance + 1, package)
                        # by moving yield up here we try to only visit dependencies
                        # that no other node already knows about. Doesn't make it faster.
                        if package not in discovered:  # redundant with not in self.nodes?
                            print(f"{json.dumps(node.package)} -> {json.dumps(package)};")
                            yield self.nodes[package]
                    if package not in discovered:
                        pass
                        # dot format valid ids: https://graphviz.org/doc/info/lang.html#ids (or quote string)

                        # we might not require "in self.nodes" neighbors since
                        # we don't need to find the shortest path

                        # yield self.nodes[package]

                    discovered.add(package)  # also doesn't make it faster

    def outgoing(self, node: Node):
        """
        All nodes that can be reached by this node, plus cost.
        """
        for n in self.neighbors(node):
            yield n, 1

    def shortest(self, start_packages):
        # nodes.visited and nodes.distance should be reset before calling
        self.nodes = {package: Node(0, package) for package in start_packages}
        unvisited = [(n.distance, n) for n in self.nodes.values()]
        while unvisited:
            original_priority, node = heapq.heappop(unvisited)
            if original_priority != node.distance:  # ???
                continue
            if node.visited:
                continue
            node.visited = True

            for next, cost in self.outgoing(node):
                if not next.visited:
                    next.distance = min(node.distance + cost, next.distance)
                    heapq.heappush(unvisited, (next.distance, next))

    def reset(self):
        self.nodes = {}


def test_traverse_shards_3(conda_no_token: None, tmp_path):
    """
    Another go at the dependency traversal algorithm.
    """

    logging.basicConfig(level=logging.INFO)
    shards.log.setLevel(logging.DEBUG)
    shard_cache.log.setLevel(logging.DEBUG)

    # installed, plus what we want to add (twine)
    root_packages = [
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

    channels = list(context.default_channels)
    channels.append(Channel("conda-forge-sharded"))

    channel_data: dict[str, ShardLike] = {}
    for channel in channels:
        for channel_url in Channel(channel).urls(True, context.subdirs):
            subdir_data = SubdirData(Channel(channel_url))
            found = fetch_shards(subdir_data)
            if not found:
                repodata_json, _ = subdir_data.repo_fetch.fetch_latest_parsed()
                found = ShardLike(repodata_json, channel_url)  # type: ignore
            channel_data[channel_url] = found

    channels = list(context.default_channels)
    print(channels)

    # state to initiate a solve
    in_state = pickle.loads((HERE / "data" / "in_state.pickle").read_bytes())
    print(in_state)

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

    print(channel_data)

    subset = RepodataSubset((*channel_data.values(),))
    subset.shortest(root_packages)
    print(len(subset.nodes), "package names discovered")

    repodata_size = 0
    for shardlike in subset.shardlikes:
        _, *channel = shardlike.url.replace("/repodata_shards.msgpack.zst", "").rsplit("/", 2)
        repodata = shardlike.build_repodata()
        repodata_path = tmp_path / ("_".join(channel))
        # most compact json
        repodata_text = json.dumps(repodata, indent=0, separators=(",", ":"))
        repodata_size += len(repodata_text)
        repodata_path.write_text(repodata_text)

    print(f"Repodata subset is {repodata_size} bytes")

    # e.g. this for noarch and osx-arm64
    # % curl https://conda.anaconda.org/conda-forge-sharded/noarch/repodata.json.zst | zstd -d | wc
    full_repodata_benchmark = 138186556 + 142680224

    print(
        f"Versus only noarch and osx-arm64 full repodata: {repodata_size / full_repodata_benchmark:.02f} times as large"
    )

    channel_names = []
    for shardlike in subset.shardlikes:
        _, *channel = shardlike.url.replace("/repodata_shards.msgpack.zst", "").rsplit("/", 2)
        channel_names.append("/".join(channel))

    print(f"Repodata subset includes {', '.join(channel_names)}")
