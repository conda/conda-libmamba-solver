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
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import msgpack
import pytest
import zstandard
from conda.base.context import context, reset_context
from conda.core.subdir_data import SubdirData
from conda.gateways.connection.session import get_session
from conda.models.channel import Channel

from conda_libmamba_solver import shard_cache, shards
from conda_libmamba_solver.shards import (
    RepodataDict,
    ShardLike,
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

    # Would eagerly download repodata.json.zst for all channels
    # helper = LibMambaIndexHelper(
    #     channels,
    #     (),  # subdirs
    #     "repodata.json",
    #     installed,
    #     (),  # pkgs_dirs to load packages locally when offline
    #     in_state=in_state,
    # )

    # accessing "helper.repos" downloads repodata.json in the traditional way
    # for all channels:
    # print(helper.repos)

    subdir_data = SubdirData(channels[0])
    found = fetch_shards(subdir_data)
    assert found, f"Shards not found for {channels[0]}"

    for package in found.packages_index:
        shard_url = found.shard_url(package)
        assert shard_url.startswith("http")  # or channel url or shards_base_url

    session = get_session(
        subdir_data.url_w_subdir
    )  # XXX session could be different based on shards_base_url and different than the packages base_url

    # download or fetch-from-cache a random set of shards

    for package in random.choices([*found.packages_index.keys()], k=16):
        shard = found.fetch_shard(package, session)

        mentioned_in_shard = shard_mentioned_packages(shard)
        print(package_names(shard), mentioned_in_shard)


def test_shards_2(conda_no_token: None):
    """
    Test all channels fetch.
    """
    channels = list(context.default_channels)
    print(channels)
    # is (pkgs/main, pkgs/r) in mine

    # Channel('pkgs/main').url()
    # 'https://repo.anaconda.com/pkgs/main/osx-arm64'
    # context.default_channels[0].url()
    # 'https://repo.anaconda.com/pkgs/main/osx-arm64'
    # Channel('main').url()

    # state to initiate a solve
    in_state = pickle.loads((HERE / "data" / "in_state.pickle").read_bytes())
    print(in_state)

    channels.append(Channel("conda-forge-sharded"))

    installed = (
        *in_state.installed.values(),
        # *in_state.virtual.values(),  # skip these which will not exist in channels
    )

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

    shards_to_get = set(package.name for package in installed)

    # what we want to install
    # shards_to_get.update(in_state.requested)
    shards_to_get.add("twine")

    shards_to_get_more = set(
        package.name for root in installed for package in root.combined_depends
    )

    print(
        "Are the installed packages the same as all their dependencies?",
        shards_to_get == shards_to_get_more,
        f"{len(shards_to_get)} from installed",
        f"{len(shards_to_get_more)} from installed's dependencies",  # includes virtual packages, doesn't matter
    )

    # e.g. shards_to_get_more - shards_to_get
    # {'expat', 'xz', 'zlib'}
    # shards_to_get - shards_to_get_more
    # {'__conda', '__archspec'}

    shards_to_get |= shards_to_get_more

    shards_have: dict[str, dict[str, shard_cache.Shard | None]] = {
        url: {} for url in channel_data
    }  # mapping between URL and shards fetched or attempted-to-fetch
    iteration = 0
    waste = 0

    time_start = time.monotonic()
    while shards_to_get:
        iteration_start = time.monotonic()
        print(f"Seek {len(shards_to_get)} shards in iteration {iteration}:")
        print("\n".join(textwrap.wrap(" ".join(sorted(shards_to_get)))))
        new_shards_to_get = set()
        for channel_url, shardlike in channel_data.items():
            session = get_session(channel_url)  # XXX inefficient but it has a @cache decorator
            new_shards_to_get = set()
            # packages available in the channel that we don't already have
            shards_to_fetch = set(
                package
                for package in shards_to_get
                if package in shardlike
                and package
                not in shardlike.visited  # may make sense to update visited with negative matches
            )
            shards_unavailable = set(
                package
                for package in shards_to_get
                if package not in shardlike
                and package
                not in shardlike.visited  # may make sense to update visited with negative matches
            )
            for package in shards_unavailable:
                shardlike.visited[package] = None

            fetched_shards = shardlike.fetch_shards(shards_to_fetch, session)
            for package, shard in fetched_shards.items():
                new_shards_to_get.update(shard_mentioned_packages(shard))

            # also a property of the shardlike instance:
            shards_have[channel_url].update(fetched_shards)

            if False:  # earlier "one at a time" method
                shards_fetched_serially = set()
                for package in shards_to_get:
                    if package not in shards_have[channel_url]:  # XXX also inefficient
                        if package in shardlike:
                            new_shard = shardlike.fetch_shard(package, session)
                            new_shards_to_get.update(shard_mentioned_packages(new_shard))
                            shards_have[channel_url][package] = package
                            shards_fetched_serially.add(package)
                        else:
                            shards_have[channel_url][package] = None
                    else:
                        waste += 1
            # print("To-fetch same both ways?", shards_fetched_serially == shards_to_fetch)

        shards_to_get = new_shards_to_get
        shards_to_get -= set(next(iter(shards_have.values())))
        iteration_end = time.monotonic()
        print(f"Iteration {iteration} took {iteration_end - iteration_start:.2f}s")
        iteration += 1
    time_end = time.monotonic()

    relevant_packages = [set(value) for value in shards_have.values()]
    assert all(len(x) == len(relevant_packages[0]) for x in relevant_packages)
    print(f"Sought data for the following {len(relevant_packages[0])} packages:")
    print("\n".join(textwrap.wrap(" ".join(sorted(relevant_packages[0])))))
    print(f"Wasted {waste} inner loop iterations")
    print(f"Took {time_end - time_start:.2f}s")

    # Now write out shards_have packages that are not None, as small
    # repodata.json for the solver.


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
        for m in range(n):
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
                    shard = shardlike.fetch_shard(node.package)
                except:
                    raise
                for package in shard_mentioned_packages(shard):
                    if package not in self.nodes:
                        self.nodes[package] = Node(node.distance + 1, package)
                        print(f"{node.package} -> {package};")
                    if package not in discovered:
                        yield self.nodes[package]

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


def test_traverse_shards_3(conda_no_token: None):
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
