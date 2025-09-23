# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Sharded repodata subsets.

Traverse dependencies of installed and to-be-installed packages to generate a
useful subset for the solver.

In this algorithm we treat a package name as a node, and all its dependencies
across all channels as edges. We then traverse all edges to discover all
reachable package names. The solver should be able to find a solution with only
this subset.

We could treat a (name, channel) tuple as a node, but it's more to keep track
of.

This subset is overgenerous since the user is unlikely to want to install very
old packages and their dependencies. If this is too slow, we could deploy
heuristics that automatically ignore older package versions. We could also allow
the user to configure minimum versions of common packages and ignore older
versions and their dependencies, falling back to a full solve if unsatisfiable.

We treat both sharded and monolithic repodata as if they were made up of
per-package shards, computing a subset of both. This is because it is possible
for the monolithic repodata to mention packages that exist in the true sharded
repodata but would not be found by only traversing the shards.

After we have the subset we write it out as monolithic repodata files, where it
can be used by a shards-unaware solver.
"""

from __future__ import annotations

import heapq
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import conda.gateways.repodata
from conda.base.context import context
from conda.core.subdir_data import SubdirData
from conda.models.channel import Channel

from conda_libmamba_solver import shards_cache

from .shards import RepodataDict, ShardLike, fetch_shards, shard_mentioned_packages


@dataclass(order=True)
class Node:
    distance: int = sys.maxsize
    package: str = ""
    visited: bool = False


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
                # check that we don't fetch the same shard twice...
                shard = shardlike.fetch_shard(node.package)
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
            # parallel fetch all unvisited shards but don't mark as visited
            original_priority, node = heapq.heappop(unvisited)
            if (
                original_priority != node.distance
            ):  # pragma: no cover; didn't match what's in the heap
                continue
            if node.visited:  # pragma: no cover
                continue
            node.visited = True

            for next, cost in self.outgoing(node):
                if not next.visited:
                    next.distance = min(node.distance + cost, next.distance)
                    heapq.heappush(unvisited, (next.distance, next))


def build_repodata_subset(tmp_path, root_packages, channels):
    channel_data = fetch_channels(channels)

    subset = RepodataSubset((*channel_data.values(),))
    subset.shortest(root_packages)
    print(len(subset.nodes), "package names discovered")

    subset_paths = {}

    repodata_size = 0
    for channel, shardlike in channel_data.items():
        repodata = shardlike.build_repodata()
        # XXX not guaranteed unique
        _, *channel_shortname = channel.rsplit("/", 2)
        repodata_path = tmp_path / ("_".join(channel_shortname))
        # most compact json
        repodata_text = json.dumps(repodata, indent=0, separators=(",", ":"), sort_keys=True)
        repodata_size += len(repodata_text)
        repodata_path.write_text(repodata_text)

        subset_paths[channel] = repodata_path

    return subset_paths, repodata_size


def fetch_channels(channels):
    channel_data: dict[str, ShardLike] = {}

    # share single disk cache for all Shards() instances
    cache = shards_cache.ShardCache(Path(conda.gateways.repodata.create_cache_dir()))

    for channel in channels:
        for channel_url in Channel(channel).urls(True, context.subdirs):
            subdir_data = SubdirData(Channel(channel_url))
            found = fetch_shards(subdir_data, cache)
            if not found:
                repodata_json, _ = subdir_data.repo_fetch.fetch_latest_parsed()
                repodata_json = RepodataDict(repodata_json)  # type: ignore
                found = ShardLike(repodata_json, channel_url)
            channel_data[channel_url] = found
    return channel_data
