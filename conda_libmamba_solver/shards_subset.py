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
import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .shards import (
    Shards,
    batch_retrieve_from_cache,
    batch_retrieve_from_network,
    fetch_channels,
    shard_mentioned_packages_2,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .shards import (
        ShardLike,
    )


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
                for package in shard_mentioned_packages_2(shard):
                    # shard_mentioned_packages_2 doesn't include the shard's own
                    # package name. Do we need to broadcast node.package across
                    # channels, or will the incoming node have already taken
                    # care of it for us?
                    if package not in self.nodes:
                        self.nodes[package] = Node(node.distance + 1, package)
                        # by moving yield up here we try to only visit dependencies
                        # that no other node already knows about. Doesn't make it faster.
                        if package not in discovered:  # redundant with not in self.nodes?
                            log.debug(f"{json.dumps(node.package)} -> {json.dumps(package)};")
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
        # If we set a greater cost for sharded repodata than the repodata that
        # is already in memory and tracked nodes as (channel, package) tuples,
        # we might be able to find more shards-to-fetch-in-parallel more
        # quickly. On the other hand our goal is that the big channels will all
        # be sharded.
        for n in self.neighbors(node):
            yield n, 1

    def shortest(self, start_packages):
        # nodes.visited and nodes.distance should be reset before calling
        self.nodes = {package: Node(0, package) for package in start_packages}
        unvisited = [(n.distance, n) for n in self.nodes.values()]
        sharded = [s for s in self.shardlikes if isinstance(s, Shards)]
        to_retrieve: set[str] = set(self.nodes)
        retrieved: set[str] = set()
        while unvisited:
            # parallel fetch all unvisited shards but don't mark as visited
            if to_retrieve:
                # can we get stuck looking for unavailable packages repeatedly?
                not_in_cache = batch_retrieve_from_cache(sharded, sorted(to_retrieve))
                batch_retrieve_from_network(not_in_cache)
            retrieved.update(to_retrieve)  # not necessary
            to_retrieve.clear()

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
                    to_retrieve.add(next.package)
                    heapq.heappush(unvisited, (next.distance, next))


def build_repodata_subset(root_packages, channels):
    channel_data = fetch_channels(channels)

    subset = RepodataSubset((*channel_data.values(),))
    subset.shortest(root_packages)
    log.debug("%d package names discovered", len(subset.nodes))

    return channel_data
