# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Sharded repodata subsets.

Traverse dependencies of installed and to-be-installed packages to generate a
useful subset for the solver.

The algorithm developed here is a direct result of the following CEP:

- https://conda.org/learn/ceps/cep-0016 (Sharded Repodata)

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

## Example usage

The following constructs several repodata (`noarch` and `linux-64`) from a single
channel name and a list of root packages:

```
from conda.models.channel import Channel
from conda_libmamba_solver.shards_subset import build_repodata_subset

channel = Channel("conda-forge-sharded/linux-64")
channel_data = build_repodata_subset(["python", "pandas"], [channel.url()])
repodata = {}

for url in channel_data:
    repodata[url] = channel_data.build_repodata()

# ... this is what's fed to the solver
```

"""

from __future__ import annotations

import heapq
import logging
import sys
from collections import deque
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
    from collections.abc import Iterable, Iterator
    from typing import Literal

    from .shards import (
        ShardBase,
    )


@dataclass(order=True)
class Node:
    distance: int = sys.maxsize
    package: str = ""
    channel: str = ""
    visited: bool = False

    def to_id(self) -> NodeId:
        return NodeId(self.package, self.channel)

    def in_shard(self, shardlike: ShardBase) -> bool:
        return self.channel == shardlike.url


@dataclass(order=True, eq=True, frozen=True)
class NodeId:
    package: str
    channel: str

    def __hash__(self):
        return hash((self.package, self.channel))

    def in_shard(self, shardlike: ShardBase) -> bool:
        return self.channel == shardlike.url


def _nodes_from_packages(
    root_packages: list[str], shardlikes: Iterable[ShardBase]
) -> Iterator[tuple[NodeId, Node]]:
    """
    Yield (NodeId, Node) for all root packages found in shardlikes.
    """
    for package in root_packages:
        for shardlike in shardlikes:
            if package in shardlike:
                node = Node(0, package, shardlike.url)
                node_id = node.to_id()
                yield node_id, node


@dataclass
class RepodataSubset:
    nodes: dict[NodeId, Node]
    shardlikes: Iterable[ShardBase]

    def __init__(self, shardlikes: Iterable[ShardBase]):
        self.nodes = {}
        self.shardlikes = shardlikes

    def neighbors(self, node: Node) -> Iterator[Node]:
        """
        Retrieve all unvisited neighbors of a node

        Neighbors in the context are dependencies of a package
        """
        discovered = set()

        for shardlike in self.shardlikes:
            if node.package not in shardlike:
                continue

            # check that we don't fetch the same shard twice...
            shard = shardlike.fetch_shard(
                node.package
            )  # XXX this is the only place that in-memory (repodata.json) shards are found for the first time

            for package in shard_mentioned_packages_2(shard):
                node_id = NodeId(package, shardlike.url)

                if node_id not in self.nodes:
                    self.nodes[node_id] = Node(node.distance + 1, package, shardlike.url)
                    yield self.nodes[node_id]

                    if package not in discovered:
                        # now this is per package name, not per (name, channel) tuple
                        discovered.add(package)

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

    def shortest_dijkstra(self, root_packages):
        """
        Fetch all root packages represented as `self.nodes`.

        This method updates associated `self.shardlikes` to contain enough data
        to build a repodata subset.
        """
        # nodes.visited and nodes.distance should be reset before calling

        self.nodes = dict(_nodes_from_packages(root_packages, self.shardlikes))

        unvisited = [(n.distance, n) for n in self.nodes.values()]
        sharded = [s for s in self.shardlikes if isinstance(s, Shards)]
        to_retrieve = set(self.nodes)  # XXX below, it expects set[str]
        retrieved: set[NodeId] = set()

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

            for next_node, cost in self.outgoing(node):
                if not next_node.visited:
                    next_node.distance = min(node.distance + cost, next_node.distance)
                    to_retrieve.add(next_node.package)
                    heapq.heappush(unvisited, (next_node.distance, next_node))

    def shortest_bfs(self, root_packages):
        """
        Fetch all root packages represented as `self.nodes` using the "breadth-first search"
        algorithm.

        This method updates associated `self.shardlikes` to contain enough data
        to build a repodata subset.
        """
        self.nodes = dict(_nodes_from_packages(root_packages, self.shardlikes))

        node_queue = deque(self.nodes.values())
        sharded = [s for s in self.shardlikes if isinstance(s, Shards)]

        while node_queue:
            # Batch fetch all nodes at current level
            to_retrieve = {node.package for node in node_queue if not node.visited}
            if to_retrieve:
                not_in_cache = batch_retrieve_from_cache(sharded, sorted(to_retrieve))
                batch_retrieve_from_network(not_in_cache)

            # Process one level
            level_size = len(node_queue)
            for _ in range(level_size):
                node = node_queue.popleft()
                if node.visited:
                    continue
                node.visited = True

                for next_node, _ in self.outgoing(node):
                    if not next_node.visited:
                        next_node.distance = node.distance + 1
                        node_queue.append(next_node)


def build_repodata_subset(
    root_packages: Iterable[str],
    channels: Iterable[str],
    algorithm: Literal["shortest_dijkstra", "shortest_bfs"] = "shortest_bfs",
) -> dict[str, ShardBase]:
    """
    Retrieve all necessary information to build a repodata subset.

    Params:
        root_packages: iterable of root package names
        channels: iterable of channel URLs
        algorithm: one of "shortest", "shortest_bfs

    TODO: Remove `algorithm` parameter once we've made a firm decision on which to use.
    """
    channel_data = fetch_channels(channels)

    subset = RepodataSubset((*channel_data.values(),))
    getattr(subset, algorithm)(root_packages)
    log.debug("%d (channel, package) nodes discovered", len(subset.nodes))

    return channel_data
