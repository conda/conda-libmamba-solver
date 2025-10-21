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
import queue
import sys
from contextlib import suppress
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
    from queue import SimpleQueue as Queue

    from conda_libmamba_solver.shards_cache import ShardCache

    from .shards import (
        ShardLike,
    )
    from .shards_typing import ShardDict


@dataclass(order=True)
class Node:
    distance: int = sys.maxsize
    package: str = ""
    channel: str = ""
    visited: bool = False

    def to_id(self) -> NodeId:
        return NodeId(self.package, self.channel)

    def in_shard(self, shardlike: ShardLike) -> bool:
        return self.channel == shardlike.url


@dataclass(order=True, eq=True, frozen=True)
class NodeId:
    package: str
    channel: str

    def __hash__(self):
        return hash((self.package, self.channel))

    def in_shard(self, shardlike: ShardLike):
        return self.channel == shardlike.url


def cache_fetch_thread(
    in_queue: Queue[list[str] | None],
    shard_out_queue: Queue[list[tuple[str, ShardDict]] | None],
    network_out_queue: Queue[list[str] | None],
    cache: ShardCache,
):
    """
    Fetch batches of shards from cache until in_queue sees None. Enqueue found
    shards to shard_out_queue, and not found shards to network_out_queue.

    When we see None on in_queue, we send None to both out queues and exit.
    """
    cache = cache.copy()

    # should we send None to out_queues when done? Or just return?
    running = True
    while running and (batch := in_queue.get()) is not None:
        shard_urls = batch[:]
        with suppress(queue.Empty):
            while running:
                batch = in_queue.get_nowait()
                if batch is None:
                    # do the work but then quit
                    running = False
                    break
                shard_urls.extend(batch)

        cached = cache.retrieve_multiple(shard_urls)

        found = []
        not_found = []
        for url, shard in cached.items():
            if shard:
                found.append((url, shard))
            else:
                not_found.append(url)

        # Might wake up the network thread by calling it first:
        network_out_queue.put(not_found)
        shard_out_queue.put(found)

    network_out_queue.put(None)
    shard_out_queue.put(None)


def network_fetch_thread(
    in_queue: Queue[NodeId],
    shard_out_queue: Queue[tuple[ShardLike, ShardDict]],
):
    """
    in_queue contains (channel, package) tuples to fetch over the network.
    While the in_queue has not received a sentinel None, empty everything from
    the queue. Fetch all of them over the network. Fetched shards go to
    shard_out_queue.
    """
    while (item := in_queue.get()) is not None:
        items = [item]
        with suppress(queue.Empty):
            while True:
                items.append(in_queue.get_nowait())

        ...
        # generate url's for all items
        # call Shards.batch_retrieve_from_network(not_in_cache)
        # all fetched shards go to shard_out_queue (do we need a Node that includes the ShardLike() (i.e. channel)?


@dataclass
class RepodataSubset:
    nodes: dict[NodeId, Node]
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
            if node.package not in shardlike:
                continue

            # check that we don't fetch the same shard twice...
            shard = shardlike.fetch_shard(node.package)

            for package in shard_mentioned_packages_2(shard):
                node_id = NodeId(package, shardlike.url)

                if node_id not in self.nodes:
                    self.nodes[node_id] = Node(node.distance + 1, package, shardlike.url)
                    yield self.nodes[node_id]

                    if package not in discovered:
                        # now this is per package name, not per (name, channel) tuple
                        log.debug("%s -> %s;", json.dumps(node.package), json.dumps(package))
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

    def shortest(self, start_packages):
        # nodes.visited and nodes.distance should be reset before calling

        def initial_nodes():
            for package in start_packages:
                for shardlike in self.shardlikes:
                    if package in shardlike:
                        node = Node(0, package, shardlike.url)
                        node_id = node.to_id()
                        yield (node_id, node)

        self.nodes = dict(initial_nodes())
        unvisited = [(n.distance, n) for n in self.nodes.values()]
        sharded = [s for s in self.shardlikes if isinstance(s, Shards)]
        to_retrieve = set(self.nodes)
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
