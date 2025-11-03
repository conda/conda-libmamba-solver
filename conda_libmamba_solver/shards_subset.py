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

import concurrent.futures
import heapq
import logging
import queue
import sys
import threading
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from queue import SimpleQueue
from typing import TYPE_CHECKING

import requests.exceptions

from .shards import (
    Shards,
    _shards_connections,
    batch_retrieve_from_cache,
    batch_retrieve_from_network,
    fetch_channels,
    shard_mentioned_packages_2,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from queue import SimpleQueue as Queue
    from typing import Literal

    from conda_libmamba_solver.shards_cache import ShardCache
    from conda_libmamba_solver.shards_typing import ShardDict

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


@dataclass(order=True, eq=True, frozen=True)
class NodeId:
    package: str
    channel: str

    def __hash__(self):
        return hash((self.package, self.channel))


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
                        node_queue.append(next_node)

    def shortest_pipelined(self, root_packages):
        """
        Build repodata subset using a main thread, a thread to fetch from
        sqlite3 and another threadpool to fetch http.
        """

        self.nodes = dict(_nodes_from_packages(root_packages, self.shardlikes))

        # if sharded is empty, we could skip everything related to threads.
        sharded = [s for s in self.shardlikes if isinstance(s, Shards)]
        cache = sharded[0].shards_cache if sharded else None
        session = sharded[0].session if sharded else None

        in_queue: SimpleQueue[list[str] | None] = SimpleQueue()
        shard_out_queue: SimpleQueue[list[tuple[str, ShardDict]]] = SimpleQueue()
        cache_miss_queue: SimpleQueue[list[str]] = SimpleQueue()

        # this kind of thread can crash, and we don't hear back without our own
        # handling.
        cache_thread = threading.Thread(
            target=cache_fetch_thread,
            args=(in_queue, shard_out_queue, cache_miss_queue, cache),
            daemon=False,
        )

        # this kind of thread can crash, and we don't hear back without our own
        # handling.
        network_thread = threading.Thread(
            target=network_fetch_thread,
            args=(cache_miss_queue, shard_out_queue, session),
            daemon=False,
        )

        # or after populating initial URLs queue
        cache_thread.start()
        network_thread.start()

        # see test_*_thread in test_shards.py for how to set up the workers

        pending = set()

        while pending:
            # in_queue.put(list of URLs to fetch)
            new_shards = shard_out_queue.get()  # with a timeout to detect dead threads?
            for shard in new_shards:
                pass  # XXX add algorithm here
            # for shard in new_shards:
            #    decompress and cache if needed (or do that in the fetch threads)
            #    call shard_mentioned_packages_2(shard)
            #    for channel in shardlikes:
            #        if mentioned package in channel
            #             add node to pending if not visited
            #             update distances
            #             if new nodes added, add their shard URLs to in_queue
            #
            #             associate each URL with a list of node ids that are
            #             waiting for it, because we imagine a future
            #             optimization of multiple packages in a single shard.
            #
            #             if new node is a ShardLike, process right away or push that shard to
            #             shard_out_queue right away.

        in_queue.put(None)

        cache_thread.join()
        network_thread.join()


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


# region workers


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
    in_queue: Queue[list[str] | None],
    shard_out_queue: Queue[list[tuple[str, bytes]] | None],
    # how to communicate errors?
    session,
):
    """
    in_queue contains urls to fetch over the network.
    While the in_queue has not received a sentinel None, empty everything from
    the queue. Fetch all of them over the network. Fetched shards go to
    shard_out_queue.
    """
    running = True

    def fetch(s, url):
        response = s.get(url)
        response.raise_for_status()
        # This needs to be decompressed and parsed, and go into the cache as raw
        # content if it was really .msgpack.zst. shard_out_queue for
        # cache_fetch_thread produces ShardDict but this produces bytes; would
        # be better if they produced the same type.
        shard_out_queue.put([(url, response.content)])
        data = response.content
        return (url, data)  # needs to be

    with concurrent.futures.ThreadPoolExecutor(max_workers=_shards_connections()) as executor:
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
                    futures = [executor.submit(fetch, session, url) for url in shard_urls]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            url, data = future.result()
                            log.debug("Fetch thread got", url, len(data))
                        except requests.exceptions.RequestException as e:
                            log.error("Error fetching shard. %s", e)

                        # This will drain the http threadpool before loading another
                        # batch. Instead, we might want to get_nowait() and
                        # executor.submit() here after each future has been
                        # retired to make sure we always have
                        # _shards_connections() in flight. Or we could use
                        # future "on completion" callbacks instead of relying so
                        # much on as_completed().

    shard_out_queue.put(None)


# endregion
