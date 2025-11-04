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

import msgpack
import requests.exceptions
import zstandard

from conda_libmamba_solver.shards_cache import AnnotatedRawShard

from .shards import (
    ZSTD_MAX_SHARD_SIZE,
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
    shard_url: str = ""

    def to_id(self) -> NodeId:
        return NodeId(self.package, self.channel, self.shard_url)


@dataclass(order=True, eq=True, frozen=True)
class NodeId:
    package: str
    channel: str
    shard_url: str = ""

    def __hash__(self):
        return hash((self.package, self.channel, self.shard_url))


def _nodes_from_packages(
    root_packages: list[str], shardlikes: Iterable[ShardBase]
) -> Iterator[tuple[NodeId, Node]]:
    """
    Yield (NodeId, Node) for all root packages found in shardlikes.
    """
    for package in root_packages:
        for shardlike in shardlikes:
            if package in shardlike:
                node = Node(0, package, shardlike.url, shard_url=shardlike.shard_url(package))
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

        in_queue: SimpleQueue[list[NodeId] | None] = SimpleQueue()
        shard_out_queue: SimpleQueue[list[tuple[NodeId, ShardDict]]] = SimpleQueue()
        cache_miss_queue: SimpleQueue[list[NodeId] | None] = SimpleQueue()

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
            args=(cache_miss_queue, shard_out_queue, cache, self.shardlikes),
            daemon=False,
        )

        # or after populating initial URLs queue
        cache_thread.start()
        network_thread.start()

        # see test_*_thread in test_shards.py for how to set up the workers

        shardlikes_by_url = {s.url: s for s in self.shardlikes}
        pending = set(self.nodes)  # to be sent to in_queue
        submitted = set()  # sent to in_queue, not received from shard_out_queue

        def enqueue_pending():
            log.info("Enqueue %s pending", sorted(node_id.package for node_id in pending))
            shard_out_batch = []
            shard_in_batch = []
            # enqueue all pending nodes
            for node_id in pending:
                # we could wind up sending duplicate node_id here?
                shardlike = shardlikes_by_url[node_id.channel]
                if shardlike.shard_in_memory(node_id.package):  # for monolithic repodata
                    shard_out_batch.append((node_id, shardlike.visit_shard(node_id.package)))
                else:
                    shard_in_batch.append(node_id)
                submitted.add(node_id)  # tracks everything not yet read from shard_out_batch
            if shard_out_batch:
                shard_out_queue.put(shard_out_batch)
            if shard_in_batch:
                in_queue.put(shard_in_batch)
            pending.clear()

        enqueue_pending()  # initial batch

        timeouts = 0

        running = True
        while running:
            try:
                new_shards = shard_out_queue.get(timeout=1)
            except queue.Empty:
                print(f"Pending {len(pending)}, Submitted {len(submitted)}")
                timeouts += 1
                if timeouts > 10:
                    log.error("Timeout waiting for shard_out_queue")
                    raise TimeoutError("Timeout waiting for shard_out_queue")
                continue
            if new_shards is None:
                running = False
                new_shards = []
            for node_id, shard in new_shards:
                # add shard to appropriate ShardLike
                submitted.remove(node_id)
                shardlike = shardlikes_by_url[node_id.channel]
                shardlike.visited[node_id.package] = shard
                mentioned_packages = list(shard_mentioned_packages_2(shard))

                for shardlike in self.shardlikes:
                    for package in mentioned_packages:
                        if package in shardlike:
                            node_id = NodeId(package, shardlike.url, shardlike.shard_url(package))
                            if node_id not in self.nodes:
                                node = Node(
                                    distance=0,
                                    package=package,
                                    channel=shardlike.url,
                                    shard_url=shardlike.shard_url(package),
                                )
                                self.nodes[node_id] = node
                                pending.add(node_id)

            enqueue_pending()
            if not submitted:
                break

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
    in_queue: Queue[list[NodeId] | None],
    shard_out_queue: Queue[list[tuple[NodeId, ShardDict]] | None],
    network_out_queue: Queue[list[NodeId] | None],
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
        node_ids = batch[:]
        with suppress(queue.Empty):
            while running:
                batch = in_queue.get_nowait()
                if batch is None:
                    # do the work but then quit
                    running = False
                    break
                node_ids.extend(batch)

        urls = [node_id.shard_url for node_id in node_ids]
        if (
            "https://conda.anaconda.org/conda-forge-sharded/osx-arm64/3302666cf70e89c4ad8b8deb80e1db74a64876ea2bb125e3f380cdef47f3728e.msgpack.zst"
            in urls
        ):
            print("Fetch dbus from cache?")
        cached = cache.retrieve_multiple([node_id.shard_url for node_id in node_ids])

        url_to_nodeid = {node_id.shard_url: node_id for node_id in node_ids}
        found = []
        not_found = []
        for url, shard in cached.items():
            if shard:
                found.append((url_to_nodeid[url], shard))
            else:
                not_found.append(url_to_nodeid[url])

        # Might wake up the network thread by calling it first:
        network_out_queue.put(not_found)
        shard_out_queue.put(found)

    network_out_queue.put(None)
    shard_out_queue.put(None)


def network_fetch_thread(
    in_queue: Queue[list[NodeId] | None],
    shard_out_queue: Queue[list[tuple[NodeId, ShardDict]] | None],
    # how to communicate errors?
    cache: ShardCache,
    shardlikes: list[ShardBase],
):
    """
    in_queue contains urls to fetch over the network.
    While the in_queue has not received a sentinel None, empty everything from
    the queue. Fetch all of them over the network. Fetched shards go to
    shard_out_queue.
    """
    cache = cache.copy()
    dctx = zstandard.ZstdDecompressor(max_window_size=ZSTD_MAX_SHARD_SIZE)
    running = True

    shardlikes_by_url = {s.url: s for s in shardlikes}

    def fetch(s, url: str, node_id: NodeId):
        print("Get", url)
        response = s.get(url)
        response.raise_for_status()
        data = response.content
        return (url, node_id, data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=_shards_connections()) as executor:
        while running and (batch := in_queue.get()) is not None:
            node_ids = batch[:]
            with suppress(queue.Empty):
                while running:
                    batch = in_queue.get_nowait()
                    if batch is None:
                        # do the work but then quit
                        running = False
                        break
                    node_ids.extend(batch)

                    futures = []
                    for node_id in node_ids:
                        if (
                            node_id.shard_url
                            == "https://conda.anaconda.org/conda-forge-sharded/osx-arm64/3302666cf70e89c4ad8b8deb80e1db74a64876ea2bb125e3f380cdef47f3728e.msgpack.zst"
                            or node_id.package == "dbus"
                        ):
                            print("Fetch dbus from network?", node_id)
                        # this worker should only recieve network node_id's:
                        session = shardlikes_by_url[node_id.channel].session
                        url = shardlikes_by_url[node_id.channel].shard_url(node_id.package)
                        futures.append(executor.submit(fetch, session, url, node_id))

                    for future in concurrent.futures.as_completed(futures):
                        try:
                            url, node_id, data = future.result()
                            log.debug("Fetch thread got %s (%s bytes)", url, len(data))

                            # Decompress and parse. If it decodes as
                            # msgpack.zst, insert into cache. Then put "known
                            # good" shard into out queue.
                            shard: ShardDict = msgpack.loads(
                                dctx.decompress(data, max_output_size=ZSTD_MAX_SHARD_SIZE)
                            )  # type: ignore[assign]
                            # we don't track the unnecessary shard package name here:
                            cache.insert(AnnotatedRawShard(url, node_id.package, data))
                            shard_out_queue.put([(node_id, shard)])

                        except requests.exceptions.RequestException as e:
                            log.error("Error fetching shard. %s", e)
                        except Exception as e:
                            # raises ZstdError for b"not zstd" test data e.g.
                            log.exception("Unhandled exception in network thread", exc_info=e)

                        # This will drain the http threadpool before loading another
                        # batch. Instead, we might want to get_nowait() and
                        # executor.submit() here after each future has been
                        # retired to make sure we always have
                        # _shards_connections() in flight. Or we could use
                        # future "on completion" callbacks instead of relying so
                        # much on as_completed().

    shard_out_queue.put(None)


# endregion
