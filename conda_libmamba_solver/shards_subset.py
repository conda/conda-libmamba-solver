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

    from conda.models.channel import Channel

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

        # these are in self.nodes but we still need the shards
        self.nodes = dict(_nodes_from_packages(root_packages, self.shardlikes))

        # if sharded is empty, we could skip everything related to threads.
        sharded = [s for s in self.shardlikes if isinstance(s, Shards)]
        cache = sharded[0].shards_cache if sharded else None

        cache_in_queue: SimpleQueue[list[NodeId] | None] = SimpleQueue()
        shard_out_queue: SimpleQueue[list[tuple[NodeId, ShardDict]]] = SimpleQueue()
        cache_miss_queue: SimpleQueue[list[NodeId] | None] = SimpleQueue()

        # this kind of thread can crash, and we don't hear back without our own
        # handling.
        cache_thread = threading.Thread(
            target=cache_fetch_thread,
            args=(cache_in_queue, shard_out_queue, cache_miss_queue, cache),
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

        shardlikes_by_url = {s.url: s for s in self.shardlikes}
        pending = set(self.nodes)  # to be sent to in_queue
        submitted = set()  # sent to in_queue, not received from shard_out_queue
        processed = set()  # received from shard_out_queue

        def enqueue_pending():
            log.debug(
                "Enqueue %s pending %s",
                len(pending),
                sorted(node_id.package for node_id in pending),
            )
            shard_out_batch = []
            shard_in_batch = []
            # enqueue all pending nodes
            for node_id in pending:
                if node_id in submitted or node_id in processed:
                    # skip already submitted or processed
                    continue
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
                cache_in_queue.put(shard_in_batch)
            pending.clear()

        enqueue_pending()  # initial batch

        timeouts = 0

        running = True
        while running:
            try:
                new_shards = shard_out_queue.get(timeout=1)
            except queue.Empty:
                # print(f"Pending {len(pending)}, Submitted {len(submitted)}")
                if all([x in self.nodes for x in submitted]):
                    # print("Done but submitted not empty")
                    running = False
                timeouts += 1
                # if timeouts > 10:
                #    log.error("Timeout waiting for shard_out_queue")
                #    raise TimeoutError("Timeout waiting for shard_out_queue")
                continue
            if new_shards is None:
                running = False
                new_shards = []
            for node_id, shard in new_shards:
                # add shard to appropriate ShardLike
                submitted.remove(node_id)
                processed.add(node_id)

                parent_node = self.nodes[node_id]
                shardlike = shardlikes_by_url[node_id.channel]
                if node_id.package not in shardlike.visited:
                    shardlike.visited[node_id.package] = shard
                else:
                    pass  # e.g. monolithic repodata is already visited

                self.neighbors_2(pending, parent_node, shard)

            enqueue_pending()
            if not submitted:
                break

        cache_in_queue.put(None)
        cache_thread.join()
        network_thread.join()

    def neighbors_2(self, pending: set[NodeId], parent_node: Node, shard):
        """Neighbors (in-memory data only) for shortest_pipelined()"""

        mentioned_packages = list(shard_mentioned_packages_2(shard))
        # Move this into function for scope clarity
        for shardlike in self.shardlikes:
            for package in mentioned_packages:
                if package in shardlike:
                    new_node_id = NodeId(package, shardlike.url, shardlike.shard_url(package))
                    if new_node_id not in self.nodes:
                        new_node = Node(
                            distance=parent_node.distance + 1,
                            package=new_node_id.package,
                            channel=new_node_id.channel,
                            shard_url=new_node_id.shard_url,
                        )
                        self.nodes[new_node_id] = new_node

                        pending.add(new_node_id)


def build_repodata_subset(
    root_packages: Iterable[str],
    channels: Iterable[Channel],
    algorithm: Literal["shortest_dijkstra", "shortest_bfs", "shortest_pipelined"] = "shortest_bfs",
) -> dict[str, ShardBase]:
    """
    Retrieve all necessary information to build a repodata subset.

    Params:
        root_packages: iterable of installed and requested package names
        channels: iterable of Channel objects
        algorithm: desired traversal algorithm

    TODO: Remove `algorithm` parameter once we've made a firm decision on which to use.
    """
    channel_data = fetch_channels(channels)

    subset = RepodataSubset((*channel_data.values(),))
    getattr(subset, algorithm)(root_packages)
    log.debug("%d (channel, package) nodes discovered", len(subset.nodes))

    return channel_data


# region workers


def combine_batches_until_none(
    in_queue: Queue[list[NodeId] | None],
) -> Iterator[list[NodeId]]:
    """
    Combine lists from in_queue until we see None. Yield combined lists.
    """
    running = True
    while running and (batch := in_queue.get()) is not None:
        node_ids = batch[:]
        with suppress(queue.Empty):
            while True:  # loop exits with break or queue.Empty exception
                batch = in_queue.get_nowait()
                if batch is None:
                    # do the work but then quit
                    running = False
                    break
                else:
                    node_ids.extend(batch)
        yield node_ids


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

    for node_ids in combine_batches_until_none(in_queue):
        cached = cache.retrieve_multiple([node_id.shard_url for node_id in node_ids])

        # should we add this into retrieve_multiple?
        found: list[tuple[NodeId, ShardDict]] = []
        not_found: list[NodeId] = []
        for node_id in node_ids:
            if shard := cached.get(node_id.shard_url):
                found.append((node_id, shard))
            else:
                not_found.append(node_id)

        # Might wake up the network thread by calling it first:
        if not_found:
            network_out_queue.put(not_found)
        if found:
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

    shardlikes_by_url = {s.url: s for s in shardlikes}

    def fetch(s, url: str, node_id: NodeId):
        response = s.get(url)
        response.raise_for_status()
        data = response.content
        return (url, node_id, data)

    def submit(node_id):
        # this worker should only recieve network node_id's:
        shardlike = shardlikes_by_url[node_id.channel]
        if not isinstance(shardlike, Shards):
            log.warning("network_fetch_thread got non-network shardlike")
            return
        session = shardlike.session
        url = shardlikes_by_url[node_id.channel].shard_url(node_id.package)
        return executor.submit(fetch, session, url, node_id)

    def handle_result(future):
        url, node_id, data = future.result()
        log.debug("Fetch thread got %s (%s bytes)", url, len(data))
        # Decompress and parse. If it decodes as
        # msgpack.zst, insert into cache. Then put "known
        # good" shard into out queue.
        shard: ShardDict = msgpack.loads(
            dctx.decompress(data, max_output_size=ZSTD_MAX_SHARD_SIZE)
        )  # type: ignore[assign]
        cache.insert(AnnotatedRawShard(url, node_id.package, data))
        shard_out_queue.put([(node_id, shard)])

    def drain_futures(futures, last_batch=False):
        new_futures = []
        for future in concurrent.futures.as_completed(futures):
            try:
                handle_result(future)

                if not last_batch:
                    # Immediately enqueue more shards. It would also be
                    # appropriate to keep (number of threads) futures in-flight,
                    # instead of len(in_queue).
                    with suppress(queue.Empty):
                        extra_node_ids = in_queue.get_nowait()
                        if extra_node_ids is None:
                            in_queue.put(None)  # relay to top of loop
                        else:
                            for node_id in extra_node_ids:
                                new_futures.append(submit(node_id))

            except requests.exceptions.RequestException as e:
                log.error("Error fetching shard. %s", e)
            except Exception as e:
                log.exception("Unhandled exception in network thread", exc_info=e)
        return new_futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=_shards_connections()) as executor:
        new_futures = []
        for node_ids in combine_batches_until_none(in_queue):
            for node_id in node_ids:
                new_futures.append(submit(node_id))

            futures = new_futures  # futures is copied into as_completed()
            new_futures = drain_futures(futures)

        # This "last chance" fetch can create new pending nodes. Examine end criteria.
        drain_futures(new_futures, last_batch=True)

    shard_out_queue.put(None)


# endregion
