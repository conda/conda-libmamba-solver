# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Sharded repodata subsets.

Traverse dependencies of installed and to-be-installed packages to generate a
useful subset for the solver.

The algorithm developed here is a direct result of the following CEP:

- https://conda.org/learn/ceps/cep-0016 (Sharded Repodata)

In this algorithm we treat a (channel, package name) as a node, its dependencies
as edges. We then traverse all edges to discover all reachable (channel, package
name) tuples. The solver should be able to find a solution with only this
subset.

This subset is overgenerous since the user is unlikely to want to install very
old packages and their dependencies. If this is too slow, we could deploy
heuristics that automatically ignore older package versions. We could also allow
the user to configure minimum versions of common packages and ignore older
versions and their dependencies, falling back to a full solve if unsatisfiable.

We treat both sharded and monolithic repodata as if they were made up of
per-package shards, computing a subset of both. This is because it is possible
for the monolithic repodata to mention packages that exist in the true sharded
repodata but would not be found by only traversing the shards.

We treat all repodata as sharded, even if no actual sharded repodata has been
found.

## Example usage

The following constructs several repodata (`noarch` and `linux-64`) from a
single channel name and a list of root packages:

``` from conda.models.channel import Channel from
conda_libmamba_solver.shards_subset import build_repodata_subset

channel = Channel("conda-forge-sharded/linux-64") channel_data =
build_repodata_subset(["python", "pandas"], [channel.url()]) repodata = {}

for url in channel_data:
    repodata[url] = channel_data.build_repodata()

# ... this is what's fed to the solver ```

"""

from __future__ import annotations

import functools
import logging
import queue
import sys
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import TYPE_CHECKING

import conda.gateways.repodata
import msgpack
import zstandard

from conda_libmamba_solver import shards_cache
from conda_libmamba_solver.shards_cache import AnnotatedRawShard

from .shards import (
    ZSTD_MAX_SHARD_SIZE,
    Shards,
    _shards_connections,
    batch_retrieve_from_cache,
    batch_retrieve_from_network,
    fetch_channels,
    shard_mentioned_packages,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from queue import SimpleQueue as Queue
    from typing import Literal, TypeVar

    from conda.models.channel import Channel

    from conda_libmamba_solver.shards_cache import ShardCache
    from conda_libmamba_solver.shards_typing import ShardDict

    from .shards import (
        ShardBase,
    )

# Waiting for worker threads to shutdown cleanly, or raise error.
THREAD_WAIT_TIMEOUT = 5  # seconds
REACHABLE_PIPELINED_MAX_TIMEOUTS = 10  # number of times we can timeout waiting for shards


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
    shardlikes: Sequence[ShardBase]
    DEFAULT_STRATEGY = "pipelined"

    def __init__(self, shardlikes: Iterable[ShardBase]):
        self.nodes = {}
        self.shardlikes = list(shardlikes)

    @classmethod
    def has_strategy(cls, strategy: str) -> bool:
        """
        Return True if this class provides the named shard traversal strategy.
        """
        return hasattr(cls, f"reachable_{strategy}")

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

            for package in shard_mentioned_packages(shard):
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

    def reachable_bfs(self, root_packages):
        """
        Fetch all packages reachable from `root_packages`' by following
        dependencies using the "breadth-first search" algorithm.

        Update associated `self.shardlikes` to contain enough data to build a
        repodata subset.
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
                if node.visited:  # pragma: no cover
                    continue  # we should never add visited nodes to node_queue
                node.visited = True

                for next_node, _ in self.outgoing(node):
                    if not next_node.visited:
                        node_queue.append(next_node)

    def reachable_pipelined(self, root_packages):
        """
        Fetch all packages reachable from `root_packages`' by following
        dependencies.

        Build repodata subset using concurrent threads to follow dependencies,
        fetch from cache, and fetch from network.
        """

        # Ignore cache on shards object, use our own. Necessary if there are no
        # sharded channels.
        cache = shards_cache.ShardCache(Path(conda.gateways.repodata.create_cache_dir()))

        cache_in_queue: SimpleQueue[list[NodeId] | None] = SimpleQueue()
        shard_out_queue: SimpleQueue[list[tuple[NodeId, ShardDict]] | Exception] = SimpleQueue()
        cache_miss_queue: SimpleQueue[list[NodeId] | None] = SimpleQueue()

        cache_thread = threading.Thread(
            target=cache_fetch_thread,
            args=(cache_in_queue, shard_out_queue, cache_miss_queue, cache),
            daemon=True,  # may have to set to False if we ever want to run in a subinterpreter
        )

        network_thread = threading.Thread(
            target=network_fetch_thread,
            args=(cache_miss_queue, shard_out_queue, cache, self.shardlikes),
            daemon=True,
        )

        try:
            cache_thread.start()
            network_thread.start()
            self.pipelined_main_thread(
                root_packages, cache_in_queue, shard_out_queue, cache_thread, network_thread
            )
        finally:
            cache_in_queue.put(None)
            # These should finish almost immediately, but if not, raise an error:
            cache_thread.join(THREAD_WAIT_TIMEOUT)
            network_thread.join(THREAD_WAIT_TIMEOUT)

    def pipelined_main_thread(
        self, root_packages, cache_in_queue, shard_out_queue, cache_thread, network_thread
    ):
        """
        Run reachibility algorithm given queues to submit and receive shards.
        """
        shardlikes_by_url = {s.url: s for s in self.shardlikes}
        pending: set[NodeId] = set()
        in_flight: set[NodeId] = set()
        timeouts = 0
        shutdown_initiated = False

        self.nodes = {}

        # create start condition
        parent_node = Node(0)
        self.visit_node(pending, parent_node, root_packages)

        def pump():
            """
            Find shards we already have and those we need. Submit those need to
            cache_in_queue, those we have to shard_out_queue.
            """
            have, need = self.drain_pending(pending, shardlikes_by_url)
            if need:
                in_flight.update(need)
                cache_in_queue.put(need)
            if have:
                in_flight.update(node_id for node_id, _ in have)
                shard_out_queue.put(have)
            return len(have) + len(need)

        running = True
        while running:
            pump()
            try:
                new_shards = shard_out_queue.get(timeout=1)
                if new_shards is None:
                    running = False
                    continue  # or break
                if isinstance(new_shards, Exception):  # error propagated from worker thread
                    raise new_shards
            except queue.Empty:
                pump_count = pump()
                log.debug("Shard timeout %s, pump_count=%d", timeouts, pump_count)
                log.debug("pending: %s...", sorted(str(node_id) for node_id in pending)[:10])
                log.debug("in_flight: %s...", sorted(str(node_id) for node_id in in_flight)[:10])
                log.debug("nodes: %d", len(self.nodes))
                log.debug("cache_thread.is_alive(): %s", cache_thread.is_alive())
                log.debug("network_thread.is_alive(): %s", network_thread.is_alive())
                log.debug("shard_out_queue.qsize(): %s", shard_out_queue.qsize())
                if not pending and not in_flight:
                    log.debug("All shards have finished processing")
                    break
                timeouts += 1
                if timeouts > REACHABLE_PIPELINED_MAX_TIMEOUTS:
                    raise TimeoutError(
                        f"Timeout waiting for shard_out_queue after {timeouts} attempts. "
                        f"pending={len(pending)}, in_flight={len(in_flight)}, "
                        f"cache_thread_alive={cache_thread.is_alive()}, "
                        f"network_thread_alive={network_thread.is_alive()}"
                    )
                continue  # immediately calls pump() at top of loop

            for node_id, shard in new_shards:
                in_flight.remove(node_id)

                # add shard to appropriate ShardLike
                parent_node = self.nodes[node_id]
                shardlike = shardlikes_by_url[node_id.channel]
                shardlike.visit_shard(node_id.package, shard)

                self.visit_node(pending, parent_node, shard_mentioned_packages(shard))

            if not pending and not in_flight and not shutdown_initiated:
                log.debug("Initiating shutdown: sending None to cache_in_queue")
                cache_in_queue.put(None)
                shutdown_initiated = True

    def visit_node(
        self, pending: set[NodeId], parent_node: Node, mentioned_packages: Iterable[str]
    ):
        """Broadcast mentioned packages across channels to pending."""
        # NOTE we have visit for Nodes which is used in the graph traversal
        # algorithm, and a separate visit for ShardBase which means "include
        # this package in the output repodata".
        for package in mentioned_packages:
            for shardlike in self.shardlikes:
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

        parent_node.visited = True

    def drain_pending(
        self, pending: set[NodeId], shardlikes_by_url: dict[str, ShardBase]
    ) -> tuple[list[tuple[NodeId, ShardDict]], list[NodeId]]:
        """
        Check pending for in-memory shards.
        Clear pending.

        Return a list of shards we have and shards we need to fetch.
        """
        shards_need = []
        shards_have = []
        for node_id in pending:
            # we should already have these nodes.
            shardlike = shardlikes_by_url[node_id.channel]
            if shardlike.shard_loaded(node_id.package):  # for monolithic repodata
                shards_have.append((node_id, shardlike.visit_package(node_id.package)))
            else:
                if self.nodes[node_id].visited:  # pragma: no cover
                    log.debug("Skip visited, should not be reached")
                    continue
                shards_need.append(node_id)
        pending.clear()
        return shards_have, shards_need


def build_repodata_subset(
    root_packages: Iterable[str],
    channels: dict[str, Channel],
    algorithm: Literal["bfs", "pipelined"] = RepodataSubset.DEFAULT_STRATEGY,
) -> dict[str, ShardBase]:
    """
    Retrieve all necessary information to build a repodata subset.

    Params:
        root_packages: iterable of installed and requested package names
        channels: iterable of Channel objects
        algorithm: desired traversal algorithm
    """
    if isinstance(channels, dict):  # True when called by LibMambaIndexHelper
        channels_: list[Channel] = list(channels.values())
    else:
        channels_ = channels
    channel_data = fetch_channels(channels_)

    subset = RepodataSubset((*channel_data.values(),))
    getattr(subset, f"reachable_{algorithm}")(root_packages)
    log.debug("%d (channel, package) nodes discovered", len(subset.nodes))

    return channel_data


# region workers

if TYPE_CHECKING:
    _T = TypeVar("_T")


def combine_batches_until_none(
    in_queue: Queue[Sequence[_T] | None],
) -> Iterator[Sequence[_T]]:
    """
    Combine lists from in_queue until we see None. Yield combined lists.
    """
    running = True
    while running:
        try:
            # Add timeout to prevent indefinite blocking if producer thread fails
            batch = in_queue.get(timeout=5)
            if batch is None:
                break
        except queue.Empty:
            # If we timeout, continue waiting - producer might still send data
            continue

        node_ids = list(batch)
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


def exception_to_queue(func):
    """
    Decorator to send unhandled exceptions to the second argument out_queue.
    """

    @functools.wraps(func)
    def wrapper(in_queue, out_queue, *args, **kwargs):
        try:
            return func(in_queue, out_queue, *args, **kwargs)
        except Exception as e:
            in_queue.put(None)  # signal termination
            out_queue.put(e)

    return wrapper


@exception_to_queue
def cache_fetch_thread(
    in_queue: Queue[Sequence[NodeId] | None],
    shard_out_queue: Queue[Sequence[tuple[NodeId, ShardDict] | Exception] | None],
    network_out_queue: Queue[Sequence[NodeId] | None],
    cache: ShardCache,
):
    """
    Fetch batches of shards from cache until in_queue sees None. Enqueue found
    shards to shard_out_queue, and not found shards to network_out_queue.

    When we see None on in_queue, send None to both out queues and exit.

    Args:
        in_queue: NodeId (URLs) to fetch.
        shard_out_queue: fetched shards sent to queue.
        network_out_queue: cache misses forwarded to queue. Same queue is
            network_fetch_thread's in_queue.
        cache: used to retrieve shards.
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


@exception_to_queue
def network_fetch_thread(
    in_queue: Queue[Sequence[NodeId | Future] | None],
    shard_out_queue: Queue[list[tuple[NodeId, ShardDict] | Exception] | None],
    cache: ShardCache,
    shardlikes: list[ShardBase],
):
    """
    Fetch shards from the network that are received on in_queue, until we see
    None.

    Unhandled exceptions also go to shard_out_queue, and exit this thread.

    Args:
        in_queue: NodeId (URLs) to fetch.
        shard_out_queue: fetched shards sent to queue.
        cache: once shards are decoded they are stored in cache.
        shardlikes: list of (network-only) shard index objects.
    """
    cache = cache.copy()
    dctx = zstandard.ZstdDecompressor(max_window_size=ZSTD_MAX_SHARD_SIZE)
    shardlikes_by_url = {s.url: s for s in shardlikes}

    def fetch(s, url: str, node_id: NodeId):
        response = s.get(url)
        response.raise_for_status()
        data = response.content
        return url, node_id, data

    def submit(node_id: NodeId):
        # this worker should only receive network node_id's:
        shardlike = shardlikes_by_url[node_id.channel]
        if not isinstance(shardlike, Shards):
            raise TypeError("network_fetch_thread got non-network shardlike")
        session = shardlike.session
        url = shardlikes_by_url[node_id.channel].shard_url(node_id.package)
        return executor.submit(fetch, session, url, node_id)

    def handle_result(future: Future):
        url, node_id, data = future.result()
        log.debug("Fetch thread got %s (%s bytes)", url, len(data))
        # Decompress and parse. If it decodes as
        # msgpack.zst, insert into cache. Then put "known
        # good" shard into out queue.
        shard: ShardDict = msgpack.loads(
            dctx.decompress(data, max_output_size=ZSTD_MAX_SHARD_SIZE)
        )  # type: ignore[assign]
        # We could send this back into the cache thread instead to
        # serialize access to sqlite3 if lock contention becomes an issue.
        cache.insert(AnnotatedRawShard(url, node_id.package, data))
        shard_out_queue.put([(node_id, shard)])

    def result_to_in_queue(future: Future):
        # Simplify waiting by putting responses back into in_queue. This
        # function is called in the ThreadPoolExecutor's thread, but we want to
        # serialize result processing in the network_fetch_thread.
        in_queue.put([future])

    with ThreadPoolExecutor(max_workers=_shards_connections()) as executor:
        for node_ids_and_results in combine_batches_until_none(in_queue):
            for node_id_or_result in node_ids_and_results:
                if isinstance(node_id_or_result, Future):
                    handle_result(node_id_or_result)
                else:
                    future = submit(node_id_or_result)
                    future.add_done_callback(result_to_in_queue)
        # TODO call executor.shutdown(cancel_futures=True) on error or otherwise
        # prevent new HTTP requests from being started e.g. "skip" flag in
        # fetch() function. Also possible to shutdown(wait=False).


# endregion
