# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
http/2 network fetching of shards, proof of concept.

The greater concurrency of http/2 should be very beneficial for sharded
repodata.
"""

import asyncio
import concurrent.futures
import logging
from queue import Queue

import httpx
import msgpack
import zstandard

from .shards import ShardBase, Shards
from .shards_cache import ZSTD_MAX_SHARD_SIZE, AnnotatedRawShard, ShardCache
from .shards_subset import NodeId, exception_to_queue
from .shards_typing import ShardDict

log = logging.getLogger(__name__)


async def _network_fetch_loop_httpx(
    in_queue: Queue[list[NodeId] | None],
    shard_out_queue: Queue[list[tuple[NodeId, ShardDict] | Exception] | None],
    cache: ShardCache,
    shardlikes: list[ShardBase],
):
    cache = cache.copy()
    dctx = zstandard.ZstdDecompressor()
    shardlikes_by_url = {s.url: s for s in shardlikes}

    async def fetch(client: httpx.AsyncClient, url: str, node_id: NodeId):
        response = await client.get(url)
        response.raise_for_status()
        data = response.content
        return (url, node_id, data)

    async def submit(client, node_id):
        # this worker should only recieve network node_id's:
        shardlike = shardlikes_by_url[node_id.channel]
        if not isinstance(shardlike, Shards):
            log.warning("network_fetch_thread got non-network shardlike")
            return
        url = shardlikes_by_url[node_id.channel].shard_url(node_id.package)
        return await fetch(client, url, node_id)

    async def get_work():
        with concurrent.futures.ThreadPoolExecutor() as thread_pool:
            while True:
                batch = await asyncio.get_running_loop().run_in_executor(thread_pool, in_queue.get)
                if batch is None:
                    break
                for node_id in batch:
                    yield node_id

    def handle_result(task: asyncio.Task):
        tasks.remove(task)
        url, node_id, data = task.result()  # can raise exceptions from fetch
        log.debug("Fetch %s (%s bytes)", url, len(data))
        # Decompress and parse. If it decodes as
        # msgpack.zst, insert into cache. Then put "known
        # good" shard into out queue.
        shard: ShardDict = msgpack.loads(
            dctx.decompress(data, max_output_size=ZSTD_MAX_SHARD_SIZE)
        )  # type: ignore[assign]
        cache.insert(AnnotatedRawShard(url, node_id.package, data))
        shard_out_queue.put([(node_id, shard)])

    async with httpx.AsyncClient(http2=True) as client:
        results: list[tuple[NodeId, ShardDict] | Exception] = []

        tasks = set()
        async for node_id in get_work():
            task = asyncio.create_task(submit(client, node_id))
            task.add_done_callback(handle_result)
            tasks.add(task)

        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                results.append(e)


@exception_to_queue
def network_fetch_thread_httpx(
    in_queue: Queue[list[NodeId] | None],
    shard_out_queue: Queue[list[tuple[NodeId, ShardDict] | Exception] | None],
    cache: ShardCache,
    shardlikes: list[ShardBase],
):
    """
    in_queue contains urls to fetch over the network.
    While the in_queue has not received a sentinel None, empty everything from
    the queue. Fetch all of them over the network. Fetched shards go to
    shard_out_queue.
    Unhandled exceptions also go to shard_out_queue, and exit this thread.
    """
    for shard in shardlikes:
        if not shard.url.startswith("http"):
            raise ValueError(f"Unsupported shard in http/2 fetch thread: {shard.url}")
    asyncio.run(_network_fetch_loop_httpx(in_queue, shard_out_queue, cache, shardlikes))
