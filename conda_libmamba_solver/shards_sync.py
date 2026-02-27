# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Synchronous shard retrieval helpers. Retained for the benchmark
"reachable_bfs()" implementation but not used by the typical end user utilizing
the default "RepodataSubset.reachable_pipelined()" strategy.
"""

from __future__ import annotations

import concurrent.futures
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import msgpack
import zstandard
from conda.gateways.repodata import conda_http_errors

from . import shards_cache

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .shards import ShardLike, Shards
    from .shards_typing import ShardDict


def shardlike_fetch_shard(shardlike: ShardLike, package: str) -> ShardDict:
    """
    "Fetch" an individual shard from in-memory repodata.

    Update shardlike.visited with all not-None packages.

    Raise KeyError if package is not in the index.
    """
    shard = shardlike.shards[package]
    shardlike.visited[package] = shard
    return shard


def shardlike_fetch_shards(shardlike: ShardLike, packages: Iterable[str]) -> dict[str, ShardDict]:
    """
    Fetch multiple in-memory shards in one go.

    Update shardlike.visited with all not-None packages.
    """
    return {package: shardlike_fetch_shard(shardlike, package) for package in packages}


def shards_fetch_shard(shards: Shards, package: str) -> ShardDict:
    """
    Fetch an individual network shard for the given package.

    Raise KeyError if package is not in the index.
    """
    return shards_fetch_shards(shards, [package])[package]


def shards_fetch_shards(shards: Shards, packages: Iterable[str]) -> dict[str, ShardDict]:
    """
    Return mapping of package names to Shard for given packages.

    If a shard is already in shards.visited, it is not fetched again.
    """
    from .shards import _shards_connections

    results = {}

    def fetch(session, url, package_to_fetch):
        response = session.get(url)
        response.raise_for_status()
        data = response.content

        return shards_cache.AnnotatedRawShard(
            url=url, package=package_to_fetch, compressed_shard=data
        )

    packages = sorted(list(packages))
    urls_packages = {}
    for package in packages:
        if package in shards.visited:
            results[package] = shards.visited[package]
        else:
            urls_packages[shards.shard_url(package)] = package

    with concurrent.futures.ThreadPoolExecutor(max_workers=_shards_connections()) as executor:
        futures = {
            executor.submit(fetch, shards.session, url, package): (url, package)
            for url, package in urls_packages.items()
            if package not in results
        }
        for future in concurrent.futures.as_completed(futures):
            log.debug(". %s", futures[future])
            url, package = futures[future]
            shards_process_fetch_result(shards, future, url, package, results)

    shards.visited.update(results)

    return results


def shards_process_fetch_result(shards: Shards, future, url, package, results):
    """
    Process a single fetched shard and update cache/results.
    """
    from .shards import ZSTD_MAX_SHARD_SIZE

    if shards.shards_cache is None:
        raise ValueError("self.shards_cache is None")

    with conda_http_errors(url, package):
        fetch_result = future.result()

    results[fetch_result.package] = msgpack.loads(
        zstandard.decompress(fetch_result.compressed_shard, max_output_size=ZSTD_MAX_SHARD_SIZE)
    )
    shards.shards_cache.insert(fetch_result)


def batch_retrieve_from_cache(sharded: list[Shards], packages: list[str]):
    """
    Given a list of Shards objects and a list of package names, fetch all URLs
    from a shared local cache, and update Shards with those per-package shards.
    Return the remaining URLs that must be fetched from the network.
    """
    from .shards import Shards

    sharded = [shardlike for shardlike in sharded if isinstance(shardlike, Shards)]

    wanted = []
    for shard in sharded:
        for package_name in packages:
            if package_name in shard:
                wanted.append((shard, package_name, shard.shard_url(package_name)))

    log.debug("%d shards to fetch", len(wanted))

    if not sharded:
        log.debug("No sharded channels found.")
        return wanted

    shared_shard_cache = sharded[0].shards_cache
    from_cache = shared_shard_cache.retrieve_multiple([shard_url for *_, shard_url in wanted])

    for shard, package, shard_url in wanted:
        if from_cache_shard := from_cache.get(shard_url):
            shard.visit_shard(package, from_cache_shard)

    return wanted


def batch_retrieve_from_network(wanted: list[tuple[Shards, str, str]]):
    """
    Given a list of (Shards, package name, shard URL) tuples, group by Shards and call fetch_shards
    with a list of all URLs for that Shard.
    """
    shard_packages: dict[Shards, list[str]] = defaultdict(list)
    for shard, package, _ in wanted:
        shard_packages[shard].append(package)

    for shard, packages in shard_packages.items():
        shard.fetch_shards(packages)
