#!/usr/bin/env python
"""
Collect all repodata shards from fast.prefix.dev/conda-forge.

Takes about 12.7 seconds, a few requests fail and would need to be re-tried.
"""

import asyncio
import sqlite3
import time

import httpx
import msgpack
import zstandard


def connect(dburi="cache.db"):
    """
    Get database connection.

    dburi: uri-style sqlite database filename; accepts certain ?= parameters.
    """
    conn = sqlite3.connect(dburi, uri=True)
    conn.row_factory = sqlite3.Row
    # conn.execute("PRAGMA journal_mode = WAL")
    # conn.execute("PRAGMA synchronous = 1")  # less fsync, good for WAL mode
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def to_url(shard):
    return f"https://fast.prefix.dev/conda-forge/linux-64/shards/{shard.hex()}.msgpack.zst"


base_url = "https://fast.prefix.dev/conda-forge/linux-64/repodata_shards.msgpack.zst"


conn = connect("conda-forge-shards-httpx.db")
conn.execute("CREATE TABLE IF NOT EXISTS shards (url TEXT PRIMARY KEY, package TEXT, shard BLOB)") # also last-used?


def shard_urls(index):
    for package, hash in index["shards"].items():
        yield package, to_url(hash)


async def get_one(sem, client, url, package):
    try:
        async with sem:
            response = await client.get(url)
        return package, url, response
    except httpx.RemoteProtocolError as err:
        return package, url, err


async def main():
    async with httpx.AsyncClient(http2=True, timeout=60) as client:
        # client._transport._pool._max_connections # is 100, does this have
        # anything to do with http/2?
        # See https://github.com/encode/httpx/issues/1171
        # Default 5s timeout is bigger issue in request failures.
        print("Max in-flight", client._transport._pool._max_connections)
        sem = asyncio.Semaphore(client._transport._pool._max_connections)
        response = await client.get(base_url)
        print(response.status_code, response.headers)
        index = msgpack.loads(zstandard.decompress(response.content))
        awaitables = [
            get_one(sem, client, shard_url, package) for package, shard_url in shard_urls(index)
        ]
        for awaitable in asyncio.as_completed(awaitables):
            package, url, response = await awaitable
            if isinstance(response, Exception):
                print("x", end="")
                # add (package, url) to retry list?
                continue
            data = response.content
            with conn as c:
                c.execute(
                    "INSERT OR IGNORE INTO SHARDS (url, package, shard) VALUES (?, ?, ?)",
                    (url, package, data),
                )
            print(".", end="")


if __name__ == "__main__":
    begin = time.time_ns()
    asyncio.run(main())
    end = time.time_ns()
    print(f"Took {(end-begin)/1e9} seconds")
    conn.close()
