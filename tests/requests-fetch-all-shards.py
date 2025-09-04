#!/usr/bin/env python
"""
Collect all repodata shards from fast.prefix.dev/conda-forge.

Takes about 23s on a hot CDN cache or 5 minutes on a cold remote cache.
"""

import concurrent.futures
import sqlite3
import time

import msgpack
import requests
import zstandard


def connect(dburi="cache.db"):
    """
    Get database connection.

    dburi: uri-style sqlite database filename; accepts certain ?= parameters.
    """
    conn = sqlite3.connect(dburi, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = 1")  # less fsync, good for WAL mode
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def to_url(shard):
    return f"https://fast.prefix.dev/conda-forge/linux-64/shards/{shard.hex()}.msgpack.zst"


base_url = "https://fast.prefix.dev/conda-forge/linux-64/repodata_shards.msgpack.zst"

s = requests.Session()

index = msgpack.loads(zstandard.decompress(s.get(base_url).content))

conn = connect("conda-forge-shards.db")
conn.execute("CREATE TABLE IF NOT EXISTS shards (url TEXT PRIMARY KEY, package TEXT, shard BLOB)")


def shard_urls():
    for package, hash in index["shards"].items():
        yield package, to_url(hash)


def fetch(s, package, url):
    b1 = time.time_ns()
    data = s.get(url).content
    e1 = time.time_ns()
    print(f"{(e1-b1)/1e9}s", package, url)
    return data


begin = time.time_ns()

# beneficial to have thread pool larger than requests' default 10 max
# connections per session
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    futures = {
        executor.submit(fetch, s, package, url): (url, package) for package, url in shard_urls()
    }
    # May be inconvenient to cancel a large number of futures. Also, ctrl-C doesn't work reliably out of the box.
    for future in concurrent.futures.as_completed(futures):
        print(".", futures[future])
        data = future.result()
        url, package = futures[future]
        with conn as c:
            c.execute(
                "INSERT OR IGNORE INTO SHARDS (url, package, shard) VALUES (?, ?, ?)",
                (url, package, data),
            )

end = time.time_ns()

conn.close()

print(f"Took {(end-begin)/1e9} seconds")
