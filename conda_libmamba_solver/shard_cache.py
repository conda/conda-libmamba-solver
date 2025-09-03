"""
Cache suitable for shards, not allowed to change because they are named
after their own sha256 hash.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import msgpack
import zstandard

if TYPE_CHECKING:
    from pathlib import Path

SHARD_CACHE_NAME = "repodata_shards.db"


Shard = TypedDict("Shard", {"packages": dict[str, dict], "packages.conda": dict[str, dict]})


@dataclass
class AnnotatedRawShard:
    def __init__(self, url: str, package: str, compressed_shard: bytes):
        # debugging
        assert "://" in url
        assert "://" not in package

        self.url = url
        self.package = package  # remove this field to avoid confusion?
        self.compressed_shard = compressed_shard

    url: str
    package: str
    compressed_shard: bytes


def connect(dburi="cache.db"):
    """
    Get database connection.

    dburi: uri-style sqlite database filename; accepts certain ?= parameters.
    """
    conn = sqlite3.connect(dburi, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


class ShardCache:
    """
    Handle caching for individual shards (not the index of shards).
    """

    def __init__(self, base: Path):
        """
        base: directory and filename prefix for cache.
        """
        # base includes /<hash for particular repodata url>
        self.base = base
        self.connect()

    def connect(self):
        dburi = f"file://{str(self.base / SHARD_CACHE_NAME)}"
        self.conn = connect(dburi)
        # this schema will also get confused if we merge packages into a single
        # shard, but the package name should be advisory.
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS shards ("
            "url TEXT PRIMARY KEY, package TEXT, shard BLOB, "
            "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )

    def insert(self, raw_shard: AnnotatedRawShard):
        """
        Args:
            url: of shard
            package: package name
            raw_shard: msgpack.zst compressed shard data
        """
        # decompress and return shard for convenience, also to validate? unless
        # caller would rather retrieve the shard from another thread.
        with self.conn as c:
            c.execute(
                "INSERT OR IGNORE INTO SHARDS (url, package, shard) VALUES (?, ?, ?)",
                (raw_shard.url, raw_shard.package, raw_shard.compressed_shard),
            )

    def retrieve(self, url) -> Shard | None:
        with self.conn as c:
            row = c.execute("SELECT shard FROM shards WHERE url = ?", (url,)).fetchone()
            return msgpack.loads(zstandard.decompress(row["shard"])) if row else None  # type: ignore

    def retrieve_multiple(self, urls: list[str]) -> dict[str, Shard | None]:
        """
        Query database for cached shard urls.

        Return a dict of urls in cache mapping to the Shard or None if not present.
        """
        # in sqlite, multiple queries are very fast since there is no server.
        # "with self.conn" may have overhead. nevertheless, try querying
        # multiple in a single call.
        query = f"SELECT url, shard FROM shards WHERE url IN ({','.join(('?',) * len(urls))}) ORDER BY url"
        with self.conn as c:
            result: dict[str, Shard | None] = {
                row["url"]: msgpack.loads(zstandard.decompress(row["shard"])) if row else None
                for row in c.execute(query, urls)  # type: ignore
            }
            return result
