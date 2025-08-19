"""
Cache suitable for shards, not allowed to change because they are named
after their own sha256 hash.
"""

import sqlite3
from pathlib import Path

import msgpack
import zstandard

SHARD_CACHE_NAME = "repodata_shards.db"


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
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS shards ("
            "url TEXT PRIMARY KEY, package TEXT, shard BLOB, "
            "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )

    def insert(self, url, package, shard):
        with self.conn as c:
            c.execute(
                "INSERT OR IGNORE INTO SHARDS (url, package, shard) VALUES (?, ?, ?)",
                (url, package, shard),
            )

    def retrieve(self, url):
        with self.conn as c:
            row = c.execute("SELECT shard FROM shards WHERE url = ?", (url,)).fetchone()
            return msgpack.loads(zstandard.decompress(row["shard"])) if row else None
