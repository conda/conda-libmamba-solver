# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests related to sharded repodata.
"""

from pathlib import Path

import msgpack
import zstandard
from conda.core.subdir_data import SubdirData
from conda.models.channel import Channel

from conda_libmamba_solver import shard_cache


def test_shard_cache(tmp_path: Path):
    cache = shard_cache.ShardCache(tmp_path)

    fake_shard = {"foo": "bar"}
    annotated_shard = shard_cache.AnnotatedRawShard(
        "https://foo",
        "foo",
        zstandard.compress(msgpack.dumps(fake_shard)),  # type: ignore
    )
    cache.insert(annotated_shard)

    data = cache.retrieve(annotated_shard.url)
    assert data == fake_shard
    assert data is not fake_shard

    data2 = cache.retrieve("notfound")
    assert data2 is None

    assert (tmp_path / shard_cache.SHARD_CACHE_NAME).exists()


def test_shard_cache_2():
    """
    Check that existing shards cache has been populated with matching shards,
    package names.
    """
    subdir_data = SubdirData(Channel("main/noarch"))
    # accepts directory that cache goes into, not database filename
    cache = shard_cache.ShardCache(Path(subdir_data.cache_path_base).parent)
    extant = {}
    for row in cache.conn.execute("SELECT url, package, shard, timestamp FROM shards"):
        extant[row["url"]] = shard_cache.AnnotatedRawShard(
            url=row["url"], package=row["package"], compressed_shard=row["shard"]
        )

    # check that url's are in the url field, and package names are in the package name field
    for url in extant:
        if shard := cache.retrieve(url):
            assert "://" in extant[url].url
            assert "://" not in extant[url].package
            assert all(
                package["name"] == extant[url].package for package in shard["packages"].values()
            )
            assert all(
                package["name"] == extant[url].package
                for package in shard["packages.conda"].values()
            )
