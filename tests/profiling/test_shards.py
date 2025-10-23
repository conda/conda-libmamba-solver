"""
Profile shard fetch components
"""

from pathlib import Path

import pytest
import msgpack
import zstandard

from conda_libmamba_solver import shards_cache



@pytest.mark.benchmark
@pytest.mark.parametrize("retrieval_type", ["retrieve_multiple", "retrieve_single"])
def test_shard_cache_multiple(retrieval_type, tmp_path: Path):
    """
    Measure the difference between `shards_cache.retrieve_multiple` and `shards_cache.retrieve`.

    `shards_cache.retrieve_multiple should be faster than `shards_cache.retrieve`.
    """
    NUM_FAKE_SHARDS = 64

    cache = shards_cache.ShardCache(tmp_path)
    fake_shards = []

    compressor = zstandard.ZstdCompressor(level=1)
    for i in range(NUM_FAKE_SHARDS):
        fake_shard = {f"foo{i}": "bar"}
        annotated_shard = shards_cache.AnnotatedRawShard(
            f"https://foo{i}",
            f"foo{i}",
            compressor.compress(msgpack.dumps(fake_shard)),  # type: ignore
        )
        cache.insert(annotated_shard)
        fake_shards.append(annotated_shard)

    if retrieval_type == "retrieve_multiple":
        retrieved = cache.retrieve_multiple([shard.url for shard in fake_shards])
        assert len(retrieved) == NUM_FAKE_SHARDS

    elif retrieval_type == "retrieve_single":
        retrieved = {}
        for i, url in enumerate([shard.url for shard in fake_shards]):
            single = cache.retrieve(url)
            retrieved[url] = single

        assert len(retrieved) == NUM_FAKE_SHARDS

    assert (tmp_path / shards_cache.SHARD_CACHE_NAME).exists()
