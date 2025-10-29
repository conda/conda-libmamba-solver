# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from conda.models.channel import Channel

from conda_libmamba_solver.shards import fetch_channels
from conda_libmamba_solver.shards_subset import RepodataSubset


def test_shortest_pipeline():
    pass


@pytest.mark.parametrize("algorithm", ("shortest", "fetch_shards_bfs"))
@pytest.mark.parametrize(
    "packages",
    (["python"], ["pandas"], ["django"], ["pytorch"]),
    ids=["python", "pandas", "django", "pytorch"],
)
def test_shortest_cold_cache(conda_cli, benchmark, algorithm, packages):
    """
    Ensure the `shortest` method retrieves all the nodes required to build a repodata

    TODO: For running extra benchmarks we could parameterize this to add more platforms
          (e.g. `osx-arm64` and `win-64`)
    """

    def setup():
        out, err, _ = conda_cli("clean", "--yes", "--all")
        assert not err
        channel = Channel("conda-forge-sharded/linux-64")
        channel_data = fetch_channels([channel.url()])

        assert len(channel_data) == 2

        subset = RepodataSubset((*channel_data.values(),))

        return (subset,), {}

    def target(subset):
        getattr(subset, algorithm)(packages)

    benchmark.pedantic(target, setup=setup, rounds=3)


@pytest.mark.limit_memory("512 MB")
@pytest.mark.parametrize("algorithm", ("shortest", "fetch_shards_bfs"))
@pytest.mark.parametrize(
    "packages",
    (["python"], ["pandas"], ["django"], ["pytorch"]),
    ids=["python", "pandas", "django", "pytorch"],
)
def test_shortest_warm_cache(conda_cli, benchmark, algorithm, packages):
    """
    Ensure the `shortest` method retrieves all the nodes required to build a repodata

    TODO: For running extra benchmarks we could parameterize this to add more platforms
          (e.g. `osx-arm64` and `win-64`)
    """
    # Clean cache just once
    out, err, _ = conda_cli("clean", "--yes", "--all")
    assert not err

    def setup():
        channel = Channel("conda-forge-sharded/linux-64")
        channel_data = fetch_channels([channel.url()])

        assert len(channel_data) == 2

        subset = RepodataSubset((*channel_data.values(),))

        return (subset,), {}

    def target(subset):
        getattr(subset, algorithm)(packages)

    benchmark.pedantic(target, setup=setup, rounds=3, warmup_rounds=1)
