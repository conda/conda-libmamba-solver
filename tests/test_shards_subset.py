# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
import json
from functools import cache
from pathlib import Path
from typing import NamedTuple

import pytest
import pytest_codspeed
from conda.models.channel import Channel

from conda_libmamba_solver.shards import fetch_channels
from conda_libmamba_solver.shards_subset import RepodataSubset, build_repodata_subset


class Scenario(NamedTuple):
    """
    Testing scenario representing packages to fetch and what settings to use
    """

    #: Unique identifier for the testing scenario
    name: str

    #: Packages to fetch
    packages: list[str]

    platform: str  # e.g. linux-64
    channel: str  # e.g. conda-forge

    #: Packages to fetch before benchmarks are run
    prefetch_packages: list[str]


@cache
def load_scenarios(scenario_filename: str) -> list[Scenario]:
    """Load scenarios from tests/data folder and load them as `Scenario` objects."""
    scenarios_file = Path(__file__).parent / "data" / scenario_filename

    scenarios_raw = json.loads(scenarios_file.read_text())

    return [Scenario(**scenario) for scenario in scenarios_raw.get("scenarios", [])]


def codspeed_supported():
    """
    TODO: temporary measure to skip these tests if we do not have pytest-codspeed >=4
    """
    try:
        major, minor, bug = pytest_codspeed.__version__.split(".")
        return int(major) >= 4
    except (ValueError, AttributeError):
        # If this fails, it means we want to skip this test
        return False


BASIC_FETCHING_SCENARIOS = load_scenarios("sharded_fetching_scenarios_basic.json")


@pytest.mark.skipif(not codspeed_supported(), reason="pytest-codspeed-version-4")
@pytest.mark.parametrize("cache_state", ("cold", "warm", "lukewarm"))
@pytest.mark.parametrize("algorithm", ("shortest_dijkstra", "shortest_bfs"))
@pytest.mark.parametrize(
    "scenario",
    (*BASIC_FETCHING_SCENARIOS,),
    ids=[scenario.name for scenario in BASIC_FETCHING_SCENARIOS],
)
def test_traversal_algorithm_benchmarks(
    conda_cli, benchmark, cache_state: str, algorithm: str, scenario: Scenario
):
    """
    Benchmark multiple traversal algorithms for retrieving repodata shards with
    a variety of cache states (defined below)

    cold:
        no shards in the SQLite cache database

    warm:
        all shards in the SQLite cache database

    lukewarm:
        only some of the shards are in the SQLite cache database
    """
    if cache_state == "warm":
        # Clean cache just once for "warm"
        out, err, _ = conda_cli("clean", "--yes", "--all")
        assert not err

    def setup():
        if cache_state != "warm":
            # For "cold" and "lukewarm", we want to clean cache before each round of benchmarking
            out, err, _ = conda_cli("clean", "--yes", "--all")
            assert not err

        channel = Channel(f"{scenario.channel}/{scenario.platform}")
        channel_data = fetch_channels([channel.url()])

        assert len(channel_data) == 2

        subset = RepodataSubset((*channel_data.values(),))

        if cache_state == "lukewarm":
            # Collect pre-fetch packages
            getattr(subset, algorithm)(scenario.prefetch_packages)

        return (subset,), {}

    def target(subset):
        getattr(subset, algorithm)(scenario.packages)

    benchmark.pedantic(target, setup=setup, rounds=3)


@pytest.mark.parametrize(
    "scenario",
    (*BASIC_FETCHING_SCENARIOS,),
    ids=[scenario.name for scenario in BASIC_FETCHING_SCENARIOS],
)
def test_traversal_algorithms_match(conda_cli, scenario: Scenario):
    """
    Ensure that all traversal algorithms return the same repodata subset.
    """
    channel = Channel(f"{scenario.channel}/{scenario.platform}")

    repodata_algorithm_map = {
        "shortest_dijkstra": build_repodata_subset(
            scenario.packages, [channel.url()], algorithm="shortest_dijkstra"
        ),
        "shortest_bfs": build_repodata_subset(
            scenario.packages, [channel.url()], algorithm="shortest_bfs"
        ),
    }

    for subdir in repodata_algorithm_map["shortest_dijkstra"].keys():
        repodatas = []

        for algorithm, repodata_subset in repodata_algorithm_map.items():
            repodatas.append(repodata_subset[subdir].build_repodata())

        assert all(x == y for x, y in zip(repodatas, repodatas[1:]))
