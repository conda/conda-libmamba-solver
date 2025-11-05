# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import pytest_codspeed
from conda.common.compat import on_win
from conda.models.channel import Channel
from conda.testing.fixtures import CondaCLIFixture

from conda_libmamba_solver.shards import fetch_channels
from conda_libmamba_solver.shards_subset import RepodataSubset, build_repodata_subset

TESTING_SCENARIOS = [
    {
        "name": "python",
        "packages": ["python"],
        "prefetch_packages": [],
        "channel": "conda-forge-sharded",
        "platform": "linux-64",
    },
    {
        "name": "data-science-basic",
        "packages": ["numpy", "pandas", "matplotlib"],
        "prefetch_packages": ["python"],
        "channel": "conda-forge-sharded",
        "platform": "linux-64",
    },
    {
        "name": "data-science-ml",
        "packages": ["scikit-learn", "matplotlib"],
        "prefetch_packages": ["python", "numpy"],
        "channel": "conda-forge-sharded",
        "platform": "linux-64",
    },
    {
        "name": "web-development",
        "packages": ["django", "celery"],
        "prefetch_packages": ["python", "requests"],
        "channel": "conda-forge-sharded",
        "platform": "linux-64",
    },
    {
        "name": "scientific-computing",
        "packages": ["scipy", "sympy", "pytorch"],
        "prefetch_packages": ["python", "numpy", "pandas"],
        "channel": "conda-forge-sharded",
        "platform": "linux-64",
    },
    {
        "name": "devops-automation",
        "packages": ["ansible", "pyyaml", "jinja2"],
        "prefetch_packages": ["python"],
        "channel": "conda-forge-sharded",
        "platform": "linux-64",
    },
]


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


def clean_cache(conda_cli: CondaCLIFixture):
    """
    Clean cache and assert it completed without error except on Windows
    """
    out, err, return_code = conda_cli("clean", "--yes", "--all")

    # Windows CI runners cannot reliably remove this file, so we don't care
    # about this assertion on that platform.

    # "err" will include log.debug output on certain test runners, so we can't
    # check it to determine whether there was an error.
    if not on_win:
        assert not return_code, "conda clean returned {return_code} != 0"


@pytest.mark.skipif(not codspeed_supported(), reason="pytest-codspeed-version-4")
@pytest.mark.parametrize("cache_state", ("cold", "warm", "lukewarm"))
@pytest.mark.parametrize("algorithm", ("dijkstra", "bfs"))
@pytest.mark.parametrize(
    "scenario",
    TESTING_SCENARIOS,
    ids=[scenario.get("name") for scenario in TESTING_SCENARIOS],
)
def test_traversal_algorithm_benchmarks(
    conda_cli: CondaCLIFixture, benchmark, cache_state: str, algorithm: str, scenario: dict
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
        clean_cache(conda_cli)

    def setup():
        if cache_state != "warm":
            # For "cold" and "lukewarm", we want to clean cache before each round of benchmarking
            clean_cache(conda_cli)

        channel = Channel(f"{scenario['channel']}/{scenario['platform']}")
        channel_data = fetch_channels([channel.url()])

        assert len(channel_data) == 2

        subset = RepodataSubset((*channel_data.values(),))

        if cache_state == "lukewarm":
            # Collect pre-fetch packages
            getattr(subset, algorithm)(scenario["prefetch_packages"])

        return (subset,), {}

    def target(subset):
        getattr(subset, algorithm)(scenario["packages"])

    benchmark.pedantic(target, setup=setup, rounds=3)


@pytest.mark.parametrize(
    "scenario",
    TESTING_SCENARIOS,
    ids=[scenario["name"] for scenario in TESTING_SCENARIOS],
)
def test_traversal_algorithms_match(conda_cli, scenario: dict):
    """
    Ensure that all traversal algorithms return the same repodata subset.
    """
    channel = Channel(f"{scenario['channel']}/{scenario['platform']}")

    repodata_algorithm_map = {
        "dijkstra": build_repodata_subset(scenario["packages"], [channel], algorithm="dijkstra"),
        "bfs": build_repodata_subset(scenario["packages"], [channel], algorithm="bfs"),
    }

    for subdir in repodata_algorithm_map["dijkstra"].keys():
        repodatas = []

        for algorithm, repodata_subset in repodata_algorithm_map.items():
            repodatas.append(repodata_subset[subdir].build_repodata())

        assert all(x == y for x, y in zip(repodatas, repodatas[1:]))
