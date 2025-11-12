# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
import concurrent.futures
import threading
import urllib.parse
from contextlib import suppress
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING
from unittest.mock import patch

import conda.gateways.repodata
import pytest
import pytest_codspeed
from conda.common.compat import on_win
from conda.core.subdir_data import SubdirData
from conda.models.channel import Channel
from conda.testing.fixtures import CondaCLIFixture
from requests.exceptions import HTTPError

from conda_libmamba_solver import shards_cache, shards_subset
from conda_libmamba_solver.shards import fetch_channels, fetch_shards_index
from conda_libmamba_solver.shards_subset import (
    NodeId,
    RepodataSubset,
    build_repodata_subset,
)
from tests.test_shards import (
    FAKE_SHARD,
    FAKE_SHARD_2,
    ROOT_PACKAGES,
    _timer,
)

if TYPE_CHECKING:
    from conda_libmamba_solver.shards_typing import ShardDict


# avoid underscores in names to parse them easily
TESTING_SCENARIOS = [
    {
        "name": "python",
        "packages": ["python"],
        "prefetch_packages": [],
        "channel": "conda-forge-sharded",
        "platform": "linux-64",
    },
    {
        "name": "data_science_basic",
        "packages": ["numpy", "pandas", "matplotlib"],
        "prefetch_packages": ["python"],
        "channel": "conda-forge-sharded",
        "platform": "linux-64",
    },
    {
        "name": "data_science_ml",
        "packages": ["scikit-learn", "matplotlib"],
        "prefetch_packages": ["python", "numpy"],
        "channel": "conda-forge-sharded",
        "platform": "linux-64",
    },
    {
        "name": "web_development",
        "packages": ["django", "celery"],
        "prefetch_packages": ["python", "requests"],
        "channel": "conda-forge-sharded",
        "platform": "linux-64",
    },
    {
        "name": "scientific_computing",
        "packages": ["scipy", "sympy", "pytorch"],
        "prefetch_packages": ["python", "numpy", "pandas"],
        "channel": "conda-forge-sharded",
        "platform": "linux-64",
    },
    {
        "name": "devops_automation",
        "packages": ["ansible", "pyyaml", "jinja2"],
        "prefetch_packages": ["python"],
        "channel": "conda-forge-sharded",
        "platform": "linux-64",
    },
    {
        "name": "vaex",
        "packages": ["vaex"],
        "prefetch_packages": ["python", "numpy", "pandas"],
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
@pytest.mark.parametrize("algorithm", ("dijkstra", "bfs", "pipelined"))
@pytest.mark.parametrize(
    "scenario",
    TESTING_SCENARIOS,
    ids=[scenario.get("name") for scenario in TESTING_SCENARIOS],
)
@pytest.mark.parametrize("defaults", ["main", "nomain"])
def test_traversal_algorithm_benchmarks(
    conda_cli: CondaCLIFixture,
    benchmark,
    cache_state: str,
    algorithm: str,
    scenario: dict,
    defaults: str,
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

    defaults:
        whether to include the "main" channel
    """
    cache = shards_cache.ShardCache(Path(conda.gateways.repodata.create_cache_dir()))
    if cache_state == "warm":
        # Clean shards cache just once for "warm"; leave index cache intact.
        cache.remove_cache()

    def setup():
        if cache_state != "warm":
            # For "cold" and "lukewarm", we want to clean shards cache before
            # each round of benchmarking
            cache.remove_cache()

        channels = [Channel(f"{scenario['channel']}/{scenario['platform']}")]
        if defaults == "main":
            channels.append(Channel("main"))
        channel_data = fetch_channels(channels)

        assert len(channel_data) in (2, 4), "Expected 2 or 4 channels fetched"

        subset = RepodataSubset((*channel_data.values(),))

        if cache_state == "lukewarm":
            # Collect pre-fetch packages
            getattr(subset, f"shortest_{algorithm}")(scenario["prefetch_packages"])

        return (subset,), {}

    def target(subset):
        with _timer(""):
            getattr(subset, f"shortest_{algorithm}")(scenario["packages"])

    benchmark.pedantic(target, setup=setup, rounds=2)


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
        "pipelined": build_repodata_subset(scenario["packages"], [channel], algorithm="pipelined"),
    }

    for subdir in repodata_algorithm_map["dijkstra"].keys():
        repodatas = []

        for algorithm, repodata_subset in repodata_algorithm_map.items():
            repodatas.append(repodata_subset[subdir].build_repodata())

        assert all(x == y for x, y in zip(repodatas, repodatas[1:]))


# region pipelined


def test_build_repodata_subset_pipelined(prepare_shards_test: None, tmp_path):
    """
    Build repodata subset using a worker threads dependency traversal algorithm.
    """
    # installed, plus what we want to add (twine)
    root_packages = ROOT_PACKAGES[:] + ["vaex"]

    channels = []
    # channels.extend(context.default_channels)
    channels.append(Channel("conda-forge-sharded"))

    with _timer("fetch_channels()"):
        channel_data = fetch_channels(channels)

    with _timer("RepodataSubset.shortest_pipelined()"):
        subset = RepodataSubset((*channel_data.values(),))
        subset.shortest_pipelined(root_packages)
        print(f"{len(subset.nodes)} (channel, package) nodes discovered")

    print("Channels:", ",".join(urllib.parse.urlparse(url).path[1:] for url in channel_data))


def test_shards_cache_thread(
    shard_cache_with_data: tuple[shards_cache.ShardCache, list[shards_cache.AnnotatedRawShard]],
):
    """
    Test sqlite3 retrieval thread.
    """
    cache, fake_shards = shard_cache_with_data
    in_queue: SimpleQueue[list[NodeId] | None] = SimpleQueue()
    shard_out_queue: SimpleQueue[list[tuple[NodeId, ShardDict]]] = SimpleQueue()
    network_out_queue: SimpleQueue[list[NodeId]] = SimpleQueue()

    # this kind of thread can crash, and we don't hear back without our own
    # handling.
    cache_thread = threading.Thread(
        target=shards_subset.cache_fetch_thread,
        args=(in_queue, shard_out_queue, network_out_queue, cache),
        daemon=False,
    )

    fake_nodes = [NodeId(shard.package, channel="", shard_url=shard.url) for shard in fake_shards]

    # several batches, then None "finish thread" sentinel
    in_queue.put(fake_nodes[:1])
    in_queue.put([NodeId("notfound", channel="", shard_url="https://example.com/notfound")])
    in_queue.put(fake_nodes[1:3])
    in_queue.put(
        [
            NodeId("notfound2", channel="", shard_url="https://example.com/notfound2"),
            NodeId("notfound3", channel="", shard_url="https://example.com/notfound3"),
        ]
    )
    in_queue.put(fake_nodes[3:])
    in_queue.put(None)

    cache_thread.start()

    while batch := shard_out_queue.get(timeout=1):
        for node_id, shard in batch:
            assert node_id in fake_nodes
            assert shard == cache.retrieve(node_id.shard_url)

    while notfound := network_out_queue.get(timeout=1):
        for node_id in notfound:
            assert node_id.shard_url.startswith("https://example.com/notfound")

    cache_thread.join(5)


def test_shards_network_thread(http_server_shards, shard_cache_with_data):
    """
    Test network retrieval thread, meant to be chained after the sqlite3 thread
    by having network_in_queue = sqlite3 thread's network_out_queue.
    """
    cache, fake_shards = shard_cache_with_data
    channel = Channel.from_url(f"{http_server_shards}/noarch")
    subdir_data = SubdirData(channel)
    found = fetch_shards_index(subdir_data)
    assert found

    network_in_queue: SimpleQueue[list[NodeId] | None] = SimpleQueue()
    shard_out_queue: SimpleQueue[list[tuple[NodeId, ShardDict]]] = SimpleQueue()

    # this kind of thread can crash, and we don't hear back without our own
    # handling.
    network_thread = threading.Thread(
        target=shards_subset.network_fetch_thread,
        args=(network_in_queue, shard_out_queue, cache, [found]),
        daemon=False,
    )

    node_ids = [NodeId(package, found.url) for package in found.package_names]

    # Only fetch "foo" and "bar" because these are valid shards
    for node_id in node_ids:
        if node_id.package in ("foo", "bar"):
            network_in_queue.put([node_id])

    # Terminate with sentinel
    network_in_queue.put(None)

    network_thread.start()

    with suppress(Empty):
        while batch := shard_out_queue.get(timeout=1):
            for url, shard in batch:
                assert isinstance(shard, dict)

                # Make sure this is either one of the two packages from above ("foo" or "bar")
                assert set(shard.get("packages", {}).keys()).intersection(("foo", "bar"))

    network_thread.join(5)


# endregion


@pytest.mark.parametrize("algorithm", ["bfs", "pipelined"])
def test_build_repodata_subset_error_propagation(http_server_shards, algorithm, mocker, tmp_path):
    """
    Ensure errors encountered during shard fetching are properly propagated.

    This test uses http_server_shards to fetch the initial shards index,
    then mocks the actual shard fetching to simulate network errors.
    """

    # Use http_server_shards to set up the initial channel with shards index
    channel = Channel.from_url(f"{http_server_shards}/noarch")
    root_packages = ["foo"]

    # Override cache dir location for tests; ensures it's empty
    mocker.patch("conda.gateways.repodata.create_cache_dir", return_value=str(tmp_path))

    # Mock batch_retrieve_from_network to raise an error for bfs algorithm
    if algorithm == "bfs":
        with patch(
            "conda_libmamba_solver.shards_subset.batch_retrieve_from_network"
        ) as mock_batch:
            # Simulate a network error when fetching shards
            mock_batch.side_effect = HTTPError("Simulated network error")

            with pytest.raises(HTTPError, match="Simulated network error"):
                build_repodata_subset(root_packages, [channel], algorithm=algorithm)

    # For pipelined algorithm, mock the session.get to raise an error
    elif algorithm == "pipelined":
        # Patch at the module level before threads start
        original_executor = concurrent.futures.ThreadPoolExecutor

        def mock_executor(*args, **kwargs):
            executor = original_executor(*args, **kwargs)
            original_submit = executor.submit

            def mock_submit(fn, *fn_args, **fn_kwargs):
                if fn.__name__ == "fetch":
                    raise HTTPError("Simulated network error during pipelined fetch")
                return original_submit(fn, *fn_args, **fn_kwargs)

            executor.submit = mock_submit
            return executor

        with patch("concurrent.futures.ThreadPoolExecutor", mock_executor):
            # The pipelined algorithm should propagate this error
            with pytest.raises(HTTPError, match="Simulated network error during pipelined fetch"):
                build_repodata_subset(root_packages, [channel], algorithm=algorithm)


@pytest.mark.parametrize("algorithm", ["bfs", "pipelined"])
def test_build_repodata_subset_package_not_found(http_server_shards, algorithm, tmp_path, mocker):
    """
    Ensure packages that cannot be found result in empty repodata.

    This test uses http_server_shards to fetch the initial shards index,
    and then tests the code to make sure an empty repodata is produced at the end.
    """

    # Use http_server_shards to set up the initial channel with shards index
    channel = Channel.from_url(f"{http_server_shards}/noarch")
    root_packages = ["404-package-not-found"]

    # Override cache dir location for tests; ensures it's empty
    mocker.patch("conda.gateways.repodata.create_cache_dir", return_value=str(tmp_path))

    channel_data = build_repodata_subset(root_packages, [channel], algorithm=algorithm)

    for shardlike in channel_data.values():
        assert not shardlike.build_repodata().get("packages")


@pytest.mark.parametrize("algorithm", ["bfs", "pipelined"])
def test_build_repodata_subset_local_server(http_server_shards, algorithm, mocker, tmp_path):
    """
    Ensure we can fetch and build a valid repodata subset from our mock local server.
    """
    channel = Channel.from_url(f"{http_server_shards}/noarch")
    root_packages = ["foo"]
    expected_repodata = {**FAKE_SHARD["packages"], **FAKE_SHARD_2["packages"]}

    # Override cache dir location for tests; ensures it's empty
    mocker.patch("conda.gateways.repodata.create_cache_dir", return_value=str(tmp_path))

    channel_data = build_repodata_subset(root_packages, [channel], algorithm=algorithm)

    for shardlike in channel_data.values():
        # expanded in fetch_channels() "channel.urls(True, context.subdirs)"
        if "/noarch/" not in shardlike.url:
            continue
        assert shardlike.build_repodata().get("packages") == expected_repodata
