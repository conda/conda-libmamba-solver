# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import asyncio
import concurrent.futures
import random
import threading
import time
import typing
import urllib.parse
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
from unittest.mock import patch

import conda.gateways.repodata
import pytest
import pytest_codspeed
import requests
from conda.common.compat import on_win
from conda.core.subdir_data import SubdirData
from conda.models.channel import Channel
from requests.exceptions import HTTPError

from conda_libmamba_solver import shards_cache, shards_subset, shards_subset_http2
from conda_libmamba_solver.shards import ShardLike, fetch_channels, fetch_shards_index
from conda_libmamba_solver.shards_subset import (
    NodeId,
    RepodataSubset,
    build_repodata_subset,
    combine_batches_until_none,
    exception_to_queue,
)
from tests.test_shards import (
    FAKE_SHARD,
    FAKE_SHARD_2,
    ROOT_PACKAGES,
    _timer,
)

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    from conda.testing.fixtures import CondaCLIFixture
    from pytest_benchmark.plugin import BenchmarkFixture

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
@pytest.mark.parametrize("cache_state", ("cold", "warm"))
@pytest.mark.parametrize("algorithm", ("bfs", "pipelined", "httpx"))
@pytest.mark.parametrize(
    "scenario",
    TESTING_SCENARIOS,
    ids=[scenario.get("name") for scenario in TESTING_SCENARIOS],
)
def test_traversal_algorithm_benchmarks(
    benchmark: BenchmarkFixture,
    cache_state: str,
    algorithm: str,
    scenario: dict,
):
    """
    Benchmark multiple traversal algorithms for retrieving repodata shards with
    a variety of parameter states (described below).

    cache_state:
        Either "cold" or "warm" representing shards available or not available in
        SQLite, respectively.

    algorithm:
        Method used to fetch shards

    scenario:
        List of packages to use to create an environment
    """
    cache = shards_cache.ShardCache(Path(conda.gateways.repodata.create_cache_dir()))
    if cache_state == "warm":
        # Clean shards cache just once for "warm"; leave index cache intact.
        cache.remove_cache()

    def setup():
        if cache_state != "warm":
            # For "cold", we want to clean shards cache before each round of benchmarking
            cache.remove_cache()

        channels = [Channel(f"{scenario['channel']}/{scenario['platform']}")]
        channel_data = fetch_channels(channels)

        assert len(channel_data) in (2, 4), "Expected 2 or 4 channels fetched"

        subset = RepodataSubset((*channel_data.values(),))

        return (subset,), {}

    def target(subset):
        with _timer(""):
            getattr(subset, f"reachable_{algorithm}")(scenario["packages"])

    warmup_rounds = 1 if cache_state == "warm" else 0

    benchmark.pedantic(target, setup=setup, rounds=1, warmup_rounds=warmup_rounds)


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
        "bfs": build_repodata_subset(scenario["packages"], [channel], algorithm="bfs"),
        "pipelined": build_repodata_subset(scenario["packages"], [channel], algorithm="pipelined"),
    }

    for subdir in repodata_algorithm_map["bfs"].keys():
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

    with _timer("RepodataSubset.reachable_pipelined()"):
        subset = RepodataSubset((*channel_data.values(),))
        subset.reachable_pipelined(root_packages)
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

    invalid_shardlike = ShardLike(
        {},  # type: ignore
        url="file:///non-network/shard/url",
    )

    network_in_queue: SimpleQueue[list[NodeId] | None] = SimpleQueue()
    shard_out_queue: SimpleQueue[list[tuple[NodeId, ShardDict]]] = SimpleQueue()

    # this kind of thread can crash, and we don't hear back without our own
    # handling.
    network_thread = threading.Thread(
        target=shards_subset.network_fetch_thread,
        args=(network_in_queue, shard_out_queue, cache, [found, invalid_shardlike]),
        daemon=False,
    )

    node_ids = [NodeId(package, found.url) for package in found.package_names]

    # Only fetch "foo" and "bar" because these are valid shards
    for node_id in node_ids:
        if node_id.package in ("foo", "bar"):
            network_in_queue.put([node_id])

    network_thread.start()

    with suppress(Empty):
        while batch := shard_out_queue.get(timeout=1):
            for url, shard in batch:
                assert isinstance(shard, dict)

                # Make sure this is either one of the two packages from above ("foo" or "bar")
                assert set(shard.get("packages", {}).keys()).intersection(("foo", "bar"))

    # Worker produces TypeError if non-network NodeId is sent and one of the
    # shardlikes has its url. (If no shardlike has NodeId's url, it produces
    # KeyError).
    network_in_queue.put([NodeId("nope", invalid_shardlike.url)])
    assert isinstance(shard_out_queue.get(timeout=1), TypeError)

    # Terminate with sentinel
    network_in_queue.put(None)

    network_thread.join(5)


def test_shards_network_thread_httpx(http_server_shards, shard_cache_with_data):
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
        target=shards_subset_http2.network_fetch_thread_httpx,
        args=(network_in_queue, shard_out_queue, cache, [found]),
        daemon=False,
    )

    node_ids = [NodeId(package, found.url) for package in found.package_names]

    # several batches, then None "finish thread" sentinel
    network_in_queue.put([node_ids[0]])
    network_in_queue.put(node_ids[1:])
    network_in_queue.put(None)

    network_thread.start()

    while batch := shard_out_queue.get(timeout=1):
        for url, shard in batch:
            print(url, len(shard), "bytes")

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
        original_executor = shards_subset.ThreadPoolExecutor

        def mock_executor(*args, **kwargs):
            executor = original_executor(*args, **kwargs)
            original_submit = executor.submit

            def mock_submit(fn, *fn_args, **fn_kwargs):
                if fn.__name__ == "fetch":
                    raise HTTPError("Simulated network error during pipelined fetch")
                return original_submit(fn, *fn_args, **fn_kwargs)

            executor.submit = mock_submit
            return executor

        with patch("conda_libmamba_solver.shards_subset.ThreadPoolExecutor", mock_executor):
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


def test_pipelined_with_slow_queue_operations(http_server_shards, mocker, tmp_path):
    """
    Test that simulates slow queue operations which can trigger race conditions
    where the main thread might timeout waiting for results.
    """
    channel = Channel.from_url(f"{http_server_shards}/noarch")
    root_packages = ["foo"]

    # Override cache dir location for tests
    mocker.patch("conda.gateways.repodata.create_cache_dir", return_value=str(tmp_path))

    # Create a custom queue that adds delays
    class SlowQueue(SimpleQueue):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.put_count = 0

        def put(self, item, *args, **kwargs):
            self.put_count += 1
            # Add delay every few puts to simulate slow operations
            if self.put_count % 3 == 0:
                time.sleep(0.05)
            return super().put(item, *args, **kwargs)

    def slow_simple_queue_factory(*args, **kwargs):
        # Only slow down shard_out_queue (not all queues)
        return SlowQueue(*args, **kwargs)

    mocker.patch("conda_libmamba_solver.shards_subset.SimpleQueue", slow_simple_queue_factory)

    # This should complete despite slow queue operations
    channel_data = build_repodata_subset(root_packages, [channel], algorithm="pipelined")

    # Verify results
    found_packages = False
    for shardlike in channel_data.values():
        if "/noarch/" in shardlike.url and shardlike.build_repodata().get("packages"):
            found_packages = True
    assert found_packages


def test_pipelined_shutdown_race_condition(http_server_shards, mocker, tmp_path):
    """
    Test the specific race condition where the main thread checks pending/in_flight
    and finds them empty, but worker threads are still processing items.
    """
    channel = Channel.from_url(f"{http_server_shards}/noarch")
    root_packages = ["foo"]

    mocker.patch("conda.gateways.repodata.create_cache_dir", return_value=str(tmp_path))

    # Track when drain_pending is called
    original_drain_pending = RepodataSubset.drain_pending
    drain_count = {"count": 0}

    def tracked_drain_pending(self, pending, shardlikes_by_url):
        drain_count["count"] += 1
        result = original_drain_pending(self, pending, shardlikes_by_url)
        # Add delay after drain to increase chance of race condition
        if drain_count["count"] > 1:
            time.sleep(0.1)
        return result

    mocker.patch.object(RepodataSubset, "drain_pending", tracked_drain_pending)

    # Run multiple times to increase chance of hitting race condition
    for _ in range(10):
        channel_data = build_repodata_subset(root_packages, [channel], algorithm="pipelined")

        # Verify we got valid results
        found_packages = False
        for shardlike in channel_data.values():
            if "/noarch/" in shardlike.url and shardlike.build_repodata().get("packages"):
                found_packages = True
        assert found_packages


def test_pipelined_timeout(http_server_shards, monkeypatch):
    """
    Test that pipelined times out if a URL is never fetched.
    """

    channel = Channel.from_url(f"{http_server_shards}/noarch")
    root_packages = ["foo"]

    shardlikes = fetch_channels([channel])

    # a slow and ineffective get()
    monkeypatch.setattr(
        "conda.gateways.connection.session.CondaSession.get",
        lambda *args, **kwargs: time.sleep(3),
    )

    # faster failure
    monkeypatch.setattr("conda_libmamba_solver.shards_subset.REACHABLE_PIPELINED_MAX_TIMEOUTS", 1)
    monkeypatch.setattr("conda_libmamba_solver.shards_subset.THREAD_WAIT_TIMEOUT", 0)

    subset = RepodataSubset(shardlikes.values())
    with pytest.raises(TimeoutError, match="shard_out_queue"):
        subset.reachable_pipelined(root_packages)


def test_combine_batches_blocking_scenario():
    """
    Test the scenario where combine_batches_until_none would block indefinitely
    without the timeout fix.

    This simulates the case where:
    1. Producer sends a few items
    2. Producer crashes or stops sending before sending None
    3. Consumer blocks forever waiting for more items
    """
    test_queue: SimpleQueue[Sequence[NodeId] | None] = SimpleQueue()

    # Put some items in the queue
    test_queue.put([NodeId("package1", "channel1")])
    test_queue.put([NodeId("package2", "channel2")])

    # Simulate producer failure - don't send None sentinel
    # Without timeout fix, this would block forever

    received = []
    timeout_occurred = False

    def consumer():
        nonlocal timeout_occurred
        try:
            for batch in combine_batches_until_none(test_queue):
                received.extend(batch)
                # After processing existing items, iterator should timeout
                # rather than block forever
        except Exception:
            timeout_occurred = True

    consumer_thread = threading.Thread(target=consumer, daemon=True)
    consumer_thread.start()

    # Wait for consumer to process existing items
    consumer_thread.join(timeout=2)

    # With the timeout fix, the thread should still be alive (waiting)
    # but not blocking indefinitely - it will timeout periodically
    # Let's verify it processed the items we sent
    assert len(received) == 2
    assert any(node.package == "package1" for node in received)


@pytest.mark.integration
def test_pipelined_extreme_race_conditions(http_server_shards, mocker, tmp_path):
    """
    Extremely aggressive test that introduces chaos to force race conditions.

    This test:
    - Adds random delays at multiple points
    - Runs many iterations
    - Uses smaller timeouts
    - Simulates thread scheduling variability
    """
    channel = Channel("conda-forge-sharded/linux-64")
    root_packages = ["python", "vaex"]

    mocker.patch("conda.gateways.repodata.create_cache_dir", return_value=str(tmp_path))

    # Create a chaotic queue class that adds random delays
    class ChaoticQueue(SimpleQueue):
        def get(self, block=True, timeout=None):
            # Random delay before get
            if random.random() < 0.3:  # 30% chance
                time.sleep(random.uniform(0.001, 0.02))
            return super().get(block=block, timeout=timeout)

        def put(self, item, block=True, timeout=None):
            # Random delay before put
            if random.random() < 0.3:  # 30% chance
                time.sleep(random.uniform(0.001, 0.02))
            return super().put(item, block=block, timeout=timeout)

    # Patch at module level
    mocker.patch("conda_libmamba_solver.shards_subset.SimpleQueue", ChaoticQueue)

    # Run multiple iterations to increase chance of hitting race condition
    failures = []
    for iteration in range(20):
        try:
            channel_data = build_repodata_subset(root_packages, [channel], algorithm="pipelined")

            # Verify we got results
            found = any(
                "/noarch/" in s.url and s.build_repodata().get("packages")
                for s in channel_data.values()
            )
            assert found, f"Iteration {iteration}: No packages found"
        except Exception as e:
            failures.append((iteration, str(e)))

    # With our fixes, there should be no failures
    assert not failures, f"Failed on iterations: {failures}"


@pytest.mark.parametrize("num_threads", [1, 2, 5])
def test_pipelined_concurrent_stress(http_server_shards, mocker, tmp_path, num_threads):
    """
    Run pipelined algorithm from multiple threads concurrently.
    This can expose race conditions in shared state or thread coordination.
    """
    channel = Channel.from_url(f"{http_server_shards}/noarch")
    root_packages = ["foo"]

    mocker.patch("conda.gateways.repodata.create_cache_dir", return_value=str(tmp_path))

    errors = []

    def run_subset():
        try:
            channel_data = build_repodata_subset(root_packages, [channel], algorithm="pipelined")
            # Verify results
            for shardlike in channel_data.values():
                if "/noarch/" in shardlike.url:
                    packages = shardlike.build_repodata().get("packages", {})
                    if packages:
                        return True
            return False
        except Exception as e:
            errors.append(e)
            raise

    # Run multiple instances concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(run_subset) for _ in range(num_threads)]
        results = [f.result(timeout=30) for f in futures]

    assert not errors, f"Errors occurred: {errors}"
    assert all(results), "Some runs didn't find packages"


def test_worker_thread_exception_propagation():
    """
    Test that exceptions in worker threads are properly propagated to main thread.
    Without proper exception handling, the main thread could timeout waiting for
    results that will never arrive.
    """
    in_queue = SimpleQueue()
    out_queue = SimpleQueue()

    @exception_to_queue
    def failing_worker(in_q, out_q):
        # Process one item successfully
        item = in_q.get()
        out_q.put(f"processed: {item}")

        # Then raise an exception
        raise ValueError("Simulated worker failure")

    # Put test data
    in_queue.put("test_item")

    # Run worker in thread
    worker = threading.Thread(target=failing_worker, args=(in_queue, out_queue), daemon=True)
    worker.start()

    # Should get the successful result first
    result = out_queue.get(timeout=1)
    assert result == "processed: test_item"

    # Should get the exception propagated
    exception = out_queue.get(timeout=1)
    assert isinstance(exception, ValueError)
    assert "Simulated worker failure" in str(exception)

    # Worker should also send None to in_queue to signal termination
    sentinel = in_queue.get(timeout=1)
    assert sentinel is None


def test_shutdown_with_pending_work(http_server_shards, mocker, tmp_path):
    """
    Test the race condition where main thread initiates shutdown while
    worker threads still have work in their queues.
    """
    channel = Channel.from_url(f"{http_server_shards}/noarch")
    root_packages = ["foo"]

    mocker.patch("conda.gateways.repodata.create_cache_dir", return_value=str(tmp_path))

    # Track shutdown events
    shutdown_events = []

    class TrackShutdownQueue(SimpleQueue):
        def put(self, item, *args, **kwargs):
            if item is None:
                shutdown_events.append(
                    {
                        "time": time.time(),
                        "thread": threading.current_thread().name,
                    }
                )
            return super().put(item, *args, **kwargs)

    # Patch at module level
    mocker.patch("conda_libmamba_solver.shards_subset.SimpleQueue", TrackShutdownQueue)

    # Run the algorithm
    channel_data = build_repodata_subset(root_packages, [channel], algorithm="pipelined")

    # Verify we got results
    found = any(
        "/noarch/" in s.url and s.build_repodata().get("packages") for s in channel_data.values()
    )
    assert found

    # Verify shutdown was initiated (None was sent to queues)
    assert len(shutdown_events) > 0, "Shutdown was never initiated"


def test_repodata_subset_misc():
    """
    Test utility functions on RepodataSubset.
    """
    assert tuple(
        RepodataSubset.has_strategy(strategy) for strategy in ("bfs", "pipelined", "squirrel")
    ) == (True, True, False)


@dataclass
class ShardFetchResult:
    node_id: NodeId
    shard: ShardDict
    size: int


class NetworkSimulator:
    """
    Simulate a network with configurable parallelism and bandwidth, for
    RepodataSubset.reachable_pipelined.

    Not sure this does a great job of simulating http/1 vs. http/2 "10 versus
    much more than 10 simultaneous requests".
    """

    def __init__(
        self,
        connections: int,
        bandwidth_mbps: float,
        latency_ms: int,
        cache: shards_cache.ShardCache,
        index_delay_bytes=0,  # wait for this many bytes before processing nodes
    ):
        self.connections = connections
        self.bandwidth_mbps = bandwidth_mbps
        self.latency_ms = latency_ms
        self.index_delay_bytes = index_delay_bytes
        self.bytes_transferred = 0

        self.in_queue: SimpleQueue[Sequence[NodeId] | None] = SimpleQueue()
        self.async_queue = asyncio.Queue()
        self.out_queue: SimpleQueue[list[tuple[NodeId, ShardDict] | Exception] | None] = (
            SimpleQueue()
        )

        self.cache = cache

    def __str__(self):
        return (
            f"NetworkSimulator(connections={self.connections}, "
            f"bandwidth_mbps={self.bandwidth_mbps}, "
            f"latency_ms={self.latency_ms}, "
            f"index_delay_bytes={self.index_delay_bytes}), "
            f"cache={self.cache.base}"
        )

    def transfer_time(self, byte_count: int):
        """
        Time to transfer byte_count bytes at configured bandwidth.
        """
        return (byte_count * 8) / (self.bandwidth_mbps * 1_000_000)

    async def delay_shards(self):
        """
        Delay shards based on connections, bandwidth, and latency.
        """
        cache = self.cache.copy()

        async def get_work() -> typing.AsyncGenerator[Sequence[NodeId], None]:
            with concurrent.futures.ThreadPoolExecutor() as thread_pool:
                while True:
                    batch = await asyncio.get_running_loop().run_in_executor(
                        thread_pool, self.in_queue.get
                    )
                    if batch is None:
                        break
                    yield batch

        connection_pool = asyncio.Semaphore(self.connections)
        bandwidth_pool = asyncio.Semaphore(1)

        latency_error = 0.0

        async def latency_task(item: ShardFetchResult):
            nonlocal latency_error
            async with connection_pool:
                # print("Get", Channel(item.node_id.channel), item.node_id.package)
                latency_start = time.monotonic()
                await asyncio.sleep(self.latency_ms / 1000.0)
                latency_end = time.monotonic()
                latency_error += (self.latency_ms / 1000.0) - (latency_end - latency_start)
            asyncio.create_task(bandwidth_task(item))

        transfer_error = 0.0

        async def bandwidth_task(item: ShardFetchResult):
            nonlocal transfer_error
            async with bandwidth_pool:
                # one task at a time waits proportional to size / bandwidth
                transfer_time = self.transfer_time(item.size)
                transfer_start = time.monotonic()
                await asyncio.sleep(transfer_time)
                transfer_end = time.monotonic()
                transfer_error += transfer_time - (transfer_end - transfer_start)
                self.bytes_transferred += item.size
                self.out_queue.put([(item.node_id, item.shard)])

        async def process_nodes():
            total_nodes = 0
            wrap_nodes = 0
            async for node_ids in get_work():
                # latency needs to be added after we go into the request queue. ignore the sqlite3 latency.
                # we could afford to cache these in RAM.
                cached = cache.retrieve_multiple_size([node_id.shard_url for node_id in node_ids])
                found: list[tuple[NodeId, tuple[ShardDict, int]]] = []
                not_found: list[NodeId] = []
                wrap_nodes += len(node_ids)
                total_nodes += len(node_ids)
                if wrap_nodes > 100:
                    print(f"Simulated {total_nodes} nodes")
                    wrap_nodes %= 100
                for node_id in node_ids:
                    if shard_info := cached.get(node_id.shard_url):
                        found.append((node_id, shard_info))
                    else:
                        not_found.append(node_id)
                        print(f"Simulator wants full cache, but missing {node_id.shard_url}")
                        self.out_queue.put(None)
                        return

                for node_id, (shard, size) in found:
                    asyncio.create_task(
                        latency_task(
                            ShardFetchResult(
                                node_id=node_id,
                                shard=shard,
                                size=size,
                            )
                        )
                    )

        print("Processing nodes")
        await process_nodes()
        print(f"Bandwidth error {transfer_error:0.3f}")
        print(f"Latency error {latency_error:0.3f}")

    def index_transfer_delay(self):
        """
        Call before RepodataSubset.build_repodata_subset to simulate initial index transfer delay.
        """
        index_transfer = self.transfer_time(self.index_delay_bytes)
        print(
            f"Wait to transfer repodata_shards.msgpack.zst ({self.index_delay_bytes} bytes, {index_transfer:.2f} s)"
        )
        time.sleep(index_transfer)

    def run(self):
        asyncio.run(self.delay_shards())
        print("NetworkSimulator end")


@pytest.fixture(scope="session")
def repodata_index_transfer_size():
    """
    Determine the transfer size of the shards index for use in network simulator.
    """
    channel = Channel("conda-forge-sharded/linux-64")
    urls = channel.urls()
    result = {}
    session = requests.Session()
    for filename in "repodata_shards.msgpack.zst", "repodata.json.zst":
        total_size = 0
        for url in urls:
            total_size += int(session.head(f"{url}/{filename}").headers.get("Content-Length", 0))
        result[filename] = total_size
    return result


# Different latency/bandwidth scenarios, loosely based on Firefox debug console.
NETWORK_SCENARIOS = {
    "4MBPS": {"bandwidth_mbps": 4, "latency_ms": 100},  # want a high latency
    "30MBPS": {"bandwidth_mbps": 30, "latency_ms": 30},  # medium
    "HALFGIG": {
        "bandwidth_mbps": 500,
        "latency_ms": 20,
    },  # and low (ping time to cloudflare) option
}


@pytest.mark.parametrize("scenario_name", NETWORK_SCENARIOS.keys())
@pytest.mark.parametrize("connections", [10, 20, 100])
@pytest.mark.parametrize(
    "packages", [["python"], ["django", "celery"], ["vaex"]], ids=["python", "django", "vaex"]
)
def test_repodata_subset_network_simulator(
    benchmark, repodata_index_transfer_size, scenario_name, connections, packages
):
    """
    Test RepodataSubset.reachable_pipelined with a simulated network. Real-world
    results may vary.
    """

    # May remove debugging "is thread alive" from pipelined_main_thread later.
    class FakeThread:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            pass

    # Set up test channel and root packages
    channel = Channel("conda-forge-sharded/linux-64")
    channel_data = fetch_channels([channel])

    # populate cache
    build_repodata_subset(packages, [channel], algorithm="httpx")

    def build_simulator():
        # set up network simulator
        simulator = NetworkSimulator(
            connections=connections,
            cache=list(channel_data.values())[0].shards_cache,  # type: ignore[arg-type]
            index_delay_bytes=repodata_index_transfer_size["repodata_shards.msgpack.zst"],
            **NETWORK_SCENARIOS[scenario_name],
        )

        simulator_thread = threading.Thread(target=simulator.run)
        simulator_thread.start()

        return simulator

    @benchmark
    def simulate():
        simulator = build_simulator()
        print(simulator)
        monolithic_transfer = simulator.transfer_time(
            repodata_index_transfer_size["repodata.json.zst"]
        )
        with _timer(f"Non-shards transfer {monolithic_transfer:.2f}s vs simulated traversal"):
            simulator.index_transfer_delay()
            for value in channel_data.values():
                value.reset()  # avoid already-loaded shortcut
            subset = RepodataSubset(channel_data.values())
            subset.pipelined_main_thread(
                packages, simulator.in_queue, simulator.out_queue, FakeThread(), FakeThread()
            )
            shards_mib = (simulator.bytes_transferred) / (2**20)
            shards_index_mib = +repodata_index_transfer_size["repodata_shards.msgpack.zst"] / (
                2**20
            )
            repodata_mib = repodata_index_transfer_size["repodata.json.zst"] / (2**20)
            print(
                f"Shards {shards_index_mib:0.2f}MiB+{shards_mib:0.2f}MiB vs repodata.json.zst {repodata_mib:0.2f}MiB for {packages}, {len(subset.nodes)} nodes"
            )

    # Missing here is the "load into LibMambaIndexHelper" step. When bandwidth
    # is high the time spent parsing repodata vs shards can dominate.
