# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""Offline benchmarks for index construction and solving a local channel."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from conda.base.context import context, reset_context
from conda.core.prefix_data import PrefixData
from conda.core.subdir_data import SubdirData
from conda.models.channel import Channel
from conda.models.records import PackageRecord

from conda_libmamba_solver.index import LibMambaIndexHelper
from conda_libmamba_solver.solver import LibMambaSolver

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize("package_count", [100, 1000])
def test_index_installed_records(benchmark: BenchmarkFixture, package_count: int) -> None:
    records = tuple(
        PackageRecord(
            name=f"package-{index}",
            version="1.0",
            build="0",
            build_number=0,
            channel="https://example.invalid/channel",
            subdir="noarch",
            depends=[f"package-{index - 1} >=1.0"] if index else [],
        )
        for index in range(package_count)
    )

    index = benchmark.pedantic(
        LibMambaIndexHelper,
        kwargs={"channels": (), "subdirs": ("noarch",), "installed_records": records},
        rounds=5,
        iterations=1,
        warmup_rounds=1,
    )
    assert index.db.repo_count() == 1
    assert index.db.package_count() == package_count


def test_solve_local_channel(
    benchmark: BenchmarkFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CONDA_PKGS_DIRS", str(tmp_path / "pkgs"))
    # Read the local channel files instead of requiring pre-populated repodata caches.
    monkeypatch.setenv("CONDA_OFFLINE", "false")
    monkeypatch.setenv("CONDA_NOTIFY_OUTDATED_CONDA", "false")
    reset_context()
    channel_path = tmp_path / "channel"
    shutil.copytree(Path(__file__).parent / "data" / "mamba_repo", channel_path)
    platform = context.subdir
    (channel_path / platform).mkdir()
    (channel_path / platform / "repodata.json").write_text(
        json.dumps({"info": {"subdir": platform}, "packages": {}, "packages.conda": {}})
    )
    channel = Channel(str(channel_path))

    def setup():
        # Clear in-memory records and recreate the solver before each measurement.
        # The local channel and filesystem caches remain warm after the first run.
        SubdirData.clear_cached_local_channel_data(exclude_file=False)
        PrefixData._cache_.clear()
        solver = LibMambaSolver(
            prefix=tmp_path / "env",
            channels=[channel],
            subdirs=(platform, "noarch"),
            specs_to_add=["test-package"],
            command="create",
        )
        return (solver,), {}

    solution = benchmark.pedantic(
        LibMambaSolver.solve_final_state,
        setup=setup,
        rounds=5,
        iterations=1,
        warmup_rounds=1,
    )
    assert [(record.name, record.version) for record in solution] == [("test-package", "0.1")]
