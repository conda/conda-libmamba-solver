# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from conda.base.context import context, reset_context
from conda.common.compat import on_win
from conda.core.subdir_data import SubdirData
from conda.gateways.logging import initialize_logging
from conda.models.channel import Channel

from conda_libmamba_solver.index import LibMambaIndexHelper, _is_sharded_repodata_enabled
from conda_libmamba_solver.shards_subset import build_repodata_subset
from conda_libmamba_solver.state import SolverInputState
from tests.test_shards import _timer

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable

    from conda.gateways.repodata import RepodataState
    from pytest_benchmark.plugin import BenchmarkFixture

    from conda_libmamba_solver.shards import ShardBase


initialize_logging()
DATA = Path(__file__).parent / "data"


def test_given_channels(monkeypatch: pytest.MonkeyPatch, tmp_path: os.PathLike):
    monkeypatch.setenv("CONDA_PKGS_DIRS", str(tmp_path))
    reset_context()
    libmamba_index = LibMambaIndexHelper.from_platform_aware_channel(
        channel=Channel("conda-test/noarch")
    )
    assert libmamba_index.db.repo_count() == 1

    conda_index = SubdirData(Channel("conda-test/noarch"))
    conda_index.load()

    assert libmamba_index.db.package_count() == len(tuple(conda_index.iter_records()))


@pytest.mark.parametrize(
    "only_tar_bz2",
    (
        pytest.param("1", id="CONDA_USE_ONLY_TAR_BZ2=true"),
        pytest.param("", id="CONDA_USE_ONLY_TAR_BZ2=false"),
    ),
)
def test_defaults_use_only_tar_bz2(monkeypatch: pytest.MonkeyPatch, only_tar_bz2: bool):
    """
    Defaults is particular in the sense that it offers both .tar.bz2 and .conda for LOTS
    of packages. SubdirData ignores .tar.bz2 entries if they have a .conda counterpart.
    So if we count all the packages in each implementation, libmamba's has way more.
    To remain accurate, we test this with `use_only_tar_bz2`:
        - When true, we only count .tar.bz2
        - When false, we only count .conda
    """
    monkeypatch.setenv("CONDA_USE_ONLY_TAR_BZ2", only_tar_bz2)
    reset_context()
    libmamba_index = LibMambaIndexHelper(
        channels=[Channel("defaults")],
        subdirs=("noarch",),
        installed_records=(),  # do not load installed
        pkgs_dirs=(),  # do not load local cache as a channel
    )
    n_repos = 3 if on_win else 2
    assert len(libmamba_index.repos) == n_repos

    libmamba_dot_conda_total = libmamba_index.n_packages(
        filter_=lambda pkg: pkg.package_url.endswith(".conda")
    )
    libmamba_tar_bz2_total = libmamba_index.n_packages(
        filter_=lambda pkg: pkg.package_url.endswith(".tar.bz2")
    )

    conda_dot_conda_total = 0
    conda_tar_bz2_total = 0
    for channel_url in Channel("defaults/noarch").urls(subdirs=("noarch",)):
        conda_index = SubdirData(Channel(channel_url))
        conda_index.load()
        for pkg in conda_index.iter_records():
            if pkg["url"].endswith(".conda"):
                conda_dot_conda_total += 1
            elif pkg["url"].endswith(".tar.bz2"):
                conda_tar_bz2_total += 1
            else:
                raise RuntimeError(f"Unrecognized package URL: {pkg['url']}")

    if only_tar_bz2:
        assert conda_tar_bz2_total == libmamba_tar_bz2_total
        assert libmamba_dot_conda_total == conda_dot_conda_total == 0
    else:
        assert conda_dot_conda_total == libmamba_dot_conda_total
        assert conda_tar_bz2_total == libmamba_tar_bz2_total


def test_reload_channels(tmp_path: Path):
    (tmp_path / "noarch").mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / "mamba_repo" / "noarch" / "repodata.json", tmp_path / "noarch")
    initial_repodata = (tmp_path / "noarch" / "repodata.json").read_text()
    index = LibMambaIndexHelper(channels=[Channel(str(tmp_path))])
    initial_count = index.n_packages()
    SubdirData._cache_.clear()

    data = json.loads(initial_repodata)
    package = data["packages"]["test-package-0.1-0.tar.bz2"]
    data["packages"]["test-package-copy-0.1-0.tar.bz2"] = {**package, "name": "test-package-copy"}
    modified_repodata = json.dumps(data)
    (tmp_path / "noarch" / "repodata.json").write_text(modified_repodata)

    assert initial_repodata != modified_repodata
    # TODO: Remove this sleep after addressing
    # https://github.com/conda/conda/issues/13783
    time.sleep(1)
    index.reload_channel(Channel(str(tmp_path)))
    assert index.n_packages() == initial_count + 1


@pytest.mark.parametrize(
    "load_type,requested",
    [
        ("shard", ("python",)),
        ("shard", ("django", "celery")),
        ("shard", ("vaex",)),
        ("repodata", ("vaex",)),
        ("main", ()),
    ],
    ids=["shard-small", "shard-medium", "shard-large", "noshard", "main"],
)
def test_load_channel_repo_info_shards(
    load_type: str,
    requested: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    benchmark: BenchmarkFixture,
):
    """
    Benchmark shards/not-shards under different dependency tree sizes.

    TODO: This test should eventually switch to just using conda-forge when that channel
          supports shards and not the `conda-forge-sharded` channel.
    """
    load_channel = "defaults" if load_type == "main" else "conda-forge-sharded"

    monkeypatch.setattr(context.plugins, "use_sharded_repodata", load_type == "shard")
    assert _is_sharded_repodata_enabled() == (load_type == "shard")

    in_state = SolverInputState(str(tmp_path / "env"), requested=requested)

    def index():
        return LibMambaIndexHelper(
            # this is expanded to noarch, linux-64 for shards.
            channels=[Channel(f"{load_channel}/linux-64")],
            subdirs=(
                "noarch",
                "linux-64",
            ),
            installed_records=(),  # do not load installed
            pkgs_dirs=(),  # do not load local cache as a channel
            in_state=in_state,
        )
        pass

    # this fails for some reason if run twice
    # cuda finder function crashes when run twice, sys.exit() called?
    index_helper = benchmark.pedantic(index, rounds=1)

    assert len(index_helper.repos) > 0


@pytest.mark.parametrize(
    "load_type,requested",
    [
        ("shard", ("python",)),
        ("shard", ("django", "celery")),
        ("shard", ("vaex",)),
        ("repodata", ("vaex",)),
        ("main", ()),
        ("shard-unavailable", ("vaex",)),
    ],
    ids=[
        "shard-small",
        "shard-medium",
        "shard-large",
        "noshard",
        "main",
        "shard-unavailable",
    ],
)
def test_load_channel_repo_info_shards_parse_only(
    load_type: str,
    requested: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    benchmark: BenchmarkFixture,
):
    """
    Benchmark shards/not-shards under different dependency tree sizes.

    Only measure parse time, not "download + parse" time.

    TODO: This test should eventually switch to just using conda-forge when that channel
          supports shards and not the `conda-forge-sharded` channel.
    """
    # defaults is more than one channel but I don't think that matters here.
    load_channel = {"main": "main", "shard-unavailable": "conda-forge"}.get(
        load_type, "conda-forge-sharded"
    )

    monkeypatch.setattr(
        context.plugins,
        "use_sharded_repodata",
        load_type in ("shard", "shard-unavailable"),
    )
    assert _is_sharded_repodata_enabled() == (load_type in ("shard", "shard-unavailable"))

    in_state = SolverInputState(str(tmp_path / "env"), requested=requested)

    urls_to_channel_ = {
        f"https://conda.anaconda.org/{load_channel}/noarch": Channel(f"{load_channel}/noarch"),
        f"https://conda.anaconda.org/{load_channel}/linux-64": Channel(f"{load_channel}/linux-64"),
    }

    repodata_jsons = {}
    channel_data = {}
    root_packages_ = (*in_state.installed.keys(), *in_state.requested)
    if load_type in ("shard", "shard-unavailable"):  # shards
        with _timer("subset monolithic repodata.json"):
            channel_data = build_repodata_subset(root_packages_, urls_to_channel_)

    else:  # no shards
        # this is an inefficient way to fetch repodata.json in the format
        # expected by LibMambaIndexHelper since it loads it into the solver.
        helper = LibMambaIndexHelper(
            # this is expanded to noarch, linux-64 for shards.
            channels=[Channel(f"{load_channel}/linux-64")],
            subdirs=(
                "noarch",
                "linux-64",
            ),
            installed_records=(),  # do not load installed
            pkgs_dirs=(),  # do not load local cache as a channel
            in_state=in_state,
        )
        repodata_jsons = helper._fetch_repodata_jsons(urls_to_channel_.keys())

    class LibMambaIndexHelperParseOnly(LibMambaIndexHelper):
        def _build_repodata_subset(
            self, root_packages: tuple[str, ...], urls_to_channel: dict[str, Channel]
        ) -> dict[str, ShardBase]:
            assert root_packages == root_packages_
            assert urls_to_channel == urls_to_channel
            return channel_data

        def _fetch_repodata_jsons(
            self, urls: Iterable[str]
        ) -> dict[str, tuple[str, RepodataState]]:
            assert sorted(urls) == sorted(urls_to_channel_.keys())
            return repodata_jsons

    def index():
        return LibMambaIndexHelperParseOnly(
            # this is expanded to noarch, linux-64 for shards.
            channels=[Channel(f"{load_channel}/linux-64")],
            subdirs=(
                "noarch",
                "linux-64",
            ),
            installed_records=(),  # do not load installed
            pkgs_dirs=(),  # do not load local cache as a channel
            in_state=in_state,
        )
        pass

    # this fails for some reason if run twice
    # cuda finder function crashes when rounds > 1, sys.exit() called?
    index_helper = benchmark.pedantic(index, rounds=1)

    assert len(index_helper.repos) > 0
