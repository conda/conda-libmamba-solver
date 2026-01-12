# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from conda.base.context import context, reset_context
from conda.common.compat import on_win
from conda.core.subdir_data import SubdirData
from conda.gateways.logging import initialize_logging
from conda.models.channel import Channel

from conda_libmamba_solver.index import LibMambaIndexHelper, _is_sharded_repodata_enabled
from conda_libmamba_solver.state import SolverInputState

from .test_shards import CONDA_FORGE_WITH_SHARDS

if TYPE_CHECKING:
    import os

    from conda.testing.fixtures import CondaCLIFixture
    from pytest_benchmark.plugin import BenchmarkFixture


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
    conda_cli: CondaCLIFixture,
    monkeypatch: pytest.MonkeyPatch,
    benchmark: BenchmarkFixture,
):
    """
    Benchmark shards/not-shards under different dependency tree sizes.
    """
    load_channel = "defaults" if load_type == "main" else CONDA_FORGE_WITH_SHARDS

    monkeypatch.setattr(context.plugins, "use_sharded_repodata", load_type == "shard")
    assert _is_sharded_repodata_enabled() == (load_type == "shard")

    in_state = SolverInputState(str(tmp_path / "env"), requested=requested)

    def index():
        return LibMambaIndexHelper(
            # this is expanded to noarch, linux-64 for shards.
            channels=[Channel(f"{load_channel}/{context.subdir}")],
            subdirs=(
                "noarch",
                context.subdir,
            ),
            installed_records=(),  # do not load installed
            pkgs_dirs=(),  # do not load local cache as a channel
            in_state=in_state,
        )

    index_helper = benchmark.pedantic(index, rounds=1)

    assert len(index_helper.repos) > 0


def test_load_channels_order(http_server_shards, http_server_shards_slow):
    # Setup two shard servers. server_one will have a small
    # delay in the response to mimic a slower response.
    server_one = http_server_shards_slow
    server_two = http_server_shards

    channel_one = Channel.from_url(f"{server_one}/noarch")
    channel_two = Channel.from_url(f"{server_two}/noarch")
    index_args = {
        "channels": [
            channel_one,
            channel_two,
        ],
        "in_state": SolverInputState(prefix="idontexist"),
    }

    with patch(
        "conda_libmamba_solver.index._is_sharded_repodata_enabled",
        return_value=True,
    ):
        shard_enabled_index = LibMambaIndexHelper(**index_args)

    # The expected output is that all of channel_one subdirs (noarch and current
    # platform) are ordered higher than channel_two subdirs.
    expected_output_channels = [
        channel_one.canonical_name,
        channel_one.canonical_name,
        channel_two.canonical_name,
        channel_two.canonical_name,
    ]
    assert [
        repo.channel.canonical_name for repo in shard_enabled_index.repos
    ] == expected_output_channels
