# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Test sharded repodata.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from conda.base.context import context, reset_context
from conda.core.subdir_data import SubdirData
from conda.gateways.connection.session import get_session
from conda.models.channel import Channel

from conda_libmamba_solver.shards import (
    ShardLike,
    fetch_shards,
    shard_mentioned_packages,
)

HERE = Path(__file__).parent


def package_names(shard):
    """
    All package names mentioned in a shard (should be a single package name)
    """
    return set(package["name"] for package in shard["packages"].values()) | set(
        package["name"] for package in shard["packages.conda"].values()
    )


@pytest.fixture
def conda_no_token(monkeypatch: pytest.MonkeyPatch):
    """
    Reset token to avoid being logged in. e.g. the testing channel doesn't understand them.
    """
    monkeypatch.setenv("CONDA_TOKEN", "")
    reset_context()


def test_shards(conda_no_token: None):
    """
    Test basic shard fetch etc.
    """
    channels = [
        Channel.from_url(f"https://conda.anaconda.org/conda-forge-sharded/{subdir}")
        for subdir in context.subdirs
    ]

    # Would eagerly download repodata.json.zst for all channels
    # helper = LibMambaIndexHelper(
    #     channels,
    #     (),  # subdirs
    #     "repodata.json",
    #     installed,
    #     (),  # pkgs_dirs to load packages locally when offline
    #     in_state=in_state,
    # )

    # accessing "helper.repos" downloads repodata.json in the traditional way
    # for all channels:
    # print(helper.repos)

    subdir_data = SubdirData(channels[0])
    found = fetch_shards(subdir_data)
    assert found, f"Shards not found for {channels[0]}"

    for package in found.packages_index:
        shard_url = found.shard_url(package)
        assert shard_url.startswith("http")  # or channel url or shards_base_url

    session = get_session(
        subdir_data.url_w_subdir
    )  # XXX session could be different based on shards_base_url and different than the packages base_url

    # download or fetch-from-cache a random set of shards

    for package in random.choices([*found.packages_index.keys()], k=16):
        shard = found.fetch_shard(package, session)

        mentioned_in_shard = shard_mentioned_packages(shard)
        print(package_names(shard), mentioned_in_shard)


def test_shardlike():
    """
    ShardLike class presents repodata.json as shards in a way that is suitable
    for our subsetting algorithm.
    """
    repodata = json.loads(
        (Path(__file__).parent / "data" / "mamba_repo" / "noarch" / "repodata.json").read_text()
    )

    # make fake packages
    for n in range(10):
        for m in range(n):
            repodata["packages"][f"test{n}{m}.tar.bz2"] = {"name": f"test{n}"}
            repodata["packages.conda"][f"test{n}{m}.tar.bz2"] = {"name": f"test{n}"}

    as_shards = ShardLike(repodata)

    assert len(as_shards.repodata_no_packages)
    assert len(as_shards.shards)

    assert sorted(as_shards.shards["test4"]["packages"].keys()) == [
        "test40.tar.bz2",
        "test41.tar.bz2",
        "test42.tar.bz2",
        "test43.tar.bz2",
    ]
