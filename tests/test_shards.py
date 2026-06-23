# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for sharded repodata integration with LibMambaIndexHelper.

Conda's own tests exercise the solver and shards code; just verify that we call
it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from conda._private.shards.shards import ShardLike
from conda.base.context import reset_context
from conda.gateways.logging import initialize_logging
from conda.models.channel import Channel

from conda_libmamba_solver.index import LibMambaIndexHelper
from conda_libmamba_solver.state import SolverInputState

if TYPE_CHECKING:
    import os


initialize_logging()
DATA = Path(__file__).parent / "data"


def test_libmamba_index_helper_uses_build_repodata_subset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """
    Test that LibMambaIndexHelper calls build_repodata_subset when both
    in_state and build_repodata_subset are provided.
    """
    monkeypatch.setenv("CONDA_PKGS_DIRS", str(tmp_path))
    reset_context()

    prefix = tmp_path / "prefix"
    prefix.mkdir()

    in_state = SolverInputState(prefix=str(prefix), requested=["test-package"])

    # Read the test repodata to create realistic shard data
    repodata_path = DATA / "mamba_repo" / "noarch" / "repodata.json"
    with open(repodata_path) as f:
        repodata = json.load(f)

    # Create mock build_repodata_subset that returns a subset of packages
    # The return value should be a dict mapping channel URLs to ShardLike objects
    # (which wrap repodata and provide an iter_records() interface)
    mock_build_repodata_subset = MagicMock()
    channel_url = f"file://{DATA / 'mamba_repo'}/noarch"
    # ShardLike's base_url is derived from the url parameter
    shardlike = ShardLike(repodata, url=channel_url + "/")
    mock_build_repodata_subset.return_value = {channel_url: shardlike}

    # Create LibMambaIndexHelper with the mock
    libmamba_index = LibMambaIndexHelper(
        channels=[Channel(str(DATA / "mamba_repo"))],
        subdirs=("noarch",),
        in_state=in_state,
        build_repodata_subset=mock_build_repodata_subset,
    )

    # Verify that build_repodata_subset was called
    assert mock_build_repodata_subset.called, "build_repodata_subset should have been called"

    # Verify the call was made with the correct arguments
    call_args = mock_build_repodata_subset.call_args
    assert call_args is not None

    root_packages, urls_to_channel = call_args[0]

    # root_packages should include installed and requested packages
    assert "test-package" in root_packages
    # urls_to_channel should be a dict mapping URLs to Channel objects
    assert isinstance(urls_to_channel, dict)
    assert len(urls_to_channel) > 0

    # The index should have been created successfully
    assert libmamba_index.db.repo_count() >= 1


def test_libmamba_index_helper_skips_build_repodata_subset_when_no_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: os.PathLike
):
    """
    Test that LibMambaIndexHelper does not call build_repodata_subset
    when in_state is None, even if build_repodata_subset is provided.

    This verifies that the shards path is only taken when both conditions are met.
    """
    monkeypatch.setenv("CONDA_PKGS_DIRS", str(tmp_path))
    reset_context()

    # Create mock for build_repodata_subset
    mock_build_repodata_subset = MagicMock()

    # Create LibMambaIndexHelper WITHOUT in_state (default None)
    libmamba_index = LibMambaIndexHelper(
        channels=[Channel(str(DATA / "mamba_repo"))],
        subdirs=("noarch",),
        in_state=None,
        build_repodata_subset=mock_build_repodata_subset,
    )

    # Verify that build_repodata_subset was NOT called
    assert not mock_build_repodata_subset.called

    # The index should still be created successfully via the standard path
    assert libmamba_index.db.repo_count() >= 1


def test_libmamba_index_helper_falls_back_when_build_repodata_subset_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: os.PathLike
):
    """
    Test that LibMambaIndexHelper falls back to standard repodata.json loading
    when build_repodata_subset returns None.

    This allows for graceful fallback in case shards are not available.
    """
    monkeypatch.setenv("CONDA_PKGS_DIRS", str(tmp_path))
    reset_context()

    # Create a temporary prefix for the solver state
    prefix = Path(tmp_path) / "prefix"
    prefix.mkdir()

    # Create a SolverInputState with some requested packages
    in_state = SolverInputState(prefix=str(prefix), requested=["test-package"])

    # Create mock build_repodata_subset that returns None to trigger fallback
    mock_build_repodata_subset = MagicMock(return_value=None)

    # Create LibMambaIndexHelper with the mock
    libmamba_index = LibMambaIndexHelper(
        channels=[Channel(str(DATA / "mamba_repo"))],
        subdirs=("noarch",),
        in_state=in_state,
        build_repodata_subset=mock_build_repodata_subset,
    )

    # Verify that build_repodata_subset was called
    assert mock_build_repodata_subset.called

    # The index should still have been created successfully via fallback
    assert libmamba_index.db.repo_count() >= 1
