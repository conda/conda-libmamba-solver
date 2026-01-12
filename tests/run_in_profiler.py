#!/usr/bin/env python
# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
To be run with python -m scalene <this file>
"""

import logging
import os
import pathlib

import pytest
from conda.base.context import reset_context

import tests.test_index
import tests.test_shards_subset
from conda_libmamba_solver import shards, shards_cache, shards_subset

os.environ["CONDA_TOKEN"] = ""
os.environ["CONDA_PLUGINS_USE_SHARDED_REPODATA"] = "1"
os.environ["CONDA_REPODATA_THREADS"] = (
    "10"  # shave a half second off our time by avoiding CondaSession() creation
)
reset_context()
pathlib.Path("/tmp/shards").mkdir(exist_ok=True)

tmp_path = pathlib.Path("/tmp/shards")

logging.basicConfig(level=logging.INFO)
for module in (shards, shards_cache, shards_subset):
    module.log.setLevel(logging.DEBUG)

# tests.test_shards_subset.test_build_repodata_subset_pipelined(None, tmp_path)


class Benchmark:
    def pedantic(self, fn, rounds: int = 1):
        return fn()
        # for _ in range(1):
        #     rc = fn()
        # return rc


monkeypatch = pytest.MonkeyPatch()
for i in range(16):
    tests.test_index.test_load_channel_repo_info_shards(
        "shard", ("django", "celery"), tmp_path, None, monkeypatch, Benchmark()
    )
