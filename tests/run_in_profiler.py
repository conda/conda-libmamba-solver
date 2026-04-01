#!/usr/bin/env python
# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Historical entry point for scalene profiling of sharded repodata traversal.

Shard implementation and tests live in conda. Profile with:

    cd /path/to/conda
    python -m scalene -m pytest \\
      tests/gateways/repodata/test_shards_subset.py::test_build_repodata_subset_pipelined
"""

from __future__ import annotations

import logging
import os

from conda.base.context import reset_context
from conda.gateways.repodata.shards import cache, core, subset

os.environ.setdefault("CONDA_TOKEN", "")
os.environ.setdefault("CONDA_PLUGINS_USE_SHARDED_REPODATA", "1")
os.environ.setdefault(
    "CONDA_REPODATA_THREADS",
    "10",
)
reset_context()

logging.basicConfig(level=logging.INFO)
for module in (core, cache, subset):
    module.log.setLevel(logging.DEBUG)

if __name__ == "__main__":
    print(  # noqa: T201
        "Profiling target moved to conda: "
        "tests/gateways/repodata/test_shards_subset.py::test_build_repodata_subset_pipelined"
    )
