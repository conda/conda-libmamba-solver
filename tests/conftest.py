# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause

pytest_plugins = (
    # Add testing fixtures and internal pytest plugins here
    "conda.testing",
    "conda.testing.fixtures",
)

# Allow fixtures from test_shards to be available globally, specifically in
# test_shards_subset:
from .test_shards import (  # noqa: F401
    http_server_shards,
    http_server_shards_slow,
    prepare_shards_test,
    shard_cache_with_data,
)
