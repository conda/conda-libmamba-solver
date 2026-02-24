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
    http_server_shards as http_server_shards,
)
from .test_shards import (
    mock_cache as mock_cache,
)
from .test_shards import (
    prepare_shards_test as prepare_shards_test,
)
from .test_shards import (
    shard_cache_with_data as shard_cache_with_data,
)
from .test_shards import (
    shard_factory as shard_factory,
)
