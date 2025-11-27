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
# Register channel_testing fixtures for use across all tests:
from .channel_testing.helpers import (  # noqa: F401
    http_server_auth_basic,
    http_server_auth_basic_email,
    http_server_auth_none,
    http_server_auth_none_debug_packages,
    http_server_auth_none_debug_repodata,
    http_server_auth_token,
)
from .test_shards import (  # noqa: F401
    http_server_shards,
    prepare_shards_test,
    shard_cache_with_data,
)
