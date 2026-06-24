# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Ensure configuration plugin functions as expected
"""

import pytest
from conda.base.context import reset_context


@pytest.fixture(scope="function", autouse=True)
def always_reset_context():
    reset_context()
