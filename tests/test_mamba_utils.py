# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import pytest
from conda.base.context import reset_context

from conda_libmamba_solver.mamba_utils import init_libmamba_context


@pytest.mark.parametrize(
    "verbosity,expected_verbosity",
    (
        pytest.param("0", 0, id="error level logging"),
        pytest.param("1", 1, id="warning level logging"),
        pytest.param("2", 2, id="info level logging"),
        pytest.param("3", 3, id="debug level logging"),
        pytest.param("4", 4, id="trace level logging"),
        pytest.param("5", 4, id="large logging level"),
    ),
)
def test_valid_verbosity(monkeypatch: pytest.MonkeyPatch, verbosity: str, expected_verbosity: int):
    monkeypatch.setenv("CONDA_VERBOSITY", verbosity)
    reset_context()
    libmamba_context = init_libmamba_context()
    assert libmamba_context.output_params.verbosity == expected_verbosity
