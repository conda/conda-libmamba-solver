# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Ensure configuration plugin functions as expected
"""

import pytest
from conda.auxlib.ish import dals
from conda.base.context import context, reset_context


@pytest.fixture(scope="function", autouse=True)
def always_reset_context():
    reset_context()


def test_enabled_sharded_repodata():
    """
    Ensure that the setting `plugins.use_sharded_repodata` exists and is set
    to the correct default value.
    """
    assert not context.plugins.use_sharded_repodata


def test_enabled_sharded_repodata_environment_variable(monkeypatch):
    """
    Ensure that the setting `plugins.use_sharded_repodata_environment_variable`
    is set correctly when set as an environment variable.
    """
    monkeypatch.setenv("CONDA_PLUGINS_USE_SHARDED_REPODATA", "true")
    context.__init__()

    assert context.plugins.use_sharded_repodata


def test_enabled_sharded_repodata_condarc(tmp_path):
    """
    Ensure that the setting `plugins.use_sharded_repodata_environment_variable`
    is set correctly when set in a condarc file.
    """
    condarc_file = tmp_path / "conda.yml"
    with condarc_file.open("w") as f:
        condarc_yml = dals("""
            plugins:
              use_sharded_repodata: true
        """)
        f.write(condarc_yml)

    context.__init__(search_path=(str(condarc_file),))

    assert context.plugins.use_sharded_repodata
