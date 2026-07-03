# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from conda.models.records import PrefixRecord

from conda_libmamba_solver.state import SolverInputState


def prefix_record(name: str) -> PrefixRecord:
    return PrefixRecord(
        name=name,
        version="1.0",
        build="0",
        build_number=0,
        channel="conda-forge",
        subdir="noarch",
        fn=f"{name}-1.0-0.conda",
    )


def test_installed_cache_tracks_installed_names(tmp_path):
    state = SolverInputState(tmp_path)
    records = state.prefix_data._prefix_records
    records["alpha"] = prefix_record("alpha")

    installed = state.installed
    assert tuple(installed) == ("alpha",)
    assert state.installed is installed

    del records["alpha"]
    records["beta"] = prefix_record("beta")

    assert tuple(state.installed) == ("beta",)
