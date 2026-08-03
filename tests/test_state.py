# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from conda.base.constants import UpdateModifier
from conda.models.match_spec import MatchSpec
from conda.models.records import PackageRecord, PrefixRecord
from libmambapy.solver import Request

from conda_libmamba_solver.solver import LibMambaSolver
from conda_libmamba_solver.state import SolverInputState, SolverOutputState


def test_installed_cache_tracks_installed_names(tmp_path):
    state = SolverInputState(tmp_path)
    records = state.prefix_data._prefix_records
    records["alpha"] = PrefixRecord(
        name="alpha",
        version="1.0",
        build="0",
        build_number=0,
        channel="conda-forge",
        subdir="noarch",
        fn="alpha-1.0-0.conda",
    )

    installed = state.installed
    assert tuple(installed) == ("alpha",)
    assert state.installed is installed

    del records["alpha"]
    records["beta"] = PrefixRecord(
        name="beta",
        version="1.0",
        build="0",
        build_number=0,
        channel="conda-forge",
        subdir="noarch",
        fn="beta-1.0-0.conda",
    )

    assert tuple(state.installed) == ("beta",)


def test_update_all_includes_python_in_always_update(tmp_path):
    state = SolverInputState(tmp_path, update_modifier=UpdateModifier.UPDATE_ALL)
    records = state.prefix_data._prefix_records
    records["python"] = PrefixRecord(
        name="python",
        version="3.13.9",
        build="0",
        build_number=0,
        channel="conda-forge",
        subdir="noarch",
        fn="python-3.13.9-0.conda",
    )
    records["numpy"] = PrefixRecord(
        name="numpy",
        version="2.0.0",
        build="0",
        build_number=0,
        channel="conda-forge",
        subdir="noarch",
        fn="numpy-2.0.0-0.conda",
    )

    assert "python" in state.always_update
    assert "numpy" in state.always_update


def test_update_all_emits_python_pin_and_update(tmp_path):
    """UPDATE_ALL must request a python update while pinning to the current major.minor.

    That combination allows patch upgrades (3.13.9 → 3.13.14) and blocks minor jumps
    (3.13 → 3.14) unless python is requested explicitly.
    """
    in_state = SolverInputState(
        tmp_path,
        update_modifier=UpdateModifier.UPDATE_ALL,
        command="update",
    )
    in_state.prefix_data._prefix_records["python"] = PrefixRecord(
        name="python",
        version="3.13.9",
        build="0",
        build_number=0,
        channel="conda-forge",
        subdir="noarch",
        fn="python-3.13.9-0.conda",
    )
    out_state = SolverOutputState(solver_input_state=in_state)
    solver = LibMambaSolver(tmp_path, channels=("defaults",), command="update")

    tasks = solver._specs_to_request_jobs_add(in_state, out_state)

    pins = [str(spec) for spec in tasks.get(Request.Pin, ())]
    updates = [str(spec) for spec in tasks.get(Request.Update, ())]

    assert "python" in in_state.always_update
    assert "python" in updates
    assert "python 3.13.*" in pins

    # Patch within major.minor is allowed; the next minor is not.
    pin = MatchSpec("python 3.13.*")
    assert pin.match(
        PackageRecord(
            name="python",
            version="3.13.14",
            build="0",
            build_number=0,
            channel="conda-forge",
            subdir="noarch",
            fn="python-3.13.14-0.conda",
        )
    )
    assert not pin.match(
        PackageRecord(
            name="python",
            version="3.14.0",
            build="0",
            build_number=0,
            channel="conda-forge",
            subdir="noarch",
            fn="python-3.14.0-0.conda",
        )
    )


def test_explicit_python_request_skips_major_minor_pin(tmp_path):
    """Explicit python on the CLI must not get the implicit X.Y.* pin."""
    in_state = SolverInputState(
        tmp_path,
        requested=("python=3.14",),
        update_modifier=UpdateModifier.UPDATE_SPECS,
        command="install",
    )
    in_state.prefix_data._prefix_records["python"] = PrefixRecord(
        name="python",
        version="3.13.9",
        build="0",
        build_number=0,
        channel="conda-forge",
        subdir="noarch",
        fn="python-3.13.9-0.conda",
    )
    out_state = SolverOutputState(solver_input_state=in_state)
    solver = LibMambaSolver(
        tmp_path,
        channels=("defaults",),
        specs_to_add=("python=3.14",),
        command="install",
    )

    tasks = solver._specs_to_request_jobs_add(in_state, out_state)
    pins = [str(spec) for spec in tasks.get(Request.Pin, ())]

    assert "python 3.13.*" not in pins
