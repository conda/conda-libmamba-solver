# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import json

import pytest
from conda.base.constants import UpdateModifier
from conda.base.context import reset_context

from conda_libmamba_solver.exceptions import LibMambaUnsatisfiableError
from conda_libmamba_solver.solver import LibMambaSolver


@pytest.fixture
def python_history_solver(tmp_path, monkeypatch):
    monkeypatch.setenv("CONDA_ADD_PIP_AS_PYTHON_DEPENDENCY", "false")
    monkeypatch.setenv("CONDA_NOTIFY_OUTDATED_CONDA", "false")
    monkeypatch.setenv("CONDA_PKGS_DIRS", str(tmp_path / "pkgs"))
    monkeypatch.setenv("CONDA_SUBDIR", "linux-64")
    reset_context()

    def make_solver(
        history="python=3",
        requested=("sqlite",),
        *,
        newer_python="python-3.13.2-0",
        python_can_keep_version=False,
        pinned=None,
        remove=(),
    ):
        packages = {
            "ncurses-6.0-0": [],
            "ncurses-6.1-0": [],
            "libedit-3.1-0": ["ncurses 6.0.*"],
            "libedit-3.1.20170329-0": ["ncurses >=6.1,<7.0a0"],
            "sqlite-3.21.0-0": ["libedit"],
            "sqlite-3.23.1-0": ["libedit"],
            "sqlite-3.24.0-0": ["libedit >=3.1.20170329,<3.2.0a0"],
            "python-3.13.1-0": [
                "ncurses >=6.0,<7.0a0" if python_can_keep_version else "ncurses 6.0.*",
                "sqlite >=3.20.1,<4.0a0",
            ],
        }
        if newer_python:
            packages[newer_python] = ["ncurses >=6.0,<7.0a0", "sqlite >=3.23.1,<4.0a0"]

        records = {}
        for key, depends in packages.items():
            name, version, build = key.rsplit("-", 2)
            records[key] = {
                "name": name,
                "version": version,
                "build": build,
                "build_number": int(build),
                "depends": depends,
                "subdir": "linux-64",
                "fn": f"{key}.tar.bz2",
            }

        channel = tmp_path / "channel"
        for subdir in ("linux-64", "noarch"):
            (channel / subdir).mkdir(parents=True)
            (channel / subdir / "repodata.json").write_text(
                json.dumps(
                    {
                        "info": {"subdir": subdir},
                        "packages": (
                            {record["fn"]: record for record in records.values()}
                            if subdir == "linux-64"
                            else {}
                        ),
                    }
                )
            )

        prefix = tmp_path / "env"
        conda_meta = prefix / "conda-meta"
        conda_meta.mkdir(parents=True)
        for key in ("ncurses-6.0-0", "libedit-3.1-0", "sqlite-3.21.0-0", "python-3.13.1-0"):
            record = records[key]
            (conda_meta / f"{key}.json").write_text(
                json.dumps(
                    {
                        **record,
                        "channel": f"{channel.as_uri()}/linux-64",
                        "url": f"{channel.as_uri()}/linux-64/{record['fn']}",
                    }
                )
            )
        specs = ["sqlite=3.21"]
        if history:
            specs.append(history)
        (conda_meta / "history").write_text(
            "==> 2026-01-01 00:00:00 <==\n"
            "# cmd: conda create python sqlite\n"
            f"# update specs: {specs!r}\n"
        )
        if pinned:
            (conda_meta / "pinned").write_text(f"{pinned}\n")

        return LibMambaSolver(
            prefix,
            channels=(channel.as_uri(),),
            subdirs=("linux-64", "noarch"),
            specs_to_add=() if remove else requested,
            specs_to_remove=remove,
            command="remove" if remove else "install",
        )

    return make_solver


@pytest.mark.parametrize(
    "history,expected_python,expected_sqlite",
    [
        ("python=3", "3.13.2", "3.24.0"),
        ("python=3.13.1", "3.13.1", "3.23.1"),
        ("python<3.13.2", "3.13.1", "3.23.1"),
    ],
)
def test_install_respects_python_history(
    python_history_solver, history, expected_python, expected_sqlite
):
    solver = python_history_solver(history)
    records = {record.name: record for record in solver.solve_final_state()}
    assert records["python"].version == expected_python
    assert records["sqlite"].version == expected_sqlite


@pytest.mark.parametrize(
    "requested,expected_build,expected_sqlite",
    [(("sqlite",), "0", "3.23.1"), (("sqlite>=3.24",), "1", "3.24.0")],
)
def test_install_preserves_python_and_relaxes_conflicting_history_build(
    python_history_solver, requested, expected_build, expected_sqlite
):
    solver = python_history_solver("python=3.13.1=0", requested, newer_python="python-3.13.1-1")
    records = {record.name: record for record in solver.solve_final_state()}
    assert records["python"].version == "3.13.1"
    assert records["python"].build == expected_build
    assert records["sqlite"].version == expected_sqlite


def test_install_cannot_remove_python_to_satisfy_requested_package(python_history_solver):
    solver = python_history_solver(requested=("sqlite>=3.24",), newer_python=None)
    with pytest.raises(LibMambaUnsatisfiableError):
        solver.solve_final_state()


def test_conda_build_does_not_preserve_unrequested_history_python(
    python_history_solver, monkeypatch
):
    solver = python_history_solver(requested=("sqlite>=3.24",), newer_python=None)
    monkeypatch.setattr(solver, "_called_from_conda_build", lambda: True)
    records = {record.name: record for record in solver.solve_final_state()}
    assert "python" not in records
    assert records["sqlite"].version == "3.24.0"


def test_install_does_not_update_python_unnecessarily(python_history_solver):
    solver = python_history_solver(python_can_keep_version=True)
    records = {record.name: record for record in solver.solve_final_state()}
    assert records["python"].version == "3.13.1"
    assert records["sqlite"].version == "3.24.0"


@pytest.mark.parametrize("history", ["python=3", "python=3.13.1"])
def test_prune_can_remove_python_from_history(python_history_solver, history):
    solver = python_history_solver(history)
    records = {record.name: record for record in solver.solve_final_state(prune=True)}
    assert "python" not in records
    assert records["sqlite"].version == "3.24.0"


def test_remove_can_remove_python_from_history(python_history_solver):
    solver = python_history_solver(remove=("sqlite",))
    records = {record.name: record for record in solver.solve_final_state()}
    assert "python" not in records
    assert "sqlite" not in records


@pytest.mark.parametrize("history", ["python=3", None])
def test_freeze_installed_preserves_python(python_history_solver, history):
    solver = python_history_solver(history)
    records = {
        record.name: record
        for record in solver.solve_final_state(update_modifier=UpdateModifier.FREEZE_INSTALLED)
    }
    assert records["python"].version == "3.13.1"
    assert records["sqlite"].version == "3.23.1"


def test_explicit_python_request_overrides_history(python_history_solver):
    solver = python_history_solver("python=3.13.1", ("sqlite", "python=3.13.2"))
    records = {record.name: record for record in solver.solve_final_state()}
    assert records["python"].version == "3.13.2"
    assert records["sqlite"].version == "3.24.0"


def test_update_all_can_update_python_past_history(python_history_solver):
    solver = python_history_solver("python=3.13.1", requested=())
    records = {
        record.name: record
        for record in solver.solve_final_state(update_modifier=UpdateModifier.UPDATE_ALL)
    }
    assert records["python"].version == "3.13.2"
    assert records["sqlite"].version == "3.24.0"


@pytest.mark.parametrize("ignore_pinned", [False, True])
def test_python_history_respects_explicit_pin(python_history_solver, ignore_pinned):
    solver = python_history_solver(pinned="python=3.13.1")
    records = {
        record.name: record for record in solver.solve_final_state(ignore_pinned=ignore_pinned)
    }
    assert records["python"].version == ("3.13.2" if ignore_pinned else "3.13.1")
    assert records["sqlite"].version == ("3.24.0" if ignore_pinned else "3.23.1")
