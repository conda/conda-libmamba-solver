# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from conda.base.context import reset_context
from conda.common.compat import on_win
from conda.core.subdir_data import SubdirData
from conda.gateways.logging import initialize_logging
from conda.models.channel import Channel

from conda_libmamba_solver.index import (
    LibMambaIndexHelper,
    _package_info_from_package_dict,
)

if TYPE_CHECKING:
    import os


initialize_logging()
DATA = Path(__file__).parent / "data"


def test_given_channels(monkeypatch: pytest.MonkeyPatch, tmp_path: os.PathLike):
    monkeypatch.setenv("CONDA_PKGS_DIRS", str(tmp_path))
    reset_context()
    libmamba_index = LibMambaIndexHelper.from_platform_aware_channel(
        channel=Channel("conda-test/noarch")
    )
    assert libmamba_index.db.repo_count() == 1

    conda_index = SubdirData(Channel("conda-test/noarch"))
    conda_index.load()

    assert libmamba_index.db.package_count() == len(tuple(conda_index.iter_records()))


@pytest.mark.parametrize(
    "only_tar_bz2",
    (
        pytest.param("1", id="CONDA_USE_ONLY_TAR_BZ2=true"),
        pytest.param("", id="CONDA_USE_ONLY_TAR_BZ2=false"),
    ),
)
def test_defaults_use_only_tar_bz2(monkeypatch: pytest.MonkeyPatch, only_tar_bz2: bool):
    """
    Defaults is particular in the sense that it offers both .tar.bz2 and .conda for LOTS
    of packages. SubdirData ignores .tar.bz2 entries if they have a .conda counterpart.
    So if we count all the packages in each implementation, libmamba's has way more.
    To remain accurate, we test this with `use_only_tar_bz2`:
        - When true, we only count .tar.bz2
        - When false, we only count .conda
    """
    monkeypatch.setenv("CONDA_USE_ONLY_TAR_BZ2", only_tar_bz2)
    reset_context()
    libmamba_index = LibMambaIndexHelper(
        channels=[Channel("defaults")],
        subdirs=("noarch",),
        installed_records=(),  # do not load installed
        pkgs_dirs=(),  # do not load local cache as a channel
    )
    n_repos = 3 if on_win else 2
    assert len(libmamba_index.repos) == n_repos

    libmamba_dot_conda_total = libmamba_index.n_packages(
        filter_=lambda pkg: pkg.package_url.endswith(".conda")
    )
    libmamba_tar_bz2_total = libmamba_index.n_packages(
        filter_=lambda pkg: pkg.package_url.endswith(".tar.bz2")
    )

    conda_dot_conda_total = 0
    conda_tar_bz2_total = 0
    for channel_url in Channel("defaults/noarch").urls(subdirs=("noarch",)):
        conda_index = SubdirData(Channel(channel_url))
        conda_index.load()
        for pkg in conda_index.iter_records():
            if pkg["url"].endswith(".conda"):
                conda_dot_conda_total += 1
            elif pkg["url"].endswith(".tar.bz2"):
                conda_tar_bz2_total += 1
            else:
                raise RuntimeError(f"Unrecognized package URL: {pkg['url']}")

    if only_tar_bz2:
        assert conda_tar_bz2_total == libmamba_tar_bz2_total
        assert libmamba_dot_conda_total == conda_dot_conda_total == 0
    else:
        assert conda_dot_conda_total == libmamba_dot_conda_total
        assert conda_tar_bz2_total == libmamba_tar_bz2_total


def test_reload_channels(tmp_path: Path):
    (tmp_path / "noarch").mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / "mamba_repo" / "noarch" / "repodata.json", tmp_path / "noarch")
    initial_repodata = (tmp_path / "noarch" / "repodata.json").read_text()
    index = LibMambaIndexHelper(channels=[Channel(str(tmp_path))])
    initial_count = index.n_packages()
    SubdirData._cache_.clear()

    data = json.loads(initial_repodata)
    package = data["packages"]["test-package-0.1-0.tar.bz2"]
    data["packages"]["test-package-copy-0.1-0.tar.bz2"] = {**package, "name": "test-package-copy"}
    modified_repodata = json.dumps(data)
    (tmp_path / "noarch" / "repodata.json").write_text(modified_repodata)

    assert initial_repodata != modified_repodata
    # TODO: Remove this sleep after addressing
    # https://github.com/conda/conda/issues/13783
    time.sleep(1)
    index.reload_channel(Channel(str(tmp_path)))
    assert index.n_packages() == initial_count + 1


def test_package_info_from_package_dict_add_pip_as_python_dependency():
    """
    Test that _package_info_from_package_dict appends "pip" to dependencies
    when add_pip_as_python_dependency=True for Python packages.
    """
    # Test with Python 3.x package and add_pip_as_python_dependency=True
    python_record = {
        "name": "python",
        "version": "3.11.0",
        "build": "h96f0305_0",
        "build_number": 0,
        "depends": ["libffi >=3.4,<4.0"],
        "subdir": "osx-64",
    }

    package_info = _package_info_from_package_dict(
        python_record,
        "python-3.11.0-h96f0305_0.tar.bz2",
        url="https://conda.anaconda.com/pkgs/main/osx-64/python-3.11.0-h96f0305_0.tar.bz2",
        channel_id="pkgs/main",
        add_pip_as_python_dependency=True,
    )

    # pip should be appended to dependencies
    assert "pip" in package_info.dependencies
    assert "libffi >=3.4,<4.0" in package_info.dependencies
    assert len(package_info.dependencies) == 2
    assert package_info.name == "python"
    assert package_info.version == "3.11.0"


def test_package_info_from_package_dict_add_pip_as_python_dependency_false():
    """
    Test that _package_info_from_package_dict does NOT append "pip" when
    add_pip_as_python_dependency=False.
    """
    python_record = {
        "name": "python",
        "version": "3.11.0",
        "build": "h96f0305_0",
        "build_number": 0,
        "depends": ["libffi >=3.4,<4.0"],
        "subdir": "osx-64",
    }

    package_info = _package_info_from_package_dict(
        python_record,
        "python-3.11.0-h96f0305_0.tar.bz2",
        url="https://conda.anaconda.com/pkgs/main/osx-64/python-3.11.0-h96f0305_0.tar.bz2",
        channel_id="pkgs/main",
        add_pip_as_python_dependency=False,
    )

    # pip should NOT be appended
    assert "pip" not in package_info.dependencies
    assert len(package_info.dependencies) == 1
    assert list(package_info.dependencies) == ["libffi >=3.4,<4.0"]


def test_package_info_from_package_dict_add_pip_python2():
    """
    Test that _package_info_from_package_dict appends "pip" for Python 2.x packages.
    """
    python_record = {
        "name": "python",
        "version": "2.7.18",
        "build": "h9ed2024_0",
        "build_number": 0,
        "depends": [],
        "subdir": "osx-64",
    }

    package_info = _package_info_from_package_dict(
        python_record,
        "python-2.7.18-h9ed2024_0.tar.bz2",
        url="https://conda.anaconda.com/pkgs/main/osx-64/python-2.7.18-h9ed2024_0.tar.bz2",
        channel_id="pkgs/main",
        add_pip_as_python_dependency=True,
    )

    # pip should be appended for Python 2.x as well
    assert "pip" in package_info.dependencies
    assert len(package_info.dependencies) == 1


def test_package_info_from_package_dict_add_pip_non_python_package():
    """
    Test that _package_info_from_package_dict does NOT append "pip" for non-Python packages.
    """
    record = {
        "name": "numpy",
        "version": "1.24.0",
        "build": "py311h5a7a992_0",
        "build_number": 0,
        "depends": ["python >=3.11"],
        "subdir": "osx-64",
    }

    package_info = _package_info_from_package_dict(
        record,
        "numpy-1.24.0-py311h5a7a992_0.tar.bz2",
        url="https://conda.anaconda.com/pkgs/main/osx-64/numpy-1.24.0-py311h5a7a992_0.tar.bz2",
        channel_id="pkgs/main",
        add_pip_as_python_dependency=True,
    )

    # pip should NOT be appended for non-Python packages
    assert "pip" not in package_info.dependencies
    assert len(package_info.dependencies) == 1
    assert list(package_info.dependencies) == ["python >=3.11"]


def test_package_info_from_package_dict_add_pip_invalid_version():
    """
    Test that _package_info_from_package_dict does NOT append "pip" for Python packages
    with versions that don't start with "2." or "3.".
    """
    record = {
        "name": "python",
        "version": "1.0.0",  # Invalid Python version
        "build": "h0000_0",
        "build_number": 0,
        "depends": [],
        "subdir": "osx-64",
    }

    package_info = _package_info_from_package_dict(
        record,
        "python-1.0.0-h0000_0.tar.bz2",
        url="https://conda.anaconda.com/pkgs/main/osx-64/python-1.0.0-h0000_0.tar.bz2",
        channel_id="pkgs/main",
        add_pip_as_python_dependency=True,
    )

    # pip should NOT be appended for invalid Python versions
    assert "pip" not in package_info.dependencies
    assert len(package_info.dependencies) == 0
