import os

import pytest
from conda.base.context import reset_context
from conda.common.compat import on_win
from conda.core.subdir_data import SubdirData
from conda.gateways.logging import initialize_logging
from conda.models.channel import Channel

from conda_libmamba_solver.index2 import LibMambaIndexHelper

initialize_logging()


def test_given_channels(monkeypatch: pytest.MonkeyPatch, tmp_path: os.PathLike):
    monkeypatch.setenv("CONDA_PKGS_DIRS", str(tmp_path))
    reset_context()
    libmamba_index = LibMambaIndexHelper(
        channels=[Channel("conda-test/noarch")],
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

    libmamba_dot_conda_total = 0
    libmamba_tar_bz2_total = 0
    for repo in libmamba_index.repos:
        for pkg in libmamba_index.db.packages_in_repo(repo):
            if pkg.package_url.endswith(".conda"):
                libmamba_dot_conda_total += 1
            elif pkg.package_url.endswith(".tar.bz2"):
                libmamba_tar_bz2_total += 1
            else:
                raise RuntimeError(f"Unrecognized package URL: {pkg.package_url}")

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
        # assert conda_tar_bz2_total == libmamba_tar_bz2_total
        # Currently errors because libmamba does not deduplicate .tar.bz2 and .conda
        # See https://github.com/mamba-org/mamba/issues/3250
        # As of 2024-03-25:
        # AssertionError: assert 5691 == 19967
