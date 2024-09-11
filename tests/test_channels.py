# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from subprocess import check_call
from typing import TYPE_CHECKING
from urllib.request import urlretrieve

import pytest
from conda.base.context import context, reset_context
from conda.common.compat import on_linux, on_win
from conda.common.io import env_vars
from conda.core.prefix_data import PrefixData
from conda.exceptions import DryRunExit
from conda.models.channel import Channel
from conda.testing.integration import (
    _get_temp_prefix,
    make_temp_env,
    package_is_installed,
)
from conda.testing.integration import run_command as conda_inprocess

from .channel_testing.helpers import http_server_auth_basic  # noqa: F401
from .channel_testing.helpers import http_server_auth_basic_email  # noqa: F401
from .channel_testing.helpers import http_server_auth_none  # noqa: F401
from .channel_testing.helpers import http_server_auth_token  # noqa: F401
from .channel_testing.helpers import create_with_channel
from .utils import conda_subprocess, write_env_config

if TYPE_CHECKING:
    from conda.testing import CondaCLIFixture, PathFactoryFixture, TmpEnvFixture
    from pytest import MonkeyPatch

DATA = Path(__file__).parent / "data"


def test_channel_matchspec(conda_cli: CondaCLIFixture, tmp_path: Path) -> None:
    stdout, *_ = conda_cli(
        "create",
        f"--prefix={tmp_path}",
        "--solver=libmamba",
        "--json",
        "--override-channels",
        "--channel=defaults",
        "--yes",
        "conda-forge::libblas=*=*openblas",
        "python=3.9",
    )
    result = json.loads(stdout)
    assert result["success"] is True
    for record in result["actions"]["LINK"]:
        if record["name"] == "numpy":
            assert record["channel"] == "conda-forge"
        elif record["name"] == "python":
            assert record["channel"] == "pkgs/main"


def test_channels_prefixdata(tmp_env: TmpEnvFixture, conda_cli: CondaCLIFixture) -> None:
    """
    Make sure libmamba does not complain about missing channels
    used in previous commands.

    See https://github.com/conda/conda/issues/11790
    """
    with tmp_env("conda-forge::xz", "python", "--solver=libmamba") as prefix:
        stdout, stderr, _ = conda_cli(
            "install",
            f"--prefix={prefix}",
            "--solver=libmamba",
            "--yes",
            "pytest",
        )
        assert (
            "Selected channel specific (or force-reinstall) job, "
            "but package is not available from channel. "
            "Solve job will fail." not in (stdout + stderr)
        )


def test_channels_installed_unavailable(
    tmp_env: TmpEnvFixture,
    conda_cli: CondaCLIFixture,
) -> None:
    """Ensure we don't fail if a channel coming ONLY from an installed pkg is unavailable"""
    with tmp_env("xz", "--solver=libmamba") as prefix:
        pd = PrefixData(prefix)
        pd.load()
        record = pd.get("xz")
        assert record
        record.channel = Channel.from_url("file:///nonexistent")

        with pytest.raises(DryRunExit):
            conda_cli(
                "install",
                f"--prefix={prefix}",
                "zlib",
                "--solver=libmamba",
                "--dry-run",
            )


def _setup_conda_forge_as_defaults(prefix, force=False):
    write_env_config(
        prefix,
        force=force,
        channels=["defaults"],
        default_channels=["conda-forge"],
    )


def _setup_channels_alias(prefix, force=False):
    write_env_config(
        prefix,
        force=force,
        channels=["conda-forge", "defaults"],
        channel_alias="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud",
        migrated_channel_aliases=["https://conda.anaconda.org"],
        default_channels=[
            "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main",
            "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r",
            "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2",
        ],
    )


def _setup_channels_custom(prefix, force=False):
    write_env_config(
        prefix,
        force=force,
        channels=["conda-forge", "defaults"],
        custom_channels={
            "conda-forge": "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud",
        },
    )


@pytest.mark.parametrize(
    "config_env",
    (
        _setup_channels_alias,
        _setup_channels_custom,
    ),
)
def test_mirrors_do_not_leak_channels(
    config_env,
    path_factory: PathFactoryFixture,
    tmp_env: TmpEnvFixture,
    monkeypatch: MonkeyPatch,
    conda_cli: CondaCLIFixture,
) -> None:
    """
    https://github.com/conda/conda-libmamba-solver/issues/108

    On existing environments, we load channels from the prefix data information
    to silence some warnings in libmamba (see `test_channels_prefixdata`).

    In some configurations that use proxies or Anaconda mirrors, this can lead to
    the non-mirrored (original) channels being loaded. In airgapped contexts, this
    is undesirable.
    """

    with env_vars({"CONDA_PKGS_DIRS": path_factory()}), tmp_env() as prefix:
        assert (prefix / "conda-meta" / "history").exists()

        # Setup conda configuration
        config_env(prefix)
        common = [f"--prefix={prefix}", "--solver=libmamba", "--json", "--yes", "-vv"]

        # fake activation so config is loaded
        monkeypatch.setenv("CONDA_PREFIX", str(prefix))

        # Create an environment using mirrored channels only
        stdout, stderr, _ = conda_cli("install", *common, "python", "pip")
        result = json.loads(stdout)
        if stderr:
            assert "conda.anaconda.org" not in stderr

        for pkg in result["actions"]["LINK"]:
            assert pkg["channel"] == "conda-forge", pkg
            assert (
                pkg["base_url"]
                == "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"
            ), pkg

        # Make a change to that channel
        stdout, stderr, _ = conda_cli("install", *common, "pytest")

        # Ensure that the loaded channels are ONLY the mirrored ones
        result = json.loads(stdout)
        if stderr:
            assert "conda.anaconda.org" not in stderr

        for pkg in result["actions"]["LINK"]:
            assert pkg["channel"] == "conda-forge", pkg
            assert (
                pkg["base_url"]
                == "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"
            ), pkg

        # Ensure that other end points were never loaded


def test_jax_and_jaxlib(monkeypatch: MonkeyPatch, conda_cli: CondaCLIFixture) -> None:
    # https://github.com/conda/conda-libmamba-solver/issues/221
    monkeypatch.setenv("CONDA_SUBDIR", "linux-64")
    reset_context()
    assert context.subdir == "linux-64"

    for specs in (("jax", "jaxlib"), ("jaxlib", "jax")):
        stdout, _, _ = conda_cli(
            "create",
            "--solver=libmamba",
            "--json",
            "--dry-run",
            "--override-channels",
            "--channel=defaults",
            f"conda-forge::{specs[0]}",
            f"conda-forge::{specs[1]}",
            raises=DryRunExit,
        )
        result = json.loads(stdout)
        assert result["success"] is True
        pkgs = {pkg["name"] for pkg in result["actions"]["LINK"]}
        assert specs[0] in pkgs
        assert specs[1] in pkgs


def test_encoding_file_paths(tmp_path: Path, conda_cli: CondaCLIFixture) -> None:
    tmp_channel = tmp_path / "channel+some+encodable+bits"
    repo = Path(__file__).parent / "data" / "mamba_repo"
    shutil.copytree(repo, tmp_channel)
    stdout, stderr, err = conda_cli(
        "create",
        f"--prefix={tmp_path / 'env'}",
        f"--channel={tmp_channel}",
        "--solver=libmamba",
        "--yes",
        "test-package",
    )
    print(stdout)
    print(stderr, file=sys.stderr)
    assert not err
    assert list((tmp_path / "env" / "conda-meta").glob("test-package-*.json"))


def test_conda_build_with_aliased_channels(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    conda_cli: CondaCLIFixture,
) -> None:
    # https://github.com/conda/conda-libmamba-solver/issues/363
    condarc = Path.home() / ".condarc"
    condarc_contents = condarc.read_text() if condarc.is_file() else None

    if on_win:
        bld_path = str(Path(os.getenv("RUNNER_TEMP", tmp_path), "bld"))
    else:
        bld_path = str(tmp_path / "conda-bld")
    monkeypatch.setenv("CONDA_BLD_PATH", bld_path)

    try:
        _setup_conda_forge_as_defaults(Path.home(), force=True)
        conda_cli(
            "build",
            "--override-channels",
            "--channel=defaults",
            DATA / "conda_build_recipes" / "jedi",
        )
    finally:
        if condarc_contents:
            condarc.write_text(condarc_contents)
        else:
            condarc.unlink()


def test_http_server_auth_none(http_server_auth_none):
    create_with_channel(http_server_auth_none)


def test_http_server_auth_basic(http_server_auth_basic):
    create_with_channel(http_server_auth_basic)


def test_http_server_auth_basic_email(http_server_auth_basic_email):
    create_with_channel(http_server_auth_basic_email)


def test_http_server_auth_token(http_server_auth_token):
    create_with_channel(http_server_auth_token)


def test_http_server_auth_token_in_defaults(http_server_auth_token):
    condarc = Path.home() / ".condarc"
    condarc_contents = condarc.read_text() if condarc.is_file() else None
    try:
        write_env_config(
            Path.home(),
            force=True,
            channels=["defaults"],
            default_channels=[http_server_auth_token],
        )
        reset_context()
        conda_subprocess("info", capture_output=False)
        conda_subprocess(
            "create",
            "-p",
            _get_temp_prefix(use_restricted_unicode=on_win),
            "--solver=libmamba",
            "test-package",
        )
    finally:
        if condarc_contents:
            condarc.write_text(condarc_contents)
        else:
            condarc.unlink()


def test_local_spec():
    "https://github.com/conda/conda-libmamba-solver/issues/398"
    env = os.environ.copy()
    env["CONDA_BLD_PATH"] = str(DATA / "mamba_repo")
    process = conda_subprocess(
        "create",
        "-p",
        _get_temp_prefix(use_restricted_unicode=on_win),
        "--dry-run",
        "--solver=libmamba",
        "--channel=local",
        "test-package",
        env=env,
    )
    assert process.returncode == 0

    process = conda_subprocess(
        "create",
        "-p",
        _get_temp_prefix(use_restricted_unicode=on_win),
        "--dry-run",
        "--solver=libmamba",
        "local::test-package",
        env=env,
    )
    assert process.returncode == 0


def test_unknown_channels_do_not_crash(tmp_path):
    "https://github.com/conda/conda-libmamba-solver/issues/418"
    DATA = Path(__file__).parent / "data"
    test_pkg = DATA / "mamba_repo" / "noarch" / "test-package-0.1-0.tar.bz2"
    with make_temp_env("ca-certificates") as prefix:
        # copy pkg to a new non-channel-like location without repodata around to obtain
        # '<unknown>' channel and reproduce the issue
        temp_pkg = Path(prefix, "test-package-0.1-0.tar.bz2")
        shutil.copy(test_pkg, temp_pkg)
        conda_inprocess("install", prefix, str(temp_pkg))
        assert package_is_installed(prefix, "test-package")
        conda_inprocess("install", prefix, "zlib")
        assert package_is_installed(prefix, "zlib")


@pytest.mark.skipif(not on_linux, reason="Only run on Linux")
def test_use_cache_works_offline_fresh_install_keep(tmp_path):
    """
    https://github.com/conda/conda-libmamba-solver/issues/396

    constructor installers have a `-k` switch (keep) to leave the
    pkgs/ cache prepopulated. Offline updating from the cache should be a
    harmless no-op, not a hard crash.
    """
    miniforge_url = (
        "https://github.com/conda-forge/miniforge/releases/"
        f"latest/download/Miniforge3-Linux-{os.uname().machine}.sh"
    )
    urlretrieve(miniforge_url, tmp_path / "miniforge.sh")
    # bkfp: batch, keep, force, prefix
    check_call(["bash", str(tmp_path / "miniforge.sh"), "-bkfp", tmp_path / "miniforge"])
    env = os.environ.copy()
    env["CONDA_ROOT_PREFIX"] = str(tmp_path / "miniforge")
    env["CONDA_PKGS_DIRS"] = str(tmp_path / "miniforge" / "pkgs")
    env["CONDA_ENVS_DIRS"] = str(tmp_path / "miniforge" / "envs")
    env["HOME"] = str(tmp_path)  # ignore ~/.condarc
    args = (
        "update",
        "-p",
        tmp_path / "miniforge",
        "--all",
        "--dry-run",
        "--override-channels",
        "--channel=conda-forge",
    )
    kwargs = {"capture_output": False, "check": True, "env": env}
    conda_subprocess(*args, "--offline", **kwargs)
    conda_subprocess(*args, "--use-index-cache", **kwargs)
    conda_subprocess(*args, "--offline", "--use-index-cache", **kwargs)
