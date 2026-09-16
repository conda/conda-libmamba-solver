# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from types import SimpleNamespace

import pytest
from conda.base.context import context
from conda.common.serialize import json
from conda.exceptions import DryRunExit
from conda.models.channel import Channel

from conda_libmamba_solver import index
from conda_libmamba_solver.index import LibMambaIndexHelper


@pytest.mark.parametrize("subdirs_kind", ("tuple", "generator", "default"))
def test_older_conda_keeps_explicit_channels(mocker, subdirs_kind):
    mocker.patch.object(index, "resolve_channel_relations", None)
    mocker.patch.object(LibMambaIndexHelper, "_init_db")
    mocker.patch.object(LibMambaIndexHelper, "_set_repo_priorities")
    load = mocker.patch.object(LibMambaIndexHelper, "_load_channels", return_value=[])
    channels = [Channel("https://example.org/alpha"), Channel("https://example.org/beta")]
    subdirs = None if subdirs_kind == "default" else ("noarch",)
    if subdirs_kind == "generator":
        subdirs = iter(subdirs)
    helper = LibMambaIndexHelper(channels, subdirs=subdirs)
    assert helper.subdirs == (context.subdirs if subdirs_kind == "default" else ("noarch",))
    assert helper.channels == channels
    assert helper.channels is not channels
    load.assert_called_once()


@pytest.mark.parametrize("use_shards", (False, True))
def test_resolve_before_repository_loading(mocker, use_shards):
    heads = [Channel("https://example.org/alpha")]
    resolved = [Channel("https://example.org/beta"), *heads]
    resolver = mocker.patch.object(index, "resolve_channel_relations", return_value=resolved)
    mocker.patch.object(LibMambaIndexHelper, "_init_db")
    mocker.patch.object(LibMambaIndexHelper, "_set_repo_priorities")
    observed = []
    mocker.patch.object(
        LibMambaIndexHelper,
        "_load_channels",
        autospec=True,
        side_effect=lambda instance: observed.append(tuple(instance.channels)) or [],
    )
    state = mocker.Mock() if use_shards else None
    subset = mocker.Mock() if use_shards else None
    helper = LibMambaIndexHelper(
        heads, subdirs=("noarch",), in_state=state, build_repodata_subset=subset
    )
    assert observed == [tuple(resolved)]
    assert heads == [Channel("https://example.org/alpha")]
    resolver.assert_called_once_with(
        heads, ("noarch",), repodata_fn="repodata.json", use_shards=use_shards
    )
    assert helper.channels == resolved


@pytest.mark.usefixtures("solver_libmamba")
@pytest.mark.parametrize(
    "relation, expected", [("base", "1.0"), ("overrides", "2.0"), (None, "2.0")]
)
def test_libmamba_dry_run_uses_related_channel_priority(
    tmp_path, monkeypatch, conda_cli, relation, expected
):
    if relation is not None and index.resolve_channel_relations is None:
        pytest.skip("conda does not provide channel relation resolution")
    monkeypatch.setenv("CONDA_PKGS_DIRS", str(tmp_path / "pkgs"))
    monkeypatch.setenv("CONDA_ENVS_PATH", str(tmp_path / "envs"))
    monkeypatch.setenv("CONDA_CHANNEL_RELATIONS_MAX_DEPTH", "10")
    monkeypatch.setenv("CONDA_REPODATA_USE_SHARDS", "false")
    for name, version in (("alpha", "2.0"), ("beta", "1.0")):
        subdir = tmp_path / name / "noarch"
        subdir.mkdir(parents=True)
        info = {"subdir": "noarch"}
        if name == "alpha" and relation is not None:
            info["channel_relations"] = {relation: "../beta"}
        package = {
            "name": "example",
            "version": version,
            "build": "0",
            "build_number": 0,
            "depends": [],
            "subdir": "noarch",
            "size": 1,
            "md5": "0" * 32,
            "sha256": "0" * 64,
        }
        (subdir / "repodata.json").write_text(
            json.dumps({"info": info, "packages": {f"example-{version}-0.tar.bz2": package}})
        )
    stdout, _, _ = conda_cli(
        "create",
        "--prefix",
        str(tmp_path / "environment"),
        "--dry-run",
        "--json",
        "--solver",
        "libmamba",
        "--strict-channel-priority",
        "--override-channels",
        "--channel",
        (tmp_path / "alpha").as_uri(),
        "--no-default-packages",
        "example",
        raises=DryRunExit,
    )
    result = json.loads(stdout)
    assert result["success"]
    assert [(record["name"], record["version"]) for record in result["actions"]["LINK"]] == [
        ("example", expected)
    ]


def test_explicit_reload_fetches_after_relation_discovery(tmp_path, mocker):
    channel = Channel("https://example.org/alpha")
    url = channel.urls(True, ("noarch",))[0]
    cached = tmp_path / "repodata.json"
    cached.write_text("{}")
    sd = mocker.Mock(_loaded=True, cache_path_json=str(cached))
    sd.repo_cache.load_state.return_value = {}
    sd.repo_fetch.fetch_latest_path.return_value = (cached, {})
    mocked_subdir_data = mocker.patch.object(index, "SubdirData", return_value=sd)
    mocked_subdir_data._cache_ = {}
    mocker.patch.object(index, "resolve_channel_relations", return_value=(channel,))
    mocker.patch.object(index.context, "offline", False)
    mocker.patch.object(index.context, "use_index_cache", False)
    mocker.patch.object(LibMambaIndexHelper, "_init_db")
    mocker.patch.object(LibMambaIndexHelper, "_set_repo_priorities")
    repo = SimpleNamespace(url_no_cred=url, url_w_cred=url, repo=mocker.Mock())

    def load_channels(helper, *args, **kwargs):
        helper._fetch_one_repodata_json(url)
        return [repo]

    mocker.patch.object(
        LibMambaIndexHelper, "_load_channels", autospec=True, side_effect=load_channels
    )
    helper = LibMambaIndexHelper((channel,), subdirs=("noarch",))
    sd.repo_fetch.fetch_latest_path.assert_not_called()
    helper.reload_channel(channel)
    sd.repo_fetch.fetch_latest_path.assert_called_once()
