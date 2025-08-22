"""
Test downloading shards; subsetting "all possible depedencies" for solver.
"""

from __future__ import annotations

import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import NotRequired, TypedDict
from urllib.parse import urljoin

import conda.gateways.repodata as repodata
import msgpack
import pytest
import zstandard
from conda.base.context import context, reset_context
from conda.core.subdir_data import SubdirData
from conda.gateways.connection.session import get_session
from conda.gateways.repodata import (
    RepodataCache,
    _add_http_value_to_dict,
    conda_http_errors,
)
from conda.models.channel import Channel
from conda.models.records import PackageRecord
from requests import HTTPError, Response  # noqa: TC002

from conda_libmamba_solver.index import LibMambaIndexHelper

from . import shard_cache

HERE = Path(__file__).parent


def unpack_record(record):
    """
    Convert bytes checksums to hex.
    """
    if sha256 := record.get("sha256"):
        record["sha256"] = sha256.hex()
    if md5 := record.get("md5"):
        record["md5"] = md5.hex()
    return record


class RepodataInfo(TypedDict):
    base_url: str  # where packages are stored
    shards_base_url: str  # where shards are stored
    subdir: str


class ShardsIndex(TypedDict):
    info: RepodataInfo
    repodata_version: int
    removed: list[str]
    shards: dict[str, bytes]


Shard = TypedDict("Shard", {"packages": dict[str, dict], "packages.conda": dict[str, dict]})


class ShardLike:
    """
    Present a "classic" repodata.json as per-package shards.
    """

    def __init__(self, repodata: dict):
        all_packages = {
            "packages": repodata.pop("packages", {}),
            "packages.conda": repodata.pop("packages.conda", {}),
        }

        shards = defaultdict(lambda: {"packages": {}, "packages.conda": {}})

        self.repodata = repodata

        for package_group in all_packages:
            for package, record in all_packages[package_group].items():
                name = record["name"]
                shards[name][package_group][package] = record

        self.shards = dict(shards)  # no longer a defaultdict


class Shards:
    def __init__(self, shards_index: ShardsIndex, url: str):
        """
        Args:
            shards_index: raw parsed msgpack dict
            url: URL of repodata_shards.msgpack.zst
        """
        self.shards_index = shards_index
        self.url = url

    def shard_url(self, package: str):
        """
        Return shard URL for a given package.

        Raise KeyError if package is not in the index.
        """
        shard_name = f"{self.shards[package].hex()}.msgpack.zst"
        # "Individual shards are stored under the URL <shards_base_url><sha256>.msgpack.zst"
        return urljoin(self.url, f"{self.shards_index['info']['shards_base_url']}{shard_name}")

    @property
    def shards(self):
        return self.shards_index["shards"]


def repodata_shards(url, cache: RepodataCache) -> bytes:
    """
    Fetch shards index with cache.

    Update cache state.

    Return shards data, either newly fetched or from cache.
    """
    session = get_session(url)  # does it care if repodata filename is included, and unexpected?

    state = cache.state
    headers = {}
    etag = state.etag
    last_modified = state.mod
    if etag:
        headers["If-None-Match"] = str(etag)
    if last_modified:
        headers["If-Modified-Since"] = str(last_modified)
    filename = "repodata_shards.msgpack.zst"

    with conda_http_errors(url, filename):
        timeout = (
            context.remote_connect_timeout_secs,
            context.remote_read_timeout_secs,
        )
        response: Response = session.get(
            url, headers=headers, proxies=session.proxies, timeout=timeout
        )
        response.raise_for_status()
        response_bytes = response.content

    if response.status_code == 304:
        # should we save cache-control to state here to put another n
        # seconds on the "make a remote request" clock and/or touch cache
        # mtime
        return cache.cache_path_shards.read_bytes()

    # We no longer add these tags to the large `resp.content` json
    saved_fields = {repodata.URL_KEY: url}
    _add_http_value_to_dict(response, "Etag", saved_fields, repodata.ETAG_KEY)
    _add_http_value_to_dict(response, "Last-Modified", saved_fields, repodata.LAST_MODIFIED_KEY)
    _add_http_value_to_dict(response, "Cache-Control", saved_fields, repodata.CACHE_CONTROL_KEY)

    state.update(saved_fields)

    # should we return the response and let caller save cache data to state?
    return response_bytes


def fetch_shards(sd: SubdirData) -> Shards | None:
    """
    Check a SubdirData's URL for shards.
    Return shards index bytes from cache or network.
    Return None if not found; caller should fetch normal repodata.
    """

    fetch = sd.repo_fetch
    cache = fetch.repo_cache
    # cache.load_state() will clear the file on JSONDecodeError but cache.load()
    # will raise the exception
    cache.load_state(binary=True)
    cache_state = cache.state

    if cache_state.should_check_format("shards"):
        try:
            # look for shards index
            shards_index_url = f"{sd.url_w_subdir}/repodata_shards.msgpack.zst"
            found = repodata_shards(shards_index_url, cache)
            cache_state.set_has_format("shards", True)
            # this will also set state["refresh_ns"] = time.time_ns(); we could
            # call cache.refresh() if we got a 304 instead:
            cache.save(found)

            # basic parse (move into caller?)
            shards_index: ShardsIndex = msgpack.loads(zstandard.decompress(found))  # type: ignore
            shards = Shards(shards_index, shards_index_url)
            return shards

        except (HTTPError, repodata.RepodataIsEmpty):
            # fetch repodata.json / repodata.json.zst instead
            cache_state.set_has_format("shards", False)
            cache.refresh(refresh_ns=1)  # expired but not falsy

    return None


@pytest.fixture
def conda_no_token(monkeypatch):
    """
    Reset token to avoid being logged in. e.g. the testing channel doesn't understand them.
    """
    monkeypatch.setenv("CONDA_TOKEN", "")
    reset_context()


def test_shards(conda_no_token):
    """
    Test basic shard fetch etc.
    """
    in_state = pickle.loads((HERE / "data" / "in_state.pickle").read_bytes())
    print(in_state)

    channels = [
        Channel.from_url(f"https://conda.anaconda.org/conda-forge-sharded/{subdir}")
        for subdir in context.subdirs
    ]

    # Would eagerly download repodata.json.zst for all channels
    # helper = LibMambaIndexHelper(
    #     channels,
    #     (),  # subdirs
    #     "repodata.json",
    #     installed,
    #     (),  # pkgs_dirs to load packages locally when offline
    #     in_state=in_state,
    # )

    # accessing "helper.repos" downloads repodata.json in the traditional way
    # for all channels:
    # print(helper.repos)

    subdir_data = SubdirData(channels[0])
    found = fetch_shards(subdir_data)
    assert found, f"Shards not found for {channels[0]}"

    for package in found.shards:
        shard_url = found.shard_url(package)
        assert shard_url.startswith("http")  # or channel url or shards_base_url

    session = get_session(
        subdir_data.url_w_subdir
    )  # XXX session could be different based on shards_base_url and different than the packages base_url
    # cache_path_base includes a per-repository hash
    shards_cache = shard_cache.ShardCache(Path(subdir_data.cache_path_base).parent)

    # download or fetch-from-cache a random set of shards

    for package in random.choices([*found.shards.keys()], k=16):
        miss = False
        shard_url = found.shard_url(package)
        shard = shards_cache.retrieve(shard_url)
        if not shard:
            miss = True
            raw_shard = session.get(shard_url).content
            shard: Shard = msgpack.loads(zstandard.decompress(raw_shard))
            shards_cache.insert(shard_url, package, raw_shard)
        print(miss, shard)

        mentioned_in_shard = shard_mentioned_packages(shard)
        print(mentioned_in_shard)

    """
    >>> in_state.requested
    mappingproxy({'ipython': MatchSpec("ipython")})
    >>> for x in in_state.installed.values(): print(x)
    conda-forge/osx-arm64::bzip2==1.0.8=h99b78c6_7
    conda-forge/noarch::ca-certificates==2025.7.14=hbd8a1cb_0
    conda-forge/osx-arm64::icu==75.1=hfee45f7_0
    conda-forge/osx-arm64::libexpat==2.7.1=hec049ff_0
    conda-forge/osx-arm64::libffi==3.4.6=h1da3d7d_1
    conda-forge/osx-arm64::liblzma==5.8.1=h39f12f2_2
    conda-forge/osx-arm64::libmpdec==4.0.0=h5505292_0
    conda-forge/osx-arm64::libsqlite==3.50.3=h4237e3c_1
    conda-forge/osx-arm64::libzlib==1.3.1=h8359307_2
    conda-forge/osx-arm64::ncurses==6.5=h5e97a16_3
    conda-forge/osx-arm64::openssl==3.5.1=h81ee809_0
    conda-forge/noarch::pip==25.1.1=pyh145f28c_0
    conda-forge/osx-arm64::python==3.13.5=hf3f3da0_102_cp313
    conda-forge/noarch::python_abi==3.13=8_cp313
    conda-forge/osx-arm64::readline==8.2=h1d1bf99_2
    conda-forge/osx-arm64::tk==8.6.13=h892fb3f_2
    conda-forge/noarch::tzdata==2025b=h78e105d_0
    """


def test_shards_2(conda_no_token):
    """
    Test all channels fetch.
    """
    channels = list(context.default_channels)
    print(channels)
    # is (pkgs/main, pkgs/r) in mine

    # Channel('pkgs/main').url()
    # 'https://repo.anaconda.com/pkgs/main/osx-arm64'
    # context.default_channels[0].url()
    # 'https://repo.anaconda.com/pkgs/main/osx-arm64'
    # Channel('main').url()

    # state to initiate a solve
    in_state = pickle.loads((HERE / "data" / "in_state.pickle").read_bytes())
    print(in_state)

    channels.append(Channel("conda-forge-sharded"))

    installed = (
        *in_state.installed.values(),
        *in_state.virtual.values(),
    )

    channel_data = {}
    for channel in channels:
        for url in Channel(channel).urls(True, context.subdirs):
            subdir_data = SubdirData(Channel(url))
            found = fetch_shards(subdir_data)
            if not found:
                repodata_json, _ = subdir_data.repo_fetch.fetch_latest_parsed()
                found = ShardLike(repodata_json)
            channel_data[url] = found

    print(channel_data)


def shard_mentioned_packages(shard: Shard):
    """
    Return all dependencies mentioned in a shard, including the shard's own
    package name.

    Includes virtual packages.
    """
    # XXX filter by package name for the possibility of a shard with multiple
    # (small) packages
    mentioned = set()
    for package in (*shard["packages"].values(), *shard["packages.conda"].values()):
        # to go faster, don't use PackageRecord, record.combined_depends, or
        # MatchSpec
        record = PackageRecord(**unpack_record(package))
        mentioned.add(record.name)
        mentioned.update(spec.name for spec in record.combined_depends)
    return mentioned


def test_shard_cache(tmp_path: Path):
    cache = shard_cache.ShardCache(tmp_path)

    fake_shard = {"foo": "bar"}
    cache.insert("foobar", "foobar_package", zstandard.compress(msgpack.dumps(fake_shard)))

    data = cache.retrieve("foobar")
    assert data == fake_shard
    assert data is not fake_shard

    data2 = cache.retrieve("notfound")
    assert data2 is None

    assert (tmp_path / shard_cache.SHARD_CACHE_NAME).exists()


def test_shardlike():
    """
    ShardLike class presents repodata.json as shards in a way that is suitable
    for our subsetting algorithm.
    """
    repodata = json.loads(
        (Path(__file__).parent / "data" / "mamba_repo" / "noarch" / "repodata.json").read_text()
    )

    # make fake packages
    for n in range(10):
        for m in range(n):
            repodata["packages"][f"test{n}{m}.tar.bz2"] = {"name": f"test{n}"}
            repodata["packages.conda"][f"test{n}{m}.tar.bz2"] = {"name": f"test{n}"}

    as_shards = ShardLike(repodata)

    assert len(as_shards.repodata)
    assert len(as_shards.shards)

    assert sorted(as_shards.shards["test4"]["packages"].keys()) == [
        "test40.tar.bz2",
        "test41.tar.bz2",
        "test42.tar.bz2",
        "test43.tar.bz2",
    ]
