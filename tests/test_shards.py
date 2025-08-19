"""
Test downloading shards; subsetting "all possible depedencies" for solver.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import NotRequired, TypedDict
from urllib.parse import urljoin

import conda.gateways.repodata as repodata
import msgpack
import zstandard
from conda.base.context import context, reset_context
from conda.core.subdir_data import SubdirData
from conda.gateways.connection.session import get_session
from conda.gateways.repodata import (
    RepodataCache,
    RepodataState,
    _add_http_value_to_dict,
    conda_http_errors,
)
from conda.models.channel import Channel
from requests import HTTPError, Response  # noqa: TC002

from conda_libmamba_solver.index import LibMambaIndexHelper

HERE = Path(__file__).parent


class MaybeSharded:
    def __init__(self, channel):
        self.channel = channel

    pass


class RepodataInfo(TypedDict):
    base_url: str  # where packages are stored
    shards_base_url: str  # where shards are stored
    subdir: str


class ShardsIndex(TypedDict):
    info: RepodataInfo
    repodata_version: int
    removed: list[str]
    shards: dict[str, bytes]


class Shards:
    def __init__(self, shards_index: ShardsIndex, url: str):
        self.shards_index = shards_index
        self.url = url
        assert self.url.endswith("/"), (
            "channel url must end with / although this would work fine if we gave the entire shards index filename as self.url"
        )

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


def repodata_shards(url, cache: RepodataCache) -> bytes | None:
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


def traverse_shards(
    channels: list[Channel], matchspecs, root_packages
) -> bytes | SubdirData | None:
    assert len(channels)  # > 0

    # Should do this for all channels
    channel = channels[0]

    # Use SubdirData to get at the correct fetch, cache objects
    sd = SubdirData(channel)
    fetch = sd.repo_fetch
    cache = fetch.repo_cache
    # cache.load_state() will clear the file on JSONDecodeError but cache.load()
    # will raise the exception
    cache.load_state(binary=True)
    cache_state = cache.state
    print(cache)

    # cache.load(binary=True) looks for <hash>.msgpack.zst

    # repodata, <RepodataState> object = sd.fetch_latest_parsed()

    if cache_state.should_check_format("shards"):
        try:
            # look for shards index
            shards_index_url = f"{sd.url_w_subdir}/repodata_shards.msgpack.zst"
            found = repodata_shards(shards_index_url, cache)
            print(len(found)) if found else print("No shards for", shards_index_url)
            if found:
                cache_state.set_has_format("shards", True)
                cache.save(found)
            # indicate that we just found shards not json
        except HTTPError as e:
            # fetch repodata.json / repodata.json.zst instead
            cache_state.set_has_format("shards", False)
            cache.refresh(refresh_ns=1)  # expired but not falsy

            found = sd.load()

    return found


def test_shards():
    """
    Reduce full index to only packages needed.
    """
    os.environ["CONDA_TOKEN"] = ""  # test channel doesn't understand tokens
    reset_context()

    in_state = pickle.loads((HERE / "data" / "in_state.pickle").read_bytes())
    print(in_state)

    channels = [
        Channel.from_url(f"https://conda.anaconda.org/conda-forge-sharded/{subdir}")
        for subdir in context.subdirs
    ]

    installed = (
        *in_state.installed.values(),
        *in_state.virtual.values(),
    )

    helper = LibMambaIndexHelper(
        channels,
        (),  # subdirs
        "repodata.json",
        installed,
        (),  # pkgs_dirs to load packages locally when offline
        in_state=in_state,
    )

    # accessing "helper.repos" downloads repodata.json in the traditional way
    # for all channels:
    # print(helper.repos)

    found = traverse_shards(channels, in_state.requested, installed)

    if not isinstance(found, bytes):
        return

    shards_index: ShardsIndex = msgpack.loads(zstandard.decompress(found))  # type: ignore
    # ensure url ends with slash, for urljoin
    channel_url = channels[0].url()
    assert isinstance(channel_url, str)
    # ensure ends with / for later urljoin
    channel_url = channel_url.rstrip("/") + "/"
    shards = Shards(shards_index, channel_url)

    for key in shards.shards.keys():
        assert shards.shard_url(key).startswith("http")

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
