"""
Test downloading shards; subsetting "all possible depedencies" for solver.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

from conda.base.context import context, reset_context
from conda.core.subdir_data import SubdirData
from conda.gateways.connection.session import get_session
from conda.gateways.repodata import RepodataState, _add_http_value_to_dict, conda_http_errors
from conda.models.channel import Channel
from requests import Response  # noqa: TC002

from conda_libmamba_solver.index import LibMambaIndexHelper

HERE = Path(__file__).parent


class MaybeSharded:
    def __init__(self, channel):
        self.channel = channel

    pass


def repodata_shards(url, state: RepodataState) -> bytes | None:
    session = get_session(url)  # does it care if repodata filename is included, and unexpected?

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

    if response.status_code == 304:
        # should we save cache-control to state here to put another n
        # seconds on the "make a remote request" clock and/or touch cache
        # mtime
        raise NotImplementedError("TODO implement cache")

    response_bytes = response.content

    # We no longer add these tags to the large `resp.content` json
    saved_fields = {"_url": url}
    _add_http_value_to_dict(response, "Etag", saved_fields, "_etag")
    _add_http_value_to_dict(response, "Last-Modified", saved_fields, "_mod")
    _add_http_value_to_dict(response, "Cache-Control", saved_fields, "_cache_control")

    state.update(saved_fields)

    # should we return the response and let caller save cache data to state?
    return response_bytes


def traverse_shards(channels: list[Channel], matchspecs, root_packages):
    assert len(channels)  # > 0
    channel = channels[0]
    sd = SubdirData(channel)
    fetch = sd.repo_fetch
    cache = fetch.repo_cache
    cache_state = (
        cache.load_state()
    )  # without reading repodata.json or repodata_shards.msgpack.zst
    print(cache)

    # cache.load(binary=True) looks for <hash>.msgpack.zst

    # repodata, <RepodataState> object = sd.fetch_latest_parsed()

    if cache_state.should_check_format("shards"):
        try:
            shards_index_url = f"{sd.url_w_subdir}/repodata_shards.msgpack.zst"
            found = repodata_shards(shards_index_url, cache_state)
            print(len(found)) if found else print("No shards for", shards_index_url)
            if found:
                cache_state.set_has_format("shards", True)
                cache.save(found)
        except Exception as e:
            cache_state.set_has_format("shards", False)
            cache.refresh(refresh_ns=1)  # expired but not falsy

            found = sd.load()

    print(found)


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

    print(helper.repos)

    traverse_shards(channels, in_state.requested, installed)

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
