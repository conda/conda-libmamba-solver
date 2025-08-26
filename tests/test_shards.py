"""
Test downloading shards; subsetting "all possible depedencies" for solver.
"""

from __future__ import annotations

import concurrent.futures
import json
import pickle
import random
import textwrap
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict
from urllib.parse import urljoin

import conda.gateways.repodata as repodata
import msgpack
import pytest
import zstandard
from conda.base.context import context, reset_context
from conda.core.subdir_data import SubdirData
from conda.gateways.connection.session import get_session
from conda.gateways.repodata import (
    _add_http_value_to_dict,
    conda_http_errors,
)
from conda.models.channel import Channel
from conda.models.records import PackageRecord
from requests import HTTPError, Response

from conda_libmamba_solver import shard_cache, shards
from conda_libmamba_solver.shards import (
    RepodataInfo,
    ShardLike,
    fetch_shards,
    shard_mentioned_packages,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from conda.gateways.repodata import (
        RepodataCache,
    )

    from ..conda_libmamba_solver.shard_cache import Shard

HERE = Path(__file__).parent


@pytest.fixture
def conda_no_token(monkeypatch: pytest.MonkeyPatch):
    """
    Reset token to avoid being logged in. e.g. the testing channel doesn't understand them.
    """
    monkeypatch.setenv("CONDA_TOKEN", "")
    reset_context()


def test_shards(conda_no_token: None):
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

    for package in found.packages_index:
        shard_url = found.shard_url(package)
        assert shard_url.startswith("http")  # or channel url or shards_base_url

    session = get_session(
        subdir_data.url_w_subdir
    )  # XXX session could be different based on shards_base_url and different than the packages base_url

    # download or fetch-from-cache a random set of shards

    for package in random.choices([*found.packages_index.keys()], k=16):
        shard = found.fetch_shard(package, session)
        print(shard)

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


def test_shards_2(conda_no_token: None):
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

    channel_data: dict[str, ShardLike] = {}
    for channel in channels:
        for channel_url in Channel(channel).urls(True, context.subdirs):
            subdir_data = SubdirData(Channel(channel_url))
            found = fetch_shards(subdir_data)
            if not found:
                repodata_json, _ = subdir_data.repo_fetch.fetch_latest_parsed()
                found = ShardLike(repodata_json)
            channel_data[channel_url] = found

    print(channel_data)

    shards_to_get = set(package.name for package in installed)

    # what we want to install
    # shards_to_get.update(in_state.requested)
    shards_to_get.add("twine")

    shards_to_get_more = set(
        package.name for root in installed for package in root.combined_depends
    )

    print(
        "Are the installed packages the same as all their dependencies?",
        shards_to_get == shards_to_get_more,
        f"{len(shards_to_get)} from installed",
        f"{len(shards_to_get_more)} from installed's dependencies",  # includes virtual packages, doesn't matter
    )

    # e.g. shards_to_get_more - shards_to_get
    # {'expat', 'xz', 'zlib'}
    # shards_to_get - shards_to_get_more
    # {'__conda', '__archspec'}

    shards_to_get |= shards_to_get_more

    shards_have = {
        url: {} for url in channel_data
    }  # mapping between URL and shards fetched or attempted-to-fetch
    iteration = 0
    waste = 0

    time_start = time.monotonic()
    while shards_to_get:
        iteration_start = time.monotonic()
        print(f"Seek {len(shards_to_get)} shards in iteration {iteration}:")
        print("\n".join(textwrap.wrap(" ".join(sorted(shards_to_get)))))
        new_shards_to_get = set()
        for channel_url, shardlike in channel_data.items():
            session = get_session(channel_url)  # XXX inefficient but it has a @cache decorator
            new_shards_to_get = set()
            # packages available in the channel that we don't already have
            shards_to_fetch = set(
                package
                for package in shards_to_get
                if package in shardlike
                and package
                not in shardlike.visited_shards  # may make sense to update visited_shards with negative matches
            )
            shards_fetched_serially = set()
            for package in shards_to_get:
                if package not in shards_have[channel_url]:  # XXX also inefficient
                    if package in shardlike:
                        new_shard = shardlike.fetch_shard(package, session)
                        new_shards_to_get.update(shard_mentioned_packages(new_shard))
                        shards_have[channel_url][package] = new_shard
                        shards_fetched_serially.add(package)
                    else:
                        shards_have[channel_url][package] = None
                else:
                    waste += 1
            print("To-fetch same both ways?", shards_fetched_serially == shards_to_fetch)

        shards_to_get = new_shards_to_get
        iteration_end = time.monotonic()
        print(f"Iteration {iteration} took {iteration_end - iteration_start:.2f}s")
        iteration += 1
    time_end = time.monotonic()

    relevant_packages = [set(value) for value in shards_have.values()]
    assert all(len(x) == len(relevant_packages[0]) for x in relevant_packages)
    print(f"Sought data for the following {len(relevant_packages[0])} packages:")
    print("\n".join(textwrap.wrap(" ".join(sorted(relevant_packages[0])))))
    print(f"Wasted {waste} inner loop iterations")
    print(f"Took {time_end - time_start:.2f}s")

    # Now write out shards_have packages that are not None, as small
    # repodata.json for the solver.


def test_shard_cache(tmp_path: Path):
    cache = shard_cache.ShardCache(tmp_path)

    fake_shard = {"foo": "bar"}
    cache.insert("foobar", "foobar_package", zstandard.compress(msgpack.dumps(fake_shard)))  # type: ignore

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
