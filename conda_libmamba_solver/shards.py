"""
Seek out, fetch, and subset sharded repodata.

When shards are in use, we can do minimal parsing of the repodata to fetch only
the subset that can be used for the solver. We only need to determine the
package names and the package names of their dependencies, and could skip
creating inefficient PackageRecord instances.

For each installed or requested package, find its dependencies by package name.
Add those package names to a set of package names to collect. Each channel
collects the metadata for the named packages, and adds those packages'
dependencies to the package names to collect.

Repeat until no new packages have been discovered.

Serialize repodata.json with only the package name visited in the prior step.
Send to the solver.
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

from . import shard_cache  # type: ignore

if TYPE_CHECKING:
    from collections.abc import Iterable

    from conda.gateways.repodata import (
        RepodataCache,
    )

    from ..conda_libmamba_solver.shard_cache import Shard

from typing import TypedDict


class RepodataInfo(TypedDict):  # noqa: F811
    base_url: str  # where packages are stored
    shards_base_url: str  # where shards are stored
    subdir: str


def maybe_unpack_record(record):
    """
    Convert bytes checksums to hex; leave unchanged if already str.
    """
    for hash_type in "sha256", "md5":
        if hash_value := record.get(hash_type):
            if isinstance(hash_value, bytes):
                record[hash_type] = hash_value.hex()
    return record


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
        record = PackageRecord(**maybe_unpack_record(package))
        mentioned.add(record.name)
        mentioned.update(spec.name for spec in record.combined_depends)
    return mentioned


class ShardsIndex(TypedDict):
    info: RepodataInfo
    repodata_version: int
    removed: list[str]
    shards: dict[str, bytes]


class ShardLike:
    """
    Present a "classic" repodata.json as per-package shards.
    """

    def __init__(self, repodata: dict):
        all_packages = {
            "packages": repodata.pop("packages", {}),
            "packages.conda": repodata.pop("packages.conda", {}),
        }
        self.repodata = repodata  # without packages, packges.conda

        shards = defaultdict(lambda: {"packages": {}, "packages.conda": {}})

        for package_group in all_packages:
            for package, record in all_packages[package_group].items():
                name = record["name"]
                shards[name][package_group][package] = record

        # defaultdict behavior no longer wanted
        self.shards: dict[str, Shard] = dict(shards)  # type: ignore

        # used to write out repodata subset
        self.visited_shards: dict[str, Shard] = {}

    def __contains__(self, package):
        return package in self.shards

    def fetch_shard(self, package: str, session) -> Shard:
        """
        "Fetch" an individual shard.

        Update self.visited_shards with all not-None packages.

        Raise KeyError if package is not in the index.
        """
        shard = self.shards[package]
        self.visited_shards[package] = shard
        return shard

    def fetch_shards(self, packages: list[str], session) -> dict[str, Shard]:
        """
        Fetch multiple shards in one go.

        Update self.visited_shards with all not-None packages.
        """
        return {package: self.fetch_shard(package, session) for package in packages}

    def build_repodata(self):
        """
        Return monolithic repodata including all visited shards.
        """
        repodata = self.repodata.copy()
        repodata.update({"packages": {}, "packages.conda": {}})
        for package, shard in self.visited_shards.items():
            for package_group in ("packages", "packages.conda"):
                repodata[package_group].update(shard[package_group])
        return repodata


class Shards(ShardLike):
    def __init__(self, shards_index: ShardsIndex, url: str, shards_cache: shard_cache.ShardCache):
        """
        Args:
            shards_index: raw parsed msgpack dict
            url: URL of repodata_shards.msgpack.zst
        """
        self.shards_index = shards_index
        self.url = url
        self.shards_cache = shards_cache

        self.repodata = {k: v for k, v in self.shards_index.items() if k not in ("shards",)}

        # used to write out repodata subset
        self.visited_shards: dict[str, Shard] = {}

    def __contains__(self, package):
        return package in self.packages_index

    @property
    def packages_index(self):
        return self.shards_index["shards"]

    def shard_url(self, package: str):
        """
        Return shard URL for a given package.

        Raise KeyError if package is not in the index.
        """
        shard_name = f"{self.packages_index[package].hex()}.msgpack.zst"
        # "Individual shards are stored under the URL <shards_base_url><sha256>.msgpack.zst"
        return urljoin(self.url, f"{self.shards_index['info']['shards_base_url']}{shard_name}")

    def fetch_shard(self, package: str, session) -> Shard:
        """
        Fetch an individual shard or retrieve from cache.

        Raise KeyError if package is not in the index.
        """

        shard_url = self.shard_url(package)
        # XXX do we call this with the same shard, decompressing twice, in a single instance?
        shard_or_none = self.shards_cache.retrieve(shard_url)
        if shard_or_none:
            self.visited_shards[package] = shard_or_none
            return shard_or_none
        else:
            raw_shard = session.get(shard_url).content
            # ensure it is real msgpack+zstd before inserting into cache
            shard: Shard = msgpack.loads(zstandard.decompress(raw_shard))  # type: ignore
            self.shards_cache.insert(shard_url, package, raw_shard)
            self.visited_shards[package] = shard
            return shard

    def fetch_shards(self, packages: Iterable[str], session) -> dict[str, Shard]:
        result = {}

        def fetch(s, package, url):
            b1 = time.time_ns()
            data = s.get(url).content
            e1 = time.time_ns()
            print(f"{(e1 - b1) / 1e9}s", package, url)
            return data

        shard_urls = {package: self.shard_url(package) for package in packages}

        cached = self.shards_cache.retrieve_multiple(list(shard_urls.values()))
        for package in cached:
            result[package] = cached[package]

        # beneficial to have thread pool larger than requests' default 10 max
        # connections per session. There is "context.repodata_threads" but it's
        # None in the REPL.
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = {
                executor.submit(fetch, session, package, url): (url, package)
                for package, url in shard_urls
                if package not in result
            }
            # May be inconvenient to cancel a large number of futures. Also, ctrl-C doesn't work reliably out of the box.
            for future in concurrent.futures.as_completed(futures):
                print(".", futures[future])
                data = future.result()
                url, package = futures[future]
                # XXX catch exception / 404 / whatever
                result[package] = msgpack.loads(zstandard.decompress(data))
                self.shards_cache.insert(url, package, data)

        return result


def repodata_shards(url, cache: RepodataCache) -> bytes:
    """
    Fetch shards index with cache.

    Update cache state.

    Return shards data, either newly fetched or from cache.
    """
    session = get_session(url)

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
            shards = Shards(
                shards_index,
                shards_index_url,
                shard_cache.ShardCache(Path(repodata.create_cache_dir())),
            )
            return shards

        except (HTTPError, repodata.RepodataIsEmpty):
            # fetch repodata.json / repodata.json.zst instead
            cache_state.set_has_format("shards", False)
            cache.refresh(refresh_ns=1)  # expired but not falsy

    return None
