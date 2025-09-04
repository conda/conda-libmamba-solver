# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Models for sharded repodata, and to make monolithic repodata look like sharded
repodata.
"""

from __future__ import annotations

import concurrent.futures
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict
from urllib.parse import urljoin

import conda.gateways.repodata as repodata
import msgpack
import zstandard
from conda.base.context import context
from conda.gateways.connection.session import get_session
from conda.gateways.repodata import (
    _add_http_value_to_dict,
    conda_http_errors,
)
from conda.models.records import PackageRecord
from requests import HTTPError

from . import shard_cache

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import NotRequired

    from conda.core.subdir_data import SubdirData
    from conda.gateways.repodata import RepodataCache
    from requests import Response, Session


class PackageRecordDict(TypedDict):
    """
    Basic package attributes that this module cares about.
    """

    name: str
    sha256: NotRequired[str | bytes]
    md5: NotRequired[str | bytes]


# in this style because "packages.conda" is not a Python identifier
Shard = TypedDict(
    "Shard",
    {"packages": dict[str, PackageRecordDict], "packages.conda": dict[str, PackageRecordDict]},
)


class RepodataInfo(TypedDict):  # noqa: F811
    base_url: str  # where packages are stored
    shards_base_url: str  # where shards are stored
    subdir: str


class RepodataDict(Shard):
    """
    Packages plus info.
    """

    info: RepodataInfo


class ShardsIndex(TypedDict):
    """
    Shards index as deserialized from repodata_shards.msgpack.zst
    """

    info: RepodataInfo
    repodata_version: int
    removed: list[str]
    shards: dict[str, bytes]


def maybe_unpack_record(record: PackageRecordDict):
    """
    Convert bytes checksums to hex; leave unchanged if already str.
    """
    for hash_type in "sha256", "md5":
        if hash_value := record.get(hash_type):
            if isinstance(hash_value, bytes):
                record[hash_type] = hash_value.hex()
    return record


def shard_mentioned_packages(shard: Shard) -> set[str]:
    """
    Return all dependency names mentioned in a shard, including the shard's own
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


class ShardLike:
    """
    Present a "classic" repodata.json as per-package shards.
    """

    def __init__(self, repodata: RepodataDict, url: str = ""):
        """
        url: affects the repr but not the functionality of this class.
        """
        all_packages = {
            "packages": repodata["packages"],
            "packages.conda": repodata["packages.conda"],
        }
        repodata.pop("packages")
        repodata.pop("packages.conda")
        self.repodata_no_packages = repodata  # without packages, packages.conda
        self.url = url

        shards = defaultdict(lambda: {"packages": {}, "packages.conda": {}})

        for package_group in all_packages:
            for package, record in all_packages[package_group].items():
                name = record["name"]
                shards[name][package_group][package] = record

        # defaultdict behavior no longer wanted
        self.shards: dict[str, Shard] = dict(shards)  # type: ignore

        # used to write out repodata subset
        self.visited: dict[str, Shard | None] = {}

    def __repr__(self):
        return self.url.join(super().__repr__().split(maxsplit=1))

    def __contains__(self, package_name: str) -> bool:
        return package_name in self.shards

    def fetch_shard(self, package: str, session: Session) -> Shard:
        """
        "Fetch" an individual shard.

        Update self.visited with all not-None packages.

        Raise KeyError if package is not in the index.
        """
        shard = self.shards[package]
        self.visited[package] = shard
        return shard

    def fetch_shards(self, packages: Iterable[str], session: Session) -> dict[str, Shard]:
        """
        Fetch multiple shards in one go.

        Update self.visited with all not-None packages.
        """
        return {package: self.fetch_shard(package, session) for package in packages}

    def build_repodata(self) -> RepodataDict:
        """
        Return monolithic repodata including all visited shards.
        """
        repodata = self.repodata_no_packages.copy()
        repodata.update({"packages": {}, "packages.conda": {}})
        for package, shard in self.visited.items():
            if shard is None:
                continue  # recorded visited but not available shards
            for package_group in ("packages", "packages.conda"):
                repodata[package_group].update(shard[package_group])
        return repodata


FETCHED_THIS_PROCESS = set()


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
        self.visited: dict[str, Shard | None] = {}

    def __contains__(self, package):
        return package in self.packages_index

    @property
    def packages_index(self):
        return self.shards_index["shards"]

    def shard_url(self, package: str) -> str:
        """
        Return shard URL for a given package.

        Raise KeyError if package is not in the index.
        """
        shard_name = f"{self.packages_index[package].hex()}.msgpack.zst"
        # "Individual shards are stored under the URL <shards_base_url><sha256>.msgpack.zst"
        return urljoin(self.url, f"{self.shards_index['info']['shards_base_url']}{shard_name}")

    def fetch_shard(self, package: str, session: Session) -> Shard:
        """
        Fetch an individual shard.

        Raise KeyError if package is not in the index.
        """

        shard_url = self.shard_url(package)
        # XXX do we call this with the same shard, decompressing twice, in a single instance?
        shard_or_none = self.shards_cache.retrieve(shard_url)
        if shard_or_none:
            self.visited[package] = shard_or_none
            return shard_or_none
        else:
            raw_shard = session.get(shard_url).content
            # ensure it is real msgpack+zstd before inserting into cache
            shard: Shard = msgpack.loads(zstandard.decompress(raw_shard))  # type: ignore
            self.shards_cache.insert(shard_cache.AnnotatedRawShard(shard_url, package, raw_shard))
            self.visited[package] = shard
            return shard

    def fetch_shards(self, packages: Iterable[str], session: Session) -> dict[str, Shard]:
        """
        Return mapping of *package names* to Shard for given packages.
        """
        result = {}

        def fetch(s, url, package):
            if url in FETCHED_THIS_PROCESS:
                print("Already got", url)
                raise RuntimeError("Can't fetch same url twice")
            FETCHED_THIS_PROCESS.add(url)
            b1 = time.time_ns()
            data = s.get(url).content
            e1 = time.time_ns()
            print(f"Fetch took {(e1 - b1) / 1e9}s", package, url)
            return shard_cache.AnnotatedRawShard(url=url, package=package, compressed_shard=data)

        packages = sorted(list(packages))
        urls_packages = {self.shard_url(package): package for package in packages}

        cached = self.shards_cache.retrieve_multiple(sorted(urls_packages))
        for url, shard in cached.items():
            package = urls_packages[url]
            assert not package.startswith(("https://", "http://"))
            result[package] = shard

        # beneficial to have thread pool larger than requests' default 10 max
        # connections per session. There is "context.repodata_threads" but it's
        # None in the REPL.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = {
                executor.submit(fetch, session, url, package): (url, package)
                for url, package in urls_packages.items()
                if package not in result
            }
            # May be inconvenient to cancel a large number of futures. Also, ctrl-C doesn't work reliably out of the box.
            for future in concurrent.futures.as_completed(futures):
                print(".", futures[future])
                fetch_result = future.result()
                # XXX catch exception / 404 / whatever
                result[fetch_result.package] = msgpack.loads(
                    zstandard.decompress(fetch_result.compressed_shard)
                )
                try:
                    package_names = [
                        p["name"]
                        for p in (
                            *result[fetch_result.package]["packages"].values(),
                            *result[fetch_result.package]["packages.conda"].values(),
                        )
                    ]
                    print("expected", fetch_result.package, "actual", set(package_names))
                except (AttributeError, KeyError):
                    print("oops")
                self.shards_cache.insert(fetch_result)

        self.visited.update(result)

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
