# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Models for sharded repodata, and to make monolithic repodata look like sharded
repodata.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urljoin

import conda.gateways.repodata
import msgpack
import zstandard
from conda.base.context import context
from conda.gateways.connection.session import get_session
from conda.gateways.repodata import (
    _add_http_value_to_dict,
    conda_http_errors,
)
from conda.models.records import PackageRecord
from libmambapy.bindings import specs
from requests import HTTPError

from . import shards_cache

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable, KeysView

    from conda.core.subdir_data import SubdirData
    from conda.gateways.repodata import RepodataCache
    from requests import Response

    from conda_libmamba_solver.shards_typing import RepodataDict, ShardsIndexDict

    from .shards_typing import PackageRecordDict, ShardDict


def ensure_hex_hash(record: PackageRecordDict):
    """
    Convert bytes checksums to hex; leave unchanged if already str.
    """
    for hash_type in "sha256", "md5":
        if hash_value := record.get(hash_type):
            if isinstance(hash_value, bytes):
                record[hash_type] = hash_value.hex()
    return record


def shard_mentioned_packages(shard: ShardDict) -> set[str]:
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
        record = PackageRecord(**ensure_hex_hash(package))
        mentioned.add(record.name)
        mentioned.update(spec.name for spec in record.combined_depends)
    return mentioned


def shard_mentioned_packages_2(shard: ShardDict) -> Iterable[str]:
    """
    Return all dependency names mentioned in a shard, not including the shard's
    own package name.
    """
    unique_specs = set()
    for package in (*shard["packages"].values(), *shard["packages.conda"].values()):
        ensure_hex_hash(package)  # otherwise we could do this at serialization
        for spec in (*package.get("depends", ()),):  # , *package.get("constrains", ())):
            if spec in unique_specs:
                continue
            unique_specs.add(spec)
            parsed_spec = specs.MatchSpec.parse(spec)
            name = str(parsed_spec.name)
            yield name  # not much improvement from only yielding unique names


class ShardLike:
    """
    Present a "classic" repodata.json as per-package shards.
    """

    def __init__(self, repodata: RepodataDict, url: str = ""):
        """
        url: affects the repr but not the functionality of this class.
        """
        self.repodata_no_packages: RepodataDict = {
            **repodata,
            "packages": {},
            "packages.conda": {},
        }
        all_packages = {
            "packages": repodata.get("packages", {}),
            "packages.conda": repodata.get("packages.conda", {}),
        }
        self.url = url

        shards = defaultdict(lambda: {"packages": {}, "packages.conda": {}})

        for group_name, group in all_packages.items():
            for package, record in group.items():
                name = record["name"]
                shards[name][group_name][package] = record

        # defaultdict behavior no longer wanted
        self.shards: dict[str, ShardDict] = dict(shards)  # type: ignore

        # used to write out repodata subset
        self.visited: dict[str, ShardDict | None] = {}

    def __repr__(self):
        left, right = super().__repr__().split(maxsplit=1)
        return f"{left} {self.url} {right}"

    @property
    def package_names(self) -> KeysView[str]:
        return self.shards.keys()

    def __contains__(self, package: str) -> bool:
        return package in self.package_names

    def fetch_shard(self, package: str) -> ShardDict:
        """
        "Fetch" an individual shard.

        Update self.visited with all not-None packages.

        Raise KeyError if package is not in the index.
        """
        shard = self.shards[package]
        self.visited[package] = shard
        return shard

    def fetch_shards(self, packages: Iterable[str]) -> dict[str, ShardDict]:
        """
        Fetch multiple shards in one go.

        Update self.visited with all not-None packages.
        """
        return {package: self.fetch_shard(package) for package in packages}

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


class Shards(ShardLike):
    """
    Handle repodata_shards.msgpack.zst and individual per-package shards.
    """

    def __init__(
        self, shards_index: ShardsIndexDict, url: str, shards_cache: shards_cache.ShardCache
    ):
        """
        Args:
            shards_index: raw parsed msgpack dict
            url: URL of repodata_shards.msgpack.zst
        """
        self.shards_index = shards_index
        self.url = url
        self.shards_cache = shards_cache

        # can we share a session for multiple subdir's of the same channel, or
        # any time self.shards_base_url is similar to another Shards() instance?
        self.session = get_session(self.shards_base_url)

        self.repodata_no_packages = {
            k: v for k, v in self.shards_index.items() if k not in ("shards",)
        }

        # used to write out repodata subset
        # not used in traversal algorithm
        self.visited: dict[str, ShardDict | None] = {}

    @property
    def package_names(self):
        return self.packages_index.keys()

    @property
    def packages_index(self):
        return self.shards_index["shards"]

    @property
    def shards_base_url(self) -> str:
        """
        Return self.url joined with shards_base_url.
        Note shards_base_url can be a relative or an absolute url.
        """
        return urljoin(self.url, self.shards_index["info"]["shards_base_url"])

    def shard_url(self, package: str) -> str:
        """
        Return shard URL for a given package.

        Raise KeyError if package is not in the index.
        """
        shard_name = f"{self.packages_index[package].hex()}.msgpack.zst"
        # "Individual shards are stored under the URL <shards_base_url><sha256>.msgpack.zst"
        return urljoin(self.shards_base_url, shard_name)

    def fetch_shard(self, package: str) -> ShardDict:
        """
        Fetch an individual shard.

        Raise KeyError if package is not in the index.
        """

        shards = self.fetch_shards((package,))
        return shards[package]

    def fetch_shards(self, packages: Iterable[str]) -> dict[str, ShardDict]:
        """
        Return mapping of *package names* to Shard for given packages.

        If a shard is already in self.visited, it is not fetched again.
        """
        result = {}

        def fetch(s, url, package):
            # due to cache, the same url won't be fetched twice
            b1 = time.time_ns()
            response = s.get(url)
            response.raise_for_status()
            data = response.content
            e1 = time.time_ns()
            log.debug(f"Fetch took {(e1 - b1) / 1e9}s %s %s", package, url)
            return shards_cache.AnnotatedRawShard(url=url, package=package, compressed_shard=data)

        packages = sorted(list(packages))
        urls_packages = {}
        for package in packages:
            if package in self.visited:
                result[package] = self.visited[package]
            else:
                urls_packages[self.shard_url(package)] = package

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
                executor.submit(fetch, self.session, url, package): (url, package)
                for url, package in urls_packages.items()
                if package not in result
            }
            # May be inconvenient to cancel a large number of futures. Also, ctrl-C doesn't work reliably out of the box.
            for future in concurrent.futures.as_completed(futures):
                log.debug(". %s", futures[future])
                # XXX future.result can raise HTTPError etc.
                fetch_result = future.result()
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
                    if set((fetch_result.package,)) != set(package_names):
                        log.debug(
                            "expected %s, actual %s", fetch_result.package, set(package_names)
                        )
                    self.shards_cache.insert(fetch_result)
                except (AttributeError, KeyError):
                    log.exception("Error fetching shard for %s", fetch_result.package)

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

    saved_fields = {conda.gateways.repodata.URL_KEY: url}
    for header, key in (
        ("Etag", conda.gateways.repodata.ETAG_KEY),
        (
            "Last-Modified",
            conda.gateways.repodata.LAST_MODIFIED_KEY,
        ),
        ("Cache-Control", conda.gateways.repodata.CACHE_CONTROL_KEY),
    ):
        _add_http_value_to_dict(response, header, saved_fields, key)

    state.update(saved_fields)

    # should we return the response and let caller save cache data to state?
    return response_bytes


def fetch_shards_index(
    sd: SubdirData, cache: shards_cache.ShardCache | None = None
) -> Shards | None:
    """
    Check a SubdirData's URL for shards.
    Return shards index bytes from cache or network.
    Return None if not found; caller should fetch normal repodata.
    """

    fetch = sd.repo_fetch
    repo_cache = fetch.repo_cache

    # cache.load_state() will clear the file on JSONDecodeError but cache.load()
    # will raise the exception.
    # repo_cache.load_state(
    #     binary=True
    # )  # won't succeed when .msgpack.zst is missing as it wants to compare the timestamp (returns empty state)

    # Load state ourselves to avoid clearing when binary cached data is missing.
    # If we fall back to monolithic repodata.json, the standard fetch code will
    # load the state again in text mode.
    try:
        # by using read_bytes, a possible UnicodeDecodeError should be converted
        # to a JSONDecodeError. But a valid json that is not a dict will fail on
        # .update().
        repo_cache.state.update(json.loads(repo_cache.cache_path_state.read_bytes()))
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    cache_state = repo_cache.state

    if cache is None:
        cache = shards_cache.ShardCache(Path(conda.gateways.repodata.create_cache_dir()))

    if cache_state.should_check_format("shards"):
        if not repo_cache.cache_path_shards.exists():
            # avoid 304 not modified if we don't have the file
            cache_state.etag = ""
            cache_state.mod = ""
        try:
            # look for shards index
            shards_index_url = f"{sd.url_w_subdir}/repodata_shards.msgpack.zst"
            found = repodata_shards(shards_index_url, repo_cache)
            cache_state.set_has_format("shards", True)
            # this will also set state["refresh_ns"] = time.time_ns(); we could
            # call cache.refresh() if we got a 304 instead:
            repo_cache.save(found)

            # basic parse (move into caller?)
            shards_index: ShardsIndexDict = msgpack.loads(zstandard.decompress(found))  # type: ignore
            shards = Shards(shards_index, shards_index_url, cache)
            return shards

        except (HTTPError, conda.gateways.repodata.RepodataIsEmpty):
            # fetch repodata.json / repodata.json.zst instead
            cache_state.set_has_format("shards", False)
            repo_cache.refresh()

    return None


def batch_retrieve_from_cache(sharded: list[Shards], packages: list[str]):
    """
    Given a list of Shards objects and a list of package names, fetch all URLs
    from a shared local cache, and update Shards with those per-package shards.
    Return the remaining URLs that must be fetched from the network.
    """
    sharded = [shardlike for shardlike in sharded if isinstance(shardlike, Shards)]

    wanted = []
    for shard in sharded:
        for package_name in packages:
            if package_name in shard:
                wanted.append((shard, package_name, shard.shard_url(package_name)))

    log.debug("%d shards to fetch", len(wanted))

    if not sharded:
        log.debug("No sharded channels found.")
        return wanted

    shared_shard_cache = sharded[0].shards_cache
    from_cache = shared_shard_cache.retrieve_multiple([shard_url for *_, shard_url in wanted])

    # add fetched Shard objects to Shards objects visited dict
    for shard, package, shard_url in wanted:
        if from_cache_shard := from_cache.get(shard_url):
            shard.visited[package] = from_cache_shard

    return wanted
