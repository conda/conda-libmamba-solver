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
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urljoin

import conda.gateways.repodata
import msgpack
import zstandard
from conda.base.context import context
from conda.core.subdir_data import SubdirData
from conda.gateways.connection.session import CondaSession, get_session
from conda.gateways.repodata import (
    _add_http_value_to_dict,
    conda_http_errors,
)
from conda.models.channel import Channel
from libmambapy.bindings import specs
from requests import HTTPError

from . import shards_cache

log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Iterable, KeysView

    from conda.gateways.repodata import RepodataCache
    from requests import Response

    from conda_libmamba_solver.shards_typing import RepodataDict, ShardsIndexDict

    from .shards_typing import PackageRecordDict, ShardDict

SHARDS_CONNECTIONS_DEFAULT = 10
ZSTD_MAX_SHARD_SIZE = 2**20 * 16  # maximum size necessary when compresed data has no size header


def _shards_connections() -> int:
    """
    If context.repodata_threads is not set, find the size of the connection pool
    in a typical https:// session. This should significantly reduce dropped
    connections. This will usually be requests' default 10.

    Is this shared between all sessions? Or do we get a different pool for a
    different get_session(url)?

    Other adapters (file://, s3://) used in conda would have different
    concurrency behavior;  we are not prepared to have separate threadpools per
    connection type.
    """
    if context.repodata_threads is not None:
        return context.repodata_threads
    session = CondaSession()
    adapter = session.get_adapter("https://")
    if poolmanager := getattr(adapter, "poolmanager"):
        try:
            return int(poolmanager.connection_pool_kw["maxsize"])
        except (KeyError, ValueError, AttributeError, TypeError):
            pass
    return SHARDS_CONNECTIONS_DEFAULT


def ensure_hex_hash(record: PackageRecordDict):
    """
    Convert bytes checksums to hex; leave unchanged if already str.
    """
    for hash_type in "sha256", "md5":
        if hash_value := record.get(hash_type):
            if isinstance(hash_value, bytes):
                record[hash_type] = hash_value.hex()
    return record


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
        url: must be unique for all ShardLike used together.
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

        # alternate location for packages, if not self.url
        try:
            base_url = self.repodata_no_packages["info"]["base_url"]
            if not isinstance(base_url, str):
                log.warning(f'repodata["info"]["base_url"] was not a str, got {type(base_url)}')
                raise TypeError()
            self._base_url = base_url
        except KeyError:
            self._base_url = ""

    def __repr__(self):
        left, right = super().__repr__().split(maxsplit=1)
        return f"{left} {self.url} {right}"

    @property
    def package_names(self) -> KeysView[str]:
        return self.shards.keys()

    @property
    def base_url(self) -> str:
        """
        Return self.url joined with base_url from repodata, or self.url if no
        base_url was present. Packages are found here.

        Note base_url can be a relative or an absolute url.
        """
        return urljoin(self.url, self._base_url)

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
        for _, shard in self.visited.items():
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
            "info": shards_index["info"],
            "packages": {},
            "packages.conda": {},
            "repodata_version": 2,
        }

        # used to write out repodata subset
        # not used in traversal algorithm
        self.visited: dict[str, ShardDict | None] = {}

        # https://github.com/conda/conda-index/pull/209 ensures that sharded
        # repodata will always include base_url, even if it an empty string;
        # this is also necessary for compatibility.
        self._base_url = shards_index["info"]["base_url"]

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
        base_url = self.shards_index["info"].get("shards_base_url", "")
        # IMO shards_base_url should end with a /, to match HTML "base url =
        # https://example.com/index.html; look for resources under
        # example.com/<file>". Append / for compatibility.
        if base_url and not base_url.endswith("/"):
            base_url += "/"
        return urljoin(self.url, base_url)

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
            response = s.get(url)
            response.raise_for_status()
            data = response.content
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=_shards_connections()) as executor:
            futures = {
                executor.submit(fetch, self.session, url, package): (url, package)
                for url, package in urls_packages.items()
                if package not in result
            }
            for future in concurrent.futures.as_completed(futures):
                log.debug(". %s", futures[future])
                # XXX future.result can raise HTTPError etc.
                fetch_result = future.result()
                result[fetch_result.package] = msgpack.loads(
                    zstandard.decompress(
                        fetch_result.compressed_shard, max_output_size=ZSTD_MAX_SHARD_SIZE
                    )
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
            shards_index: ShardsIndexDict = msgpack.loads(
                zstandard.decompress(found, max_output_size=ZSTD_MAX_SHARD_SIZE)
            )  # type: ignore
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
    # XXX update batch_retrieve_from_cache to work with (Shards, package name)
    # tuples instead of broadcasting across shards itself.
    for shard in sharded:
        for package_name in packages:
            if package_name in shard:  # and not package_name in shard.visited
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


def batch_retrieve_from_network(wanted: list[tuple[Shards, str, str]]):
    """
    Given a list of (Shards, package name, shard URL) tuples, group by Shards and call fetch_shards with a list of all urls for that Shard.
    """
    if not wanted:
        return

    shard_packages: dict[Shards, list[str]] = defaultdict(list)
    for shard, package, _ in wanted:
        shard_packages[shard].append(package)

    # XXX it might be better to pull networking and Session() out of Shards(),
    # so that we can e.g. use the same session for a Channel(); typically a
    # noarch+arch pair of subdirs.
    # Could we share a ThreadPoolExecutor and see better session utilization?
    for shard, packages in shard_packages.items():
        # this function also checks the database again, though we should have
        # just called batch_retrieve_from_cache:
        shard.fetch_shards(packages)


def fetch_channels(channels):
    channel_data: dict[str, ShardLike] = {}

    # share single disk cache for all Shards() instances
    cache = shards_cache.ShardCache(Path(conda.gateways.repodata.create_cache_dir()))

    # The parallel version may reorder channels, does this matter?

    with concurrent.futures.ThreadPoolExecutor(max_workers=_shards_connections()) as executor:
        futures = {
            executor.submit(
                fetch_shards_index, SubdirData(Channel(channel_url)), cache
            ): channel_url
            for channel in channels
            for channel_url in Channel(channel).urls(True, context.subdirs)
        }
        futures_non_sharded = {}
        for future in concurrent.futures.as_completed(futures):
            channel_url = futures[future]
            found = future.result()
            if found:
                channel_data[channel_url] = found
            else:
                futures_non_sharded[
                    executor.submit(
                        SubdirData(Channel(channel_url)).repo_fetch.fetch_latest_parsed
                    )
                ] = channel_url

        for future in concurrent.futures.as_completed(futures_non_sharded):
            channel_url = futures_non_sharded[future]
            repodata_json, _ = future.result()
            # the filename is not strictly repodata.json since we could have
            # fetched the same data from repodata.json.zst; but makes the
            # urljoin consistent with shards which end with
            # /repodata_shards.msgpack.zst
            url = f"{channel_url}/repodata.json"
            found = ShardLike(repodata_json, url)
            channel_data[channel_url] = found

    return channel_data
