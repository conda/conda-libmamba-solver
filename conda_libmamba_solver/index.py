# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
This module provides a convenient interface between `libmamba.Solver`
and conda's `PrefixData`. In other words, it allows to expose channels
loaded in `conda` to the `libmamba` machinery without using the
`libmamba` networking stack.

Internally, the `libmamba`'s index is made of:

- A 'Pool' object, exposed to libsolv.
- The pool is made of `Repo` objects.
- Each repo corresponds to a repodata.json file.
- Each repodata comes from a channel+subdir combination.

Some notes about channels
-------------------------

In a way, conda channels are an abstraction over a collection of
channel subdirs. For example, when the user wants 'conda-forge', it
actually means 'repodata.json' files from the configured platform subdir
and 'noarch'. Some channels are actually 'MultiChannel', which provide
a collection of channels. The most common example is 'defaults', which
includes 'main', 'r' and 'msys2'.

So, for conda-forge on Linux amd64 we get:

- https://conda.anaconda.org/conda-forge/linux-64
- https://conda.anaconda.org/conda-forge/noarch

For defaults on macOS with Apple Silicon (M1 and friends):

- https://repo.anaconda.org/main/osx-arm64
- https://repo.anaconda.org/main/noarch
- https://repo.anaconda.org/r/osx-arm64
- https://repo.anaconda.org/r/noarch
- https://repo.anaconda.org/msys2/osx-arm64
- https://repo.anaconda.org/msys2/noarch

However, users will just say 'defaults' or 'conda-forge', for convenience.
This means that we need to deal with several formats of channel information,
which ultimately lead to a collection of subdir-specific URLs:

- Channel names from the CLI or configuration files / env-vars
- Channel URLs if names are not available (channel not served in anaconda.org)
- conda.models.channel.Channel objects

Their origins can be:

- Specified by the user on the command-line (-c arguments)
- Specified by the configuration files (.condarc) or environment vars (context object)
- Added from channel-specific MatchSpec (e.g. `conda-forge::python`)
- Added from installed packages in target environment (e.g. a package that was installed
  from a non-default channel remembers where it comes from)

Also note that a channel URL might have authentication in the form:

- https://user:password@server.com/channel
- https://server.com/t/your_token_goes_here/channel

Finally, a channel can be mounted in a local directory and referred to via
a regular path, or a file:// URL, with or without normalization on Windows.

The approach
------------
We pass the subdir-specific, authenticated URLs to conda's 'SubdirData.repo_patch',
which download the JSON files but do not process them to PackageRecords.
Once the cache has been populated, we can instantiate 'libmamba.Repo' objects directly.
We maintain a map of subdir-specific URLs to `conda.model.channel.Channel`
and `libmamba.Repo` objects.
"""
import logging
import os
from dataclasses import dataclass
from functools import lru_cache, partial
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, Tuple, Union

import libmambapy as api
from conda.base.constants import REPODATA_FN
from conda.base.context import context
from conda.common.io import DummyExecutor, ThreadLimitedThreadPoolExecutor
from conda.common.serialize import json_dump, json_load
from conda.common.url import percent_decode, remove_auth, split_anaconda_token
from conda.core.subdir_data import SubdirData
from conda.models.channel import Channel
from conda.models.match_spec import MatchSpec
from conda.models.records import PackageRecord

from .mamba_utils import set_channel_priorities
from .state import IndexHelper
from .utils import escape_channel_url

log = logging.getLogger(f"conda.{__name__}")


@dataclass(frozen=True)
class _ChannelRepoInfo:
    "A dataclass mapping conda Channels, libmamba Repos and URLs"
    channel: Channel
    repo: api.Repo
    full_url: str
    noauth_url: str


class LibMambaIndexHelper(IndexHelper):
    def __init__(
        self,
        installed_records: Iterable[PackageRecord] = (),
        channels: Iterable[Union[Channel, str]] = None,
        subdirs: Iterable[str] = None,
        repodata_fn: str = REPODATA_FN,
    ):
        self._channels = context.channels if channels is None else channels
        self._subdirs = context.subdirs if subdirs is None else subdirs
        self._repodata_fn = repodata_fn

        self._repos = []
        self._pool = api.Pool()

        installed_repo = self._load_installed(installed_records)
        self._repos.append(installed_repo)

        self._index = self._load_channels()
        self._repos += [info.repo for info in self._index.values()]

        self._query = api.Query(self._pool)
        self._format = api.QueryFormat.JSON

    def get_info(self, key: str) -> _ChannelRepoInfo:
        orig_key = key
        if not key.startswith("file://"):
            # The conda functions (specifically remove_auth) assume the input
            # is a url; a file uri on windows with a drive letter messes them up.
            # For the rest, we remove all forms of authentication
            key = split_anaconda_token(remove_auth(key))[0]
        try:
            return self._index[key]
        except KeyError as exc:
            # some libmamba versions return encoded URLs
            try:
                return self._index[percent_decode(key)]
            except KeyError:
                pass  # raise original error below
            raise KeyError(
                f"Channel info for {orig_key} ({key}) not found. "
                f"Available keys: {list(self._index)}"
            ) from exc

    def reload_local_channels(self):
        """
        Reload a channel that was previously loaded from a local directory.
        """
        for url, info in self._index.items():
            if url.startswith("file://"):
                url, json_path = self._fetch_channel(url)
                new = self._json_path_to_repo_info(url, json_path)
                self._repos[self._repos.index(info.repo)] = new.repo
                self._index[url] = new
        set_channel_priorities(self._index)

    def _repo_from_records(
        self, pool: api.Pool, repo_name: str, records: Iterable[PackageRecord] = ()
    ) -> api.Repo:
        """
        Build a libmamba 'Repo' object from conda 'PackageRecord' objects.

        This is done by rebuilding a repodata.json-like dictionary, which is
        then exported to a temporary file that will be loaded with 'libmambapy.Repo'.
        """
        exported = {"packages": {}}
        additional_infos = {}
        for record in records:
            record_data = dict(record.dump())
            # These fields are expected by libmamba, but they don't always appear
            # in the record.dump() dict (e.g. exporting from S3 channels)
            # ref: https://github.com/mamba-org/mamba/blob/ad46f318b/libmamba/src/core/package_info.cpp#L276-L318  # noqa
            for field in (
                "sha256",
                "track_features",
                "license",
                "size",
                "url",
                "noarch",
                "platform",
                "timestamp",
            ):
                if field in record_data:
                    continue  # do not overwrite
                value = getattr(record, field, None)
                if value is not None:
                    if field == "timestamp" and value:
                        value = int(value * 1000)  # from s to ms
                    record_data[field] = value
            exported["packages"][record.fn] = record_data

            # extra info for libmamba
            info = api.ExtraPkgInfo()
            if record.noarch:
                info.noarch = record.noarch.value
            if record.channel and record.channel.subdir_url:
                info.repo_url = record.channel.subdir_url
            additional_infos[record.name] = info

        with NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write(json_dump(exported))

        try:
            repo = api.Repo(pool, repo_name, f.name, "")
            repo.add_extra_pkg_info(additional_infos)
            return repo
        finally:
            os.unlink(f.name)

    def _fetch_channel(self, url: str) -> Tuple[str, os.PathLike]:
        channel = Channel.from_url(url)
        if not channel.subdir:
            raise ValueError(f"Channel URLs must specify a subdir! Provided: {url}")

        if "PYTEST_CURRENT_TEST" in os.environ:
            # Workaround some testing issues - TODO: REMOVE
            # Fix conda.testing.helpers._patch_for_local_exports by removing last line
            maybe_cached = SubdirData._cache_.get((url, self._repodata_fn))
            if maybe_cached and maybe_cached._mtime == float("inf"):
                del SubdirData._cache_[(url, self._repodata_fn)]
            # /Workaround

        log.debug("Fetching %s with SubdirData.repo_fetch", channel)
        subdir_data = SubdirData(channel, repodata_fn=self._repodata_fn)
        json_path, _ = subdir_data.repo_fetch.fetch_latest_path()

        return url, json_path

    def _json_path_to_repo_info(self, url: str, json_path: str) -> _ChannelRepoInfo:
        channel = Channel.from_url(url)
        noauth_url = channel.urls(with_credentials=False, subdirs=(channel.subdir,))[0]
        repo = api.Repo(self._pool, noauth_url, str(json_path), escape_channel_url(noauth_url))
        return _ChannelRepoInfo(
            repo=repo,
            channel=channel,
            full_url=url,
            noauth_url=noauth_url,
        )

    def _load_channels(self) -> Dict[str, _ChannelRepoInfo]:
        # 1. Obtain and deduplicate URLs from channels
        urls = []
        seen_noauth = set()
        for _c in self._channels:
            c = Channel(_c)
            noauth_urls = c.urls(with_credentials=False, subdirs=self._subdirs)
            if seen_noauth.issuperset(noauth_urls):
                continue
            if c.auth or c.token:  # authed channel always takes precedence
                urls += Channel(c).urls(with_credentials=True, subdirs=self._subdirs)
                seen_noauth.update(noauth_urls)
                continue
            # at this point, we are handling an unauthed channel; in some edge cases,
            # an auth'd variant of the same channel might already be present in `urls`.
            # we only add them if we haven't seen them yet
            for url in noauth_urls:
                if url not in seen_noauth:
                    urls.append(url)
                    seen_noauth.add(url)

        urls = tuple(dict.fromkeys(urls))  # de-duplicate

        # 2. Fetch URLs (if needed)
        Executor = (
            DummyExecutor
            if context.debug or context.repodata_threads == 1
            else partial(ThreadLimitedThreadPoolExecutor, max_workers=context.repodata_threads)
        )
        with Executor() as executor:
            jsons = {url: str(path) for (url, path) in executor.map(self._fetch_channel, urls)}

        # 3. Create repos in same order as `urls`
        index = {}
        for url in urls:
            info = self._json_path_to_repo_info(url, jsons[url])
            index[info.noauth_url] = info

        # 4. Configure priorities
        set_channel_priorities(index)

        return index

    def _load_installed(self, records: Iterable[PackageRecord]) -> api.Repo:
        repo = self._repo_from_records(self._pool, "installed", records)
        repo.set_installed()
        return repo

    def whoneeds(self, query: str, records=True) -> Union[Iterable[PackageRecord], dict]:
        result_str = self._query.whoneeds(query, self._format)
        return self._process_query_result(result_str, records=records)

    def depends(self, query: str, records=True) -> Union[Iterable[PackageRecord], dict]:
        result_str = self._query.depends(query, self._format)
        return self._process_query_result(result_str, records=records)

    def search(self, query: str, records=True) -> Union[Iterable[PackageRecord], dict]:
        result_str = self._query.find(query, self._format)
        return self._process_query_result(result_str, records=records)

    def explicit_pool(self, specs: Iterable[MatchSpec]) -> Iterable[str]:
        """
        Returns all the package names that (might) depend on the passed specs
        """
        explicit_pool = set()
        for spec in specs:
            pkg_records = self.depends(spec.dist_str())
            for record in pkg_records:
                explicit_pool.add(record.name)
        return tuple(explicit_pool)

    def _process_query_result(
        self,
        result_str,
        records=True,
    ) -> Union[Iterable[PackageRecord], dict]:
        result = json_load(result_str)
        if result.get("result", {}).get("status") != "OK":
            query_type = result.get("query", {}).get("type", "<Unknown>")
            query = result.get("query", {}).get("query", "<Unknown>")
            error_msg = result.get("result", {}).get("msg", f"Faulty response: {result_str}")
            raise ValueError(f"{query_type} query '{query}' failed: {error_msg}")
        if records:
            pkg_records = []
            for pkg in result["result"]["pkgs"]:
                record = PackageRecord(**pkg)
                pkg_records.append(record)
            return pkg_records
        return result


# for conda-build
_CachedLibMambaIndexHelper = lru_cache(maxsize=None)(LibMambaIndexHelper)
