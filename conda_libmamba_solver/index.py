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

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from conda.base.constants import KNOWN_SUBDIRS, REPODATA_FN, ChannelPriority
from conda.base.context import context
from conda.common.compat import on_win
from conda.common.io import DummyExecutor, ThreadLimitedThreadPoolExecutor
from conda.common.url import path_to_url, remove_auth, split_anaconda_token
from conda.core.package_cache_data import PackageCacheData
from conda.core.subdir_data import SubdirData
from conda.models.channel import Channel
from conda.models.match_spec import MatchSpec
from conda.models.records import PackageRecord
from libmambapy import MambaNativeException, Query
from libmambapy.solver.libsolv import (
    Database,
    PackageTypes,
    PipAsPythonDependency,
    Priorities,
    RepodataOrigin,
)
from libmambapy.specs import (
    Channel as LibmambaChannel,
)
from libmambapy.specs import (
    ChannelResolveParams,
    CondaURL,
    NoArchType,
    PackageInfo,
)

from .mamba_utils import logger_callback

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any, Literal

    from conda.common.path import PathsType
    from conda.gateways.repodata import RepodataState
    from libmambapy import QueryResult
    from libmambapy.solver.libsolv import RepoInfo


log = logging.getLogger(f"conda.{__name__}")


@dataclass
class _ChannelRepoInfo:
    "A dataclass mapping conda Channels, libmamba Repos and URLs"

    channel: Channel | None
    repo: RepoInfo
    url_w_cred: str
    url_no_cred: str

    @property
    def canonical_name(self) -> str:
        if self.channel:
            return self.channel.canonical_name
        url_parts = self.url_no_cred.split("/")
        if url_parts[-1] in KNOWN_SUBDIRS:
            return url_parts[-2]
        return url_parts[-1]


class LibMambaIndexHelper:
    """
    Interface between conda and libmamba for the purpose of building the "index".
    The index is the collection of package records that can be part of a solution.
    It is built by collecting all the repodata.json files from channels and their
    subdirs. For existing environments, the installed packages are also added to
    the index (this helps with simplifying solutions and outputs). The local cache
    can also be added as a "channel", which is useful in offline mode or with no
    channels configured.
    """

    def __init__(
        self,
        channels: Iterable[Channel],
        subdirs: Iterable[str] = (),
        repodata_fn: str = REPODATA_FN,
        installed_records: Iterable[PackageRecord] = (),
        pkgs_dirs: PathsType = (),
    ):
        platform_less_channels = []
        for channel in channels:
            if channel.platform:
                # When .platform is defined, .urls() will ignore subdirs kw. Remove!
                log.info(
                    "Platform-aware channels are not supported. "
                    "Ignoring platform %s from channel %s. "
                    "Use subdirs keyword if necessary.",
                    channel.platform,
                    channel,
                )
                channel = Channel(**{k: v for k, v in channel.dump().items() if k != "platform"})
            platform_less_channels.append(channel)
        self.channels = platform_less_channels
        self.subdirs = subdirs or context.subdirs
        self.repodata_fn = repodata_fn
        self.db = self._init_db()
        self.repos: list[_ChannelRepoInfo] = self._load_channels()
        if pkgs_dirs:
            self.repos.extend(self._load_pkgs_cache(pkgs_dirs))
        if installed_records:
            self.repos.append(self._load_installed(installed_records))
        self._set_repo_priorities()

    @classmethod
    def from_platform_aware_channel(cls, channel: Channel) -> LibMambaIndexHelper:
        if not channel.platform:
            raise ValueError(f"Channel {channel} must define 'platform' attribute.")
        subdir = channel.platform
        channel = Channel(**{k: v for k, v in channel.dump().items() if k != "platform"})
        return cls(channels=(channel,), subdirs=(subdir,))

    def n_packages(
        self,
        repos: Iterable[RepoInfo] | None = None,
        filter_: callable | None = None,
    ) -> int:
        repos = repos or [repo_info.repo for repo_info in self.repos]
        count = 0
        for repo in repos:
            if filter_:
                for pkg in self.db.packages_in_repo(repo):
                    if filter_(pkg):
                        count += 1
            else:
                count += len(self.db.packages_in_repo(repo))
        return count

    def reload_channel(self, channel: Channel) -> None:
        urls = {}
        for url in channel.urls(with_credentials=False, subdirs=self.subdirs):
            for repo_info in self.repos:
                if repo_info.url_no_cred == url:
                    log.debug("Reloading repo %s", repo_info.url_no_cred)
                    urls[repo_info.url_w_cred] = channel
                    self.db.remove_repo(repo_info.repo)
        for new_repo_info in self._load_channels(urls, try_solv=False):
            for repo_info in self.repos:
                if repo_info.url_no_cred == new_repo_info.url_no_cred:
                    repo_info.repo = new_repo_info.repo
        self._set_repo_priorities()

    def _init_db(self) -> Database:
        custom_channels = {
            # Add custom channels as a workaround for this weird conda behavior
            # See https://github.com/conda/conda/issues/13501
            **{c.name: c for c in self.channels if c.location != context.channel_alias.location},
            **context.custom_channels,
        }
        custom_channels = {
            name: LibmambaChannel(
                url=CondaURL.parse(channel.base_url.replace(" ", "%20")),
                display_name=name,
                platforms=set(self.subdirs),
            )
            for (name, channel) in custom_channels.items()
            if channel.base_url
        }
        custom_multichannels = {
            channel_name: [
                custom_channels.get(
                    channel.name,
                    LibmambaChannel(
                        url=CondaURL.parse(channel.base_url.replace(" ", "%20")),
                        display_name=channel.name,
                        platforms=set(self.subdirs),
                    ),
                )
                for channel in channels
                if channel.base_url
            ]
            for channel_name, channels in context.custom_multichannels.items()
        }
        params = ChannelResolveParams(
            platforms=set(self.subdirs),
            channel_alias=CondaURL.parse(str(context.channel_alias)),
            custom_channels=ChannelResolveParams.ChannelMap(custom_channels),
            custom_multichannels=ChannelResolveParams.MultiChannelMap(custom_multichannels),
            home_dir=str(Path.home()),
            current_working_dir=os.getcwd(),
        )
        db = Database(params)
        db.set_logger(logger_callback)
        return db

    def _load_channels(
        self,
        urls_to_channel: dict[str, Channel] | None = None,
        try_solv: bool = True,
    ) -> list[_ChannelRepoInfo]:
        if urls_to_channel is None:
            urls_to_channel = self._channel_urls()

        # conda.common.url.path_to_url does not %-encode spaces
        encoded_urls_to_channel = {}
        for url, channel in urls_to_channel.items():
            if url.startswith("file://"):
                url = url.replace(" ", "%20")
            encoded_urls_to_channel[url] = channel
        urls_to_channel = encoded_urls_to_channel

        urls_to_json_path_and_state = self._fetch_repodata_jsons(tuple(urls_to_channel.keys()))
        channel_repo_infos = []
        for url_w_cred, (json_path, state) in urls_to_json_path_and_state.items():
            url_no_token, _ = split_anaconda_token(url_w_cred)
            url_no_cred = remove_auth(url_no_token)
            repo = self._load_repo_info_from_json_path(
                json_path,
                url_no_cred,
                state,
                try_solv=try_solv,
            )
            channel_repo_infos.append(
                _ChannelRepoInfo(
                    channel=urls_to_channel[url_w_cred],
                    repo=repo,
                    url_w_cred=url_w_cred,
                    url_no_cred=url_no_cred,
                )
            )
        return channel_repo_infos

    def _channel_urls(self) -> dict[str, Channel]:
        "Maps authenticated URLs to channel objects"
        urls = {}
        seen_noauth = set()
        channels_with_subdirs = []
        for channel in self.channels:
            for url in channel.urls(with_credentials=True, subdirs=self.subdirs):
                channels_with_subdirs.append(Channel(url))
        for channel in channels_with_subdirs:
            noauth_urls = [
                url
                for url in channel.urls(with_credentials=False)
                if url.endswith(tuple(self.subdirs))
            ]
            if seen_noauth.issuperset(noauth_urls):
                continue
            auth_urls = [
                url.replace(" ", "%20")
                for url in channel.urls(with_credentials=True)
                if url.endswith(tuple(self.subdirs))
            ]
            if noauth_urls != auth_urls:  # authed channel always takes precedence
                urls.update({url: channel for url in auth_urls})
                seen_noauth.update(noauth_urls)
                continue
            # at this point, we are handling an unauthed channel; in some edge cases,
            # an auth'd variant of the same channel might already be present in `urls`.
            # we only add them if we haven't seen them yet
            for url in noauth_urls:
                if url not in seen_noauth:
                    urls[url] = channel
                    seen_noauth.add(url)
        return urls

    def _fetch_repodata_jsons(self, urls: dict[str, str]) -> dict[str, tuple[str, RepodataState]]:
        Executor = (
            DummyExecutor
            if context.debug or context.repodata_threads == 1
            else partial(ThreadLimitedThreadPoolExecutor, max_workers=context.repodata_threads)
        )
        with Executor() as executor:
            return {
                url: (str(path), state)
                for (url, path, state) in executor.map(self._fetch_one_repodata_json, urls)
            }

    def _fetch_one_repodata_json(self, url: str) -> tuple[str, os.PathLike, RepodataState]:
        channel = Channel.from_url(url)
        if not channel.subdir:
            raise ValueError("Channel URLs must specify a subdir!")

        if "PYTEST_CURRENT_TEST" in os.environ:
            # Workaround some testing issues - TODO: REMOVE
            # Fix conda.testing.helpers._patch_for_local_exports by removing last line
            for key, cached in list(SubdirData._cache_.items()):
                if not isinstance(key, tuple):
                    continue  # should not happen, but avoid IndexError just in case
                if key[:2] == (url, self.repodata_fn) and cached._mtime == float("inf"):
                    del SubdirData._cache_[key]
            # /Workaround

        subdir_data = SubdirData(channel, repodata_fn=self.repodata_fn)
        if context.offline or context.use_index_cache:
            # This might not exist (yet, anymore), but that's ok because we'll check
            # for existence later and safely ignore if needed
            json_path = subdir_data.cache_path_json
            state = subdir_data.repo_cache.load_state()
        else:
            # TODO: This method loads reads the whole JSON file (does not parse)
            json_path, state = subdir_data.repo_fetch.fetch_latest_path()
        return url, json_path, state

    def _load_repo_info_from_json_path(
        self, json_path: str, channel_url: str, state: RepodataState, try_solv: bool = True
    ) -> RepoInfo | None:
        if try_solv and on_win:
            # .solv loading is so slow on Windows is not even worth it. Use JSON instead.
            # https://github.com/mamba-org/mamba/pull/2753#issuecomment-1739122830
            log.debug("Overriding truthy 'try_solv' as False on Windows for performance reasons.")
            try_solv = False
        json_path = Path(json_path)
        solv_path = json_path.with_suffix(".solv")
        if state:
            repodata_origin = RepodataOrigin(url=channel_url, etag=state.etag, mod=state.mod)
        else:
            repodata_origin = None
        channel = Channel(channel_url)
        channel_id = channel.canonical_name
        if channel_id in context.custom_multichannels:
            # In multichannels, the canonical name of a "subchannel" is the multichannel name
            # which makes it ambiguous for `channel::specs`. In those cases, take the channel
            # regular name; e.g. for repo.anaconda.com/pkgs/main, do not take defaults, but
            # pkgs/main instead.
            channel_id = channel.name
        if try_solv and repodata_origin:
            try:
                log.debug(
                    "Loading %s (%s) from SOLV repodata at %s", channel_id, channel_url, solv_path
                )
                return self.db.add_repo_from_native_serialization(
                    path=str(solv_path),
                    expected=repodata_origin,
                    channel_id=channel_id,
                    add_pip_as_python_dependency=PipAsPythonDependency(
                        context.add_pip_as_python_dependency
                    ),
                )
            except Exception as exc:
                log.debug("Failed to load from SOLV. Trying JSON.", exc_info=exc)
        try:
            log.debug(
                "Loading %s (%s) from JSON repodata at %s", channel_id, channel_url, json_path
            )
            repo = self.db.add_repo_from_repodata_json(
                path=str(json_path),
                url=channel_url,
                channel_id=channel_id,
                add_pip_as_python_dependency=PipAsPythonDependency(
                    context.add_pip_as_python_dependency
                ),
                package_types=(
                    PackageTypes.TarBz2Only
                    if context.use_only_tar_bz2
                    else PackageTypes.CondaOrElseTarBz2
                ),
            )
        except MambaNativeException as exc:
            if "does not exist" in str(exc) and context.offline:
                # Ignore errors in offline mode. This is needed to pass
                # tests/test_create.py::test_offline_with_empty_index_cache.
                # In offline mode, with no repodata cache available, conda can still
                # create a channel from the pkgs/ content. For that to work, we must
                # not error out this early. If the package is still not found, the solver
                # will complain that the package cannot be found.
                log.warning("Could not load repodata for %s.", channel_id)
                log.debug("Ignored MambaNativeException in offline mode: %s", exc, exc_info=exc)
                return None
            raise exc
        if try_solv and repodata_origin:
            try:
                self.db.native_serialize_repo(
                    repo=repo, path=str(solv_path), metadata=repodata_origin
                )
            except MambaNativeException as exc:
                log.debug("Ignored SOLV writing error for %s", channel_id, exc_info=exc)
        return repo

    def _load_installed(self, records: Iterable[PackageRecord]) -> _ChannelRepoInfo:
        packages = [self._package_info_from_package_record(record) for record in records]
        repo = self.db.add_repo_from_packages(
            packages=packages,
            name="installed",
            add_pip_as_python_dependency=PipAsPythonDependency.No,
        )
        self.db.set_installed_repo(repo)
        return _ChannelRepoInfo(
            channel=None, repo=repo, url_w_cred="installed", url_no_cred="installed"
        )

    def _load_pkgs_cache(self, pkgs_dirs: PathsType) -> list[RepoInfo]:
        repos = []
        for path in pkgs_dirs:
            package_cache_data = PackageCacheData(path)
            package_cache_data.load()
            packages = [
                self._package_info_from_package_record(record)
                for record in package_cache_data.values()
            ]
            repo = self.db.add_repo_from_packages(packages=packages, name=path)
            # path_to_url does not %-encode spaces
            path_as_url = path_to_url(path).replace(" ", "%20")
            repos.append(
                _ChannelRepoInfo(
                    channel=None, repo=repo, url_w_cred=path_as_url, url_no_cred=path_as_url
                )
            )
        return repos

    def _package_info_from_package_record(self, record: PackageRecord) -> PackageInfo:
        if record.get("noarch", None) and record.noarch.value in ("python", "generic"):
            noarch = NoArchType(record.noarch.value.title())
        else:
            noarch = NoArchType("No")
        return PackageInfo(
            name=record.name,
            version=record.version,
            build_string=record.build or "",
            build_number=record.build_number or 0,
            channel=str(record.channel),
            package_url=record.get("url") or "",
            platform=record.subdir,
            filename=record.fn or f"{record.name}-{record.version}-{record.build or ''}",
            license=record.get("license") or "",
            md5=record.get("md5") or "",
            sha256=record.get("sha256") or "",
            signatures=record.get("signatures") or "",
            # conda can have list or tuple, but libmamba only accepts lists
            track_features=list(record.get("track_features") or []),
            depends=list(record.get("depends") or []),
            constrains=list(record.get("constrains") or []),
            defaulted_keys=list(record.get("defaulted_keys") or []),
            noarch=noarch,
            size=record.get("size") or 0,
            timestamp=int((record.get("timestamp") or 0) * 1000),
        )

    def _set_repo_priorities(self) -> None:
        has_priority = context.channel_priority in (
            ChannelPriority.STRICT,
            ChannelPriority.FLEXIBLE,
        )

        subprio_index = len(self.repos)
        if has_priority:
            # max channel priority value is the number of unique channels
            channel_prio = len({repo.canonical_name for repo in self.repos})
            current_channel_name = self.repos[0].canonical_name

        for repo_info in self.repos:
            if repo_info.repo is None:
                continue
            if has_priority:
                if repo_info.canonical_name != current_channel_name:
                    channel_prio -= 1
                    current_channel_name = repo_info.canonical_name
                priority = channel_prio
            else:
                priority = 0
            if has_priority:
                # NOTE: -- This was originally 0, but we need 1.
                # Otherwise, conda/conda @ test_create::test_force_remove fails :shrug:
                subpriority = 1
            else:
                subpriority = subprio_index
                subprio_index -= 1

            log.debug(
                "Channel: %s, prio: %s : %s",
                repo_info.url_no_cred,
                priority,
                subpriority,
            )
            self.db.set_repo_priority(repo_info.repo, Priorities(priority, subpriority))

    # region Repoquery
    #################

    def search(
        self,
        queries: Iterable[str | MatchSpec] | str | MatchSpec,
        return_type: Literal["records", "dict", "raw"] = "records",
    ) -> list[PackageRecord] | dict[str, Any] | QueryResult:
        if isinstance(queries, (str, MatchSpec)):
            queries = [queries]
        queries = list(map(str, queries))
        result = Query.find(self.db, queries)
        return self._process_query_result(result, return_type)

    def depends(
        self,
        query: str | MatchSpec,
        tree: bool = False,
        return_type: Literal["records", "dict", "raw"] = "records",
    ) -> list[PackageRecord] | dict[str, Any] | QueryResult:
        query = str(query)
        result = Query.depends(self.db, query, tree)
        return self._process_query_result(result, return_type)

    def whoneeds(
        self,
        query: str | MatchSpec,
        tree: bool = False,
        return_type: Literal["records", "dict", "raw"] = "records",
    ) -> list[PackageRecord] | dict[str, Any] | QueryResult:
        query = str(query)
        result = Query.whoneeds(self.db, query, tree)
        return self._process_query_result(result, return_type)

    def explicit_pool(self, specs: Iterable[MatchSpec]) -> tuple[str]:
        """
        Returns all the package names that (might) depend on the passed specs
        """
        explicit_pool: set[str] = set()
        for spec in specs:
            pkg_records = self.depends(spec.dist_str())
            for record in pkg_records:
                explicit_pool.add(record.name)
        return tuple(explicit_pool)

    def _process_query_result(
        self,
        result: QueryResult,
        return_type: Literal["records", "dict", "raw"] = "records",
    ) -> list[PackageRecord] | dict[str, Any] | QueryResult:
        if return_type == "raw":
            return result
        result = result.to_dict()
        if result.get("result", {}).get("status") != "OK":
            query_type = result.get("query", {}).get("type", "<Unknown>")
            query = result.get("query", {}).get("query", "<Unknown>")
            error_msg = result.get("result", {}).get("msg", f"Faulty response: {result.json()}")
            raise ValueError(f"{query_type} query '{query}' failed: {error_msg}")
        if return_type == "records":
            pkg_records = []
            for pkg in result["result"]["pkgs"]:
                record = PackageRecord(**pkg)
                pkg_records.append(record)
            return pkg_records
        # return_type == "dict"
        return result

    # endregion
