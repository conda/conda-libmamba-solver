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
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from conda.base.constants import REPODATA_FN
from conda.base.context import context
from conda.common.io import DummyExecutor, ThreadLimitedThreadPoolExecutor
from conda.core.package_cache_data import PackageCacheData
from conda.core.subdir_data import SubdirData
from conda.models.channel import Channel
from conda.models.match_spec import MatchSpec
from conda.models.records import PackageRecord
from libmambapy import ChannelContext, Context, Query
from libmambapy.solver.libsolv import (
    Database,
    PipAsPythonDependency,
    Priorities,
    RepodataOrigin,
    UseOnlyTarBz2,
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

if TYPE_CHECKING:
    from typing import Iterable, Literal

    from conda.gateways.repodata import RepodataState
    from libmambapy import QueryResult
    from libmambapy.solver.libsolv import RepoInfo


log = logging.getLogger(f"conda.{__name__}")


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
        subdirs: Iterable[str] = None,
        repodata_fn: str = REPODATA_FN,
        installed_records: Iterable[PackageRecord] = (),
        pkgs_dirs: Iterable[os.PathLike] = (),
    ):
        self.channels = channels
        self.subdirs = subdirs or context.subdirs
        self.repodata_fn = repodata_fn
        self.db = self._init_db()
        self.repos = self._load_channels()
        if pkgs_dirs:
            self.repos.extend(self._load_pkgs_cache(pkgs_dirs))
        if installed_records:
            self.repos.append(self._load_installed(installed_records))

        n_repos = len(self.repos)
        for i, repo in enumerate(self.repos):
            priority = n_repos - i
            subpriority = 0 if repo.name == "noarch" else 1
            self.db.set_repo_priority(repo, Priorities(priority, subpriority))

    def reload_channel(self, channel: Channel):
        raise NotImplementedError  # TODO

    def _init_db_from_context(self) -> Database:
        return Database(ChannelContext.make_conda_compatible(Context.instance()).params())

    def _init_db(self) -> Database:
        custom_channels = {
            name: LibmambaChannel(
                url=CondaURL.parse(channel.base_url),
                display_name=name,
                platforms=set(self.subdirs),
            )
            for (name, channel) in context.custom_channels.items()
            if channel.base_url
        }
        custom_multichannels = {
            channel_name: [
                custom_channels.get(
                    channel.name,
                    LibmambaChannel(
                        url=CondaURL.parse(channel.base_url),
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
        return Database(params)

    def _load_channels(self):
        urls_to_channel = self._channel_urls()
        urls_to_json_path_and_state = self._fetch_repodata_jsons(tuple(urls_to_channel.keys()))
        repos = []
        for url, (json_path, state) in urls_to_json_path_and_state.items():
            repos.append(self._load_repo_info_from_json_path(json_path, url, state))
        return repos

    def _channel_urls(self) -> dict[str, Channel]:
        urls = {}
        seen_noauth = set()
        channels_with_subdirs = []
        for channel in self.channels:
            if channel.subdir:
                channels_with_subdirs.append(channel)
                continue
            for url in channel.urls(with_credentials=True, subdirs=self.subdirs):
                channels_with_subdirs.append(Channel(url))
        for channel in channels_with_subdirs:
            noauth_urls = channel.urls(with_credentials=False, subdirs=(channel.subdir,))
            if seen_noauth.issuperset(noauth_urls):
                continue
            auth_urls = channel.urls(with_credentials=True, subdirs=(channel.subdir,))
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

    def _fetch_repodata_jsons(self, urls: dict[str, str]):
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

        subdir_data = SubdirData(channel, repodata_fn=self.repodata_fn)
        if context.offline or context.use_index_cache:
            # This might not exist (yet, anymore), but that's ok because we'll check
            # for existence later and safely ignore if needed
            json_path = subdir_data.cache_path_json
            state = None
        else:
            # TODO: This method loads reads the whole JSON file (does not parse)
            json_path, state = subdir_data.repo_fetch.fetch_latest_path()
        return url, json_path, state

    def _load_repo_info_from_json_path(
        self, json_path: str, channel_url: str, state: RepodataState
    ) -> RepoInfo:
        json_path = Path(json_path)
        solv_path = json_path.with_suffix(".solv")
        if state:
            repodata_origin = RepodataOrigin(url=channel_url, etag=state.etag, mod=state.mod)
        else:
            repodata_origin = None

        if repodata_origin:
            try:
                return self.db.add_repo_from_native_serialization(
                    path=str(json_path),
                    expected=repodata_origin,
                    channel_id=channel_url,
                    add_pip_as_python_dependency=context.add_pip_as_python_dependency,
                )
            except Exception as exc:
                log.debug("Could not load from serialized repdoata", exc_info=exc)

        repo = self.db.add_repo_from_repodata_json(
            path=str(json_path),
            url=channel_url,
            channel_id=channel_url,
            add_pip_as_python_dependency=PipAsPythonDependency(
                context.add_pip_as_python_dependency
            ),
            use_only_tar_bz2=UseOnlyTarBz2(context.use_only_tar_bz2),
        )
        if repodata_origin:
            self.db.native_serialize_repo(repo=repo, path=str(solv_path), metadata=repodata_origin)
        return repo

    def _load_installed(self, records: Iterable[PackageRecord]) -> Iterable[RepoInfo]:
        packages = [self._package_info_from_package_record(record) for record in records]
        repo = self.db.add_repo_from_packages(packages=packages, name="installed")
        self.db.set_installed_repo(repo)
        return repo

    def _load_pkgs_cache(self, pkgs_dirs: Iterable[os.PathLike]) -> Iterable[RepoInfo]:
        repos = []
        for path in pkgs_dirs:
            package_cache_data = PackageCacheData(path)
            package_cache_data.load()
            packages = [
                self._package_info_from_package_record(record)
                for record in package_cache_data.values()
            ]
            repo = self.db.add_repo_from_packages(packages=packages, name=path)
            repos.append(repo)
        return repos

    def _package_info_from_package_record(self, record: PackageRecord) -> PackageInfo:
        if record.get("noarch", None) and record.noarch.value in ("python", "generic"):
            noarch = NoArchType(record.noarch.value.title())
        else:
            noarch = NoArchType("No")
        return PackageInfo(
            name=record.name,
            version=record.version,
            build_string=record.build,
            build_number=record.build_number,
            channel=str(record.channel),
            package_url=record.get("url") or "",
            platform=record.subdir,
            filename=record.fn,
            license=record.get("license") or "",
            md5=record.md5,
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

    # region Repoquery
    #################

    def search(
        self,
        queries: Iterable[str | MatchSpec] | str | MatchSpec,
        return_type: Literal["records", "dict", "raw"] = "records",
    ) -> Iterable[PackageRecord] | dict | QueryResult:
        if isinstance(queries, (str, MatchSpec)):
            queries = [queries]
        result = Query.find(self.db, queries)
        return self._process_query_result(result, return_type)

    def depends(
        self,
        query: str | MatchSpec,
        tree: bool = False,
        return_type: Literal["records", "dict", "raw"] = "records",
    ) -> Iterable[PackageRecord] | dict | QueryResult:
        result = Query.depends(self.db, query, tree)
        return self._process_query_result(result, return_type)

    def whoneeds(
        self,
        query: str | MatchSpec,
        tree: bool = False,
        return_type: Literal["records", "dict", "raw"] = "records",
    ) -> Iterable[PackageRecord] | dict | QueryResult:
        result = Query.whoneeds(self.db, query, tree)
        return self._process_query_result(result, return_type)

    def explicit_pool(self, specs: Iterable[MatchSpec]) -> tuple[str]:
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
        result: QueryResult,
        return_type: Literal["records", "dict", "raw"] = "records",
    ) -> Iterable[PackageRecord] | dict | QueryResult:
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
