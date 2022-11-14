import os
import logging
from tempfile import NamedTemporaryFile
from typing import Iterable, Union

from conda.base.constants import REPODATA_FN
from conda.base.context import context
from conda.common.io import ThreadLimitedThreadPoolExecutor
from conda.common.serialize import json_dump, json_load
from conda.core.subdir_data import SubdirData
from conda.models.channel import Channel
from conda.models.match_spec import MatchSpec
from conda.models.records import PackageRecord
import libmambapy as api

from . import __version__
from .mamba_utils import set_channel_priorities
from .state import IndexHelper
from .utils import escape_channel_url

log = logging.getLogger(f"conda.{__name__}")


class LibMambaIndexHelper(IndexHelper):
    _LIBMAMBA_PROTOCOLS = ("http", "https", "file", "", None)

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

        installed_records = list(installed_records)  # in case we are passed a lazy iterator
        installed_repo = self._repo_from_records(self._pool, "installed", installed_records)
        installed_repo.set_installed()
        self._repos.append(installed_repo)

        # See https://github.com/conda/conda/issues/11790
        for record in installed_records:
            if record.channel.auth or record.channel.token:
                # skip if the channel has authentication info, because
                # it might cause issues with expired tokens and what not
                continue
            if record.channel.name in ("@", "<develop>", "pypi"):
                # These "channels" are not really channels, more like
                # metadata placeholders
                continue
            self._channels.append(record.channel.subdir_url)

        free_channel = "https://repo.anaconda.com/pkgs/free"
        if context.restore_free_channel:
            self._channels.append(Channel(free_channel))

        self._channel_lookup = None
        self._index = self._load_channels()

        self._query = api.Query(self._pool)
        self._format = api.QueryFormat.JSON

    @staticmethod
    def _generate_channel_lookup(index):
        lookup = {}
        for _, info in index:
            try:  # libmamba api
                url = info["channel"].platform_url(info["platform"], with_credentials=False)
            except AttributeError:  # conda api
                url = info["channel"].urls(with_credentials=False, subdirs=(info["platform"],))[0]
            lookup[url] = info
        return lookup

    @staticmethod
    def _fix_channel_url(url):
        """
        Two fixes here:
        1. The subdir is sometimes appended to the channel URL, but this causes errors in
           test_activate_deactivate_modify_path_bash and local channels (path to url) issues
        2. Escape the URL so is %-encoded (spaces as %20, etc), so libcurl doesn't choke
        """
        parts = url.rstrip("/").rsplit("/", 1)  # try to remove subdir from url, if present
        if len(parts) == 2 and parts[1] in context.known_subdirs:
            url = parts[0]
        return escape_channel_url(url)

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
            # ref: https://github.com/mamba-org/mamba/blob/ad46f318b/libmamba/src/core/package_info.cpp#L276-L318
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

        repo = self._repo_from_json_path(pool, repo_name, f.name)
        repo.add_extra_pkg_info(additional_infos)
        os.unlink(f.name)

        return repo

    @staticmethod
    def _repo_from_json_path(pool: api.Pool, repo_name: str, path: str, url: str = "") -> api.Repo:
        return api.Repo(pool, repo_name, path, url)

    def _fetch_channel_url_with_conda(self, url: str):
        # We could load from subdir_data.iter_records(), but that means
        # re-exporting everything to a temporary JSON path and well,
        # `subdir_data.load()` already did!
        channel = Channel(url)
        try:
            subdir_data = _DownloadOnlySubdirData(channel, repodata_fn=self._repodata_fn)
            subdir_data.load()
            loaded_with = "conda_libmamba_solver.index.DownloadOnlySubdirData"
        except Exception as exc:
            log.debug(
                "Optimized DownloadOnlySubdirData call failed: %s\n"
                "Retrying with conda's SubdirData.",
                exc,
            )
            subdir_data = SubdirData(channel, repodata_fn=self._repodata_fn)
            subdir_data.load()
            loaded_with = "conda.core.subdir_data.SubdirData"
        repo = self._repo_from_json_path(
            self._pool,
            url,
            subdir_data.cache_path_json,
            url=url,
        )
        return (
            repo,
            {
                "platform": channel.subdir,
                "url": url,
                "channel": channel,
                "loaded_with": loaded_with,
            },
        )

    def _load_channels(self):
        """
        Handle all types of channels with a hybrid approach that uses
        libmamba whenever possible, but delegates to conda internals as
        a fallback. The important bit is to maintain order so priorities
        are computed adequately. This is not as easy as it looks, given
        the unspecified Channel spec.

        A user can pass a channel in several ways:
        - By name (e.g. defaults or conda-forge).
        - By URL (e.g. https://conda.anaconda.org/pytorch)
        - By URL + subdir (e.g. https://conda.anaconda.org/pytorch/noarch)

        All of those are acceptable, but libmamba will eventually expect a list
        of URLs with subdirs so it can delegate to libcurl. libcurl expects
        _escaped_ URLs (with %-encoded espaces, for example: %20). This means
        that libmamba will see a different URL compared to the user-provided one.
        libmamba might also modify the URL to include tokens.

        To sum up, we might obtain a different URL back, so we can't just blindly
        use the input URLs as dictionary keys to keep references. Instead, we will
        have to rely on the input order, very carefully.
        """
        urls = [
            url
            for c in self._channels
            for url in Channel(c).urls(with_credentials=True, subdirs=self._subdirs)
        ]
        urls = tuple(dict.fromkeys(urls))  # de-duplicate

        full_index = []
        with ThreadLimitedThreadPoolExecutor() as executor:
            for fetched in executor.map(self._fetch_channel_url_with_conda, urls):
                full_index.append(fetched)

        set_channel_priorities(self._pool, full_index, self._repos)

        self._channel_lookup = self._generate_channel_lookup(full_index)

        return full_index

    def whoneeds(self, query: str, records=True):
        result_str = self._query.whoneeds(query, self._format)
        return self._process_query_result(result_str, records=records)

    def depends(self, query: str, records=True):
        result_str = self._query.depends(query, self._format)
        return self._process_query_result(result_str, records=records)

    def search(self, query: str, records=True):
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

    def _process_query_result(self, result_str, records=True):
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


class _DownloadOnlySubdirData(SubdirData):
    _internal_state_template = {
        "_package_records": {},
        "_names_index": {},
        "_track_features_index": {},
    }

    def _read_local_repdata(self, *args):
        return self._internal_state_template

    def _process_raw_repodata_str(self, *args):
        return self._internal_state_template

    def _process_raw_repodata(self, *args):
        return self._internal_state_template

    def _pickle_me(self, *args):
        return
