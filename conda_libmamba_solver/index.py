import os
import logging
from tempfile import NamedTemporaryFile
from typing import Iterable, Union

from conda.base.constants import REPODATA_FN
from conda.base.context import context
from conda.common.io import Spinner
from conda.common.serialize import json_dump, json_load
from conda.core.subdir_data import SubdirData
from conda.models.channel import Channel
from conda.models.match_spec import MatchSpec
from conda.models.records import PackageRecord
import libmambapy as api

from . import __version__
from .mamba_utils import get_index, set_channel_priorities
from .state import IndexHelper
from .utils import escape_channel_url

log = logging.getLogger(f"conda.{__name__}")


class LibMambaIndexHelper(IndexHelper):
    _LIBMAMBA_PROTOCOLS = ("http://", "https://", "file://")

    def __init__(
        self,
        installed_records: Iterable[PackageRecord] = (),
        channels: Iterable[Union[Channel, str]] = None,
        subdirs: Iterable[str] = None,
        repodata_fn: str = REPODATA_FN,
    ):
        # Check channel support
        if channels is None:
            channels = context.channels
        self._channels = channels
        if subdirs is None:
            subdirs = context.subdirs
        self._subdirs = subdirs
        self._repodata_fn = repodata_fn

        self._repos = []
        self._pool = api.Pool()

        installed_repo = self._load_from_records(self._pool, "installed", installed_records)
        self._repos.append(installed_repo)

        self._index = self._load_channels()

        self._query = api.Query(self._pool)
        self._format = api.QueryFormat.JSON

    @property
    def _channel_lookup(self):
        return {
            info["channel"].platform_url(info["platform"], with_credentials=False): info
            for _, info in self._index
        }

    @staticmethod
    def _load_from_records(
        pool: api.Pool, repo_name: str, records: Iterable[PackageRecord] = ()
    ) -> api.Repo:
        # export installed records to a temporary json file
        exported = {"packages": {}}
        additional_infos = {}
        for record in records:
            exported["packages"][record.fn] = {
                **record.dist_fields_dump(),
                "depends": record.depends,
                "constrains": record.constrains,
                "build": record.build,
            }
            info = api.ExtraPkgInfo()
            if record.noarch:
                info.noarch = record.noarch.value
            if record.url:
                info.repo_url = record.url
            additional_infos[record.name] = info

        with NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write(json_dump(exported))

        installed = api.Repo(pool, repo_name, f.name, "")
        installed.add_extra_pkg_info(additional_infos)
        installed.set_installed()
        os.unlink(f.name)

        return installed

    def _load_channels(self):
        # Libmamba only supports a subset of the protocols offered in conda
        # We first filter the channel list to use the appropriate loaders
        # We will finally merge everything together in the right order
        libmamba_urls = []
        subdirdata_channels = []
        channel_loaders = []
        for i, channel in enumerate(self._channels):
            # channel can be str or Channel; make sure it's Channel!
            channel = Channel(channel)
            if channel.scheme and not channel.scheme.startswith(self._LIBMAMBA_PROTOCOLS):
                # protocol not supported by libmamba
                log.debug(
                    "Channel %s not supported by libmamba. Using conda.core.SubdirData", channel
                )
                for url in channel.urls(with_credentials=True):
                    channel_with_subdir = Channel(url)
                    subdirdata_channels.append(channel_with_subdir)
                    channel_loaders.append("conda")
            else:
                # This fixes test_activate_deactivate_modify_path_bash
                # and other local channels (path to url) issues
                for url in channel.urls(with_credentials=True):
                    url = url.rstrip("/").rsplit("/", 1)[0]  # remove subdir
                    libmamba_urls.append(escape_channel_url(url))
                    channel_loaders.append("libmamba")

        if context.restore_free_channel:
            libmamba_urls.append("https://repo.anaconda.com/pkgs/free")
            channel_loaders.append("libmamba")

        # delegate to libmamba loaders
        libmamba_index = get_index(
            channel_urls=libmamba_urls,
            prepend=False,
            use_local=context.use_local,
            platform=self._subdirs,
            repodata_fn=self._repodata_fn,
        )

        # load channels with conda, if needed
        conda_index = []
        for channel in subdirdata_channels:
            with Spinner(
                channel,
                enabled=not context.verbosity and not context.quiet,
                json=context.json,
            ):
                subdir_data = SubdirData(channel, repodata_fn=self._repodata_fn)
                subdir_data.load()
                repo = self._load_from_records(
                    self._pool, channel.name, subdir_data.iter_records()
                )
                conda_index.append(
                    (
                        repo,
                        {
                            "platform": channel.subdir,
                            "url": channel.url(with_credentials=True),
                            "channel": channel,
                            "loaded_with": "conda.core.subdir_data.SubdirData",
                        },
                    )
                )

        full_index = []
        iter_conda_index = iter(conda_index)
        iter_libmamba_index = iter(libmamba_index)
        for loader in channel_loaders:
            if loader == "conda":
                full_index.append(next(iter_conda_index))
            else:
                full_index.append(next(iter_libmamba_index))
        set_channel_priorities(self._pool, full_index, self._repos)
        return full_index

    def whoneeds(self, query: str):
        return self._query.whoneeds(query, self._format)

    def depends(self, query: str):
        return self._query.depends(query, self._format)

    def search(self, query: str):
        return self._query.find(query, self._format)

    def explicit_pool(self, specs: Iterable[MatchSpec]) -> Iterable[str]:
        """
        Returns all the package names that (might) depend on the passed specs
        """
        explicit_pool = set()
        for spec in specs:
            result_str = self.depends(spec.dist_str())
            result = json_load(result_str)
            for pkg in result["result"]["pkgs"]:
                explicit_pool.add(pkg["name"])
        return tuple(explicit_pool)
