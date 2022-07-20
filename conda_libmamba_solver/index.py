import os
import logging
from collections import defaultdict
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
    _LIBMAMBA_PROTOCOLS = ("http", "https", "file")

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

        self._channel_lookup = None
        self._index = self._load_channels()

        self._query = api.Query(self._pool)
        self._format = api.QueryFormat.JSON

    @staticmethod
    def _generate_channel_lookup(index):
        return {
            info["channel"].platform_url(info["platform"], with_credentials=True): info
            for _, info in index
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
        conda_urls = []
        channels_to_urls = defaultdict(list)
        for channel in self._channels:
            # channel can be str or Channel; make sure it's Channel!
            channel_obj = Channel(channel)
            channel_urls = channel.urls(with_credentials=True, subdirs=self._subdirs)
            if channel_obj.scheme and not channel_obj.scheme in self._LIBMAMBA_PROTOCOLS:
                # These channels are loaded with conda
                log.debug(
                    "Channel %s not supported by libmamba. Using conda.core.SubdirData",
                    channel_obj,
                )
                for url in channel_urls:
                    channels_to_urls[channel].append(url)
                    conda_urls.append(url)
            else:
                # These channels are loaded with libmamba
                for url in channel_urls:
                    # This fixes test_activate_deactivate_modify_path_bash
                    # and other local channels (path to url) issues
                    channels_to_urls[channel].append(url)
                    url = url.rstrip("/").rsplit("/", 1)[0]  # remove subdir
                    url = escape_channel_url(url)
                    libmamba_urls.append(url)

        free_channel = "https://repo.anaconda.com/pkgs/free"
        if context.restore_free_channel and free_channel not in libmamba_urls:
            libmamba_urls.append(free_channel)

        # delegate to libmamba loaders
        full_index = {
            entry["url"]: (subdir, entry)
            for (subdir, entry) in get_index(
                channel_urls=libmamba_urls,
                prepend=False,
                use_local=context.use_local,
                platform=self._subdirs,
                repodata_fn=self._repodata_fn,
            )
        }

        # load channels with conda, if needed
        for url in conda_urls:
            with Spinner(
                url,
                enabled=not context.verbosity and not context.quiet,
                json=context.json,
            ):
                subdir_data = SubdirData(url, repodata_fn=self._repodata_fn)
                subdir_data.load()
                channel = Channel(url)
                repo = self._load_from_records(self._pool, url, subdir_data.iter_records())
                full_index[url] = (
                    repo,
                    {
                        "platform": channel.subdir,
                        "url": url,
                        "channel": channel,
                        "loaded_with": "conda.core.subdir_data.SubdirData",
                    },
                )

        prioritized_index = []
        for channel in self._channels:
            for url in channels_to_urls[channel]:
                # Not all calculated URLs have to exist! Some platforms might be missing
                # Also, the URL might have been escaped for proper curl handling; try with that too
                index_entry = full_index.get(url, full_index.get(escape_channel_url(url)))
                if index_entry is None:
                    log.debug(
                        f"URL '{url}' for channel '{channel}' does not exist or could not be fetched. "
                        "Skipping..."
                    )
                else:
                    prioritized_index.append(index_entry)

        # This one is added above by us, but it's not in 'self._channels'; add manually again
        if free_channel in libmamba_urls:
            for url in Channel(free_channel).urls(with_credentials=True, subdirs=self._subdirs):
                # in this case all urls are guaranteed to exist in the free channel
                prioritized_index.append(full_index[url])

        set_channel_priorities(self._pool, prioritized_index, self._repos)

        self._channel_lookup = self._generate_channel_lookup(prioritized_index)

        return prioritized_index

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
