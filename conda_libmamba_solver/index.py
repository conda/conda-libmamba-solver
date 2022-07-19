import os
from collections import OrderedDict
from tempfile import NamedTemporaryFile
from typing import Iterable, Union

from conda.base.constants import REPODATA_FN
from conda.base.context import context
from conda.common.constants import NULL
from conda.common.serialize import json_dump, json_load
from conda.common.url import (
    split_anaconda_token,
    remove_auth,
    percent_decode,
)
from conda.models.channel import Channel
from conda.models.match_spec import MatchSpec
from conda.models.records import PackageRecord
import libmambapy as api

from . import __version__
from .exceptions import LibMambaChannelError
from .mamba_utils import load_channels
from .state import IndexHelper
from .utils import escape_channel_url


class LibMambaIndexHelper(IndexHelper):
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
        if subdirs is None:
            subdirs = context.subdirs

        channel_urls = self._channel_urls(channels)

        self._repos = []
        self._pool = api.Pool()

        # export installed records to a temporary json file
        exported_installed = {"packages": {}}
        additional_infos = {}
        for record in installed_records:
            exported_installed["packages"][record.fn] = {
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
            f.write(json_dump(exported_installed))
        installed = api.Repo(self._pool, "installed", f.name, "")
        installed.add_extra_pkg_info(additional_infos)
        installed.set_installed()
        self._repos.append(installed)
        os.unlink(f.name)

        if channels is None:
            channels = context.channels
        if subdirs is None:
            subdirs = context.subdirs

        self._channels = channels
        self._subdirs = subdirs
        self._index = load_channels(
            pool=self._pool,
            channels=channel_urls,
            repos=self._repos,
            prepend=False,
            use_local=context.use_local,
            platform=subdirs,
            repodata_fn=repodata_fn,
        )

        self._query = api.Query(self._pool)
        self._format = api.QueryFormat.JSON

    @property
    def _channel_lookup(self):
        return {
            info["channel"].platform_url(info["platform"], with_credentials=False): info
            for _, info in self._index
        }

    @staticmethod
    def _channel_urls(channels: Iterable[Union[Channel, str]]):
        """
        TODO: libmambapy could handle path to url, and escaping
        but so far we are doing it ourselves
        """

        def _channel_to_url_or_name(channel):
            # This fixes test_activate_deactivate_modify_path_bash
            # and other local channels (path to url) issues
            urls = []
            for url in channel.urls(with_credentials=True):
                url = url.rstrip("/").rsplit("/", 1)[0]  # remove subdir
                urls.append(escape_channel_url(url))
            # deduplicate
            urls = list(OrderedDict.fromkeys(urls))
            return urls

        # Check channel support
        checked_channels = []
        for channel in channels:
            channel = Channel(channel)
            if channel.scheme == "s3":
                raise LibMambaChannelError(
                    f"'{channel}' is not yet supported on conda-libmamba-solver"
                )
            checked_channels.append(channel)

        channels = [url for _ in channels for url in _channel_to_url_or_name(checked_channels)]
        if context.restore_free_channel and "https://repo.anaconda.com/pkgs/free" not in channels:
            channels.append("https://repo.anaconda.com/pkgs/free")

        return tuple(channels)

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
