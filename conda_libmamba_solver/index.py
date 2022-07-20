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
        self._channels = channels

        if subdirs is None:
            subdirs = context.subdirs
        self._subdirs = subdirs

        self._repos = []
        self._pool = api.Pool()

        installed_pool = self._load_installed(self._pool, installed_records)
        self._repos.append(installed_pool)

        self._index = load_channels(
            pool=self._pool,
            channels=self._channel_urls(self._channels),
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
        channel_urls = []
        for channel in channels:
            channel = Channel(channel)
            # Check channel support
            if channel.scheme == "s3":
                raise LibMambaChannelError(
                    f"'{channel}' is not yet supported on conda-libmamba-solver"
                )
            # This fixes test_activate_deactivate_modify_path_bash
            # and other local channels (path to url) issues
            for url in channel.urls(with_credentials=True):
                url = url.rstrip("/").rsplit("/", 1)[0]  # remove subdir
                channel_urls.append(escape_channel_url(url))

        if context.restore_free_channel:
            channel_urls.append("https://repo.anaconda.com/pkgs/free")

        # deduplicate
        channel_urls = list(OrderedDict.fromkeys(channel_urls))

        return tuple(channel_urls)

    @staticmethod
    def _load_installed(pool: api.Pool, installed_records: Iterable[PackageRecord] = ()):
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

        installed = api.Repo(pool, "installed", f.name, "")
        installed.add_extra_pkg_info(additional_infos)
        installed.set_installed()
        os.unlink(f.name)

        return installed

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
