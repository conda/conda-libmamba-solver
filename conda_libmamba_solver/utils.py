# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
from functools import lru_cache
from logging import getLogger
from pathlib import Path
from urllib.parse import quote

import requests
from conda.base.context import context
from conda.common.compat import on_win
from conda.common.path import url_to_path
from conda.common.url import urlparse

log = getLogger(f"conda.{__name__}")


def escape_channel_url(channel):
    if channel.startswith("file:"):
        if "%" in channel:  # it's escaped already
            return channel
        if on_win:
            channel = channel.replace("\\", "/")
    parts = urlparse(channel)
    if parts.scheme:
        components = parts.path.split("/")
        if on_win:
            if parts.netloc and len(parts.netloc) == 2 and parts.netloc[1] == ":":
                # with absolute paths (e.g. C:/something), C:, D:, etc might get parsed as netloc
                path = "/".join([parts.netloc] + [quote(p) for p in components])
                parts = parts.replace(netloc="")
            else:
                path = "/".join(components[:2] + [quote(p) for p in components[2:]])
        else:
            path = "/".join([quote(p) for p in components])
        parts = parts.replace(path=path)
        return str(parts)
    return channel


@lru_cache(maxsize=None)
def is_channel_available(channel_url) -> bool:
    if context.offline:
        # We don't know where the channel might be (even file:// might be a network share)
        # so we play it safe and assume it's not available
        return False
    if channel_url.startswith("file://"):
        return Path(url_to_path(channel_url)).is_dir()
    return requests.head(f"{channel_url}/noarch/repodata.json").ok
