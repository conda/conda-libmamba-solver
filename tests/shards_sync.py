# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from conda.base.context import context

from conda_libmamba_solver.index import LibMambaIndexHelper

if TYPE_CHECKING:
    from collections.abc import Iterable

    from conda.models.channel import Channel

# was conda-forge-sharded during testing
CONDA_FORGE_WITH_SHARDS = "conda-forge"

ROOT_PACKAGES = [
    "__archspec",
    "__conda",
    "__osx",
    "__unix",
    "bzip2",
    "ca-certificates",
    "expat",
    "icu",
    "libexpat",
    "libffi",
    "liblzma",
    "libmpdec",
    "libsqlite",
    "libzlib",
    "ncurses",
    "openssl",
    "pip",
    "python",
    "python_abi",
    "readline",
    "tk",
    "twine",
    "tzdata",
    "xz",
    "zlib",
]


def expand_channels(channels: list[Channel], subdirs: Iterable[str] | None = None):
    """
    Expand channels list into a dict of subdir-aware channels, matching
    LibMambaIndexHelper behavior.
    """
    subdirs_ = list(context.subdirs) if subdirs is None else subdirs
    channels_urls = LibMambaIndexHelper._channel_urls(subdirs_, channels)
    channels_urls = LibMambaIndexHelper._encoded_urls_to_channels(channels_urls)
    return channels_urls
