# Copyright (C) 2019 QuantStack and the Mamba contributors.
# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause

# TODO: Temporarily vendored from mamba.utils v0.19 on 2021.12.02
# Decide what to do with it when we split into a plugin
# 2022.02.15: updated vendored parts to v0.21.2
# 2022.11.14: only keeping channel prioritization and context initialization logic now
# 2024.09.24: parameterize init_api_context

from __future__ import annotations

import os
import logging
import sys
from collections.abc import Iterable
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import libmambapy
from conda.base.constants import ChannelPriority
from conda.base.context import context
from conda.common.compat import on_win

if TYPE_CHECKING:
    from .index import _ChannelRepoInfo



log = logging.getLogger(f"conda.{__name__}")
_db_log = logging.getLogger("conda.libmamba.db")
_libmamba_context = None

@lru_cache(maxsize=1)
def mamba_version() -> str:
    return version("libmambapy")


def _get_base_url(url: str, name: str | None = None) -> str:
    tmp = url.rsplit("/", 1)[0]
    if name:
        if tmp.endswith(name):
            return tmp.rsplit("/", 1)[0]
    return tmp


def init_libmamba_context(
    channels: Iterable[str] | None = None,
    platform: str | None = None,
    target_prefix: str | None = None,
) -> libmambapy.Context:
    global _libmamba_context
    if _libmamba_context is None:
        # This function has to be called BEFORE 1st initialization of the context
        _libmamba_context = libmambapy.Context(
            libmambapy.ContextOptions(
                enable_signal_handling=False,
                enable_logging=True,
            )
        )
    libmamba_context = _libmamba_context

    # Output params
    libmamba_context.output_params.json = context.json
    if libmamba_context.output_params.json:
        libmambapy.cancel_json_output(libmamba_context)
    libmamba_context.output_params.quiet = context.quiet
    libmamba_context.output_params.verbosity = context.verbosity
    libmamba_context.set_log_level(
        {
            4: libmambapy.LogLevel.TRACE,
            3: libmambapy.LogLevel.DEBUG,
            2: libmambapy.LogLevel.INFO,
            1: libmambapy.LogLevel.WARNING,
            0: libmambapy.LogLevel.ERROR,
        }[context.verbosity]
    )

    # Prefix params
    libmamba_context.prefix_params.conda_prefix = context.conda_prefix
    libmamba_context.prefix_params.root_prefix = context.root_prefix
    libmamba_context.prefix_params.target_prefix = str(
        target_prefix if target_prefix is not None else context.target_prefix
    )

    # Networking params -- we always operate offline from libmamba's perspective
    libmamba_context.remote_fetch_params.user_agent = context.user_agent
    libmamba_context.local_repodata_ttl = context.local_repodata_ttl
    libmamba_context.offline = True
    libmamba_context.use_index_cache = True

    # General params
    libmamba_context.add_pip_as_python_dependency = context.add_pip_as_python_dependency
    libmamba_context.always_yes = context.always_yes
    libmamba_context.dry_run = context.dry_run
    libmamba_context.envs_dirs = context.envs_dirs
    libmamba_context.pkgs_dirs = context.pkgs_dirs
    libmamba_context.use_lockfiles = False
    libmamba_context.use_only_tar_bz2 = context.use_only_tar_bz2

    # Channels and platforms
    libmamba_context.platform = platform if platform is not None else context.subdir
    libmamba_context.channels = list(channels) if channels is not None else context.channels
    libmamba_context.channel_alias = str(
        _get_base_url(context.channel_alias.url(with_credentials=True))
    )

    RESERVED_NAMES = {"local", "defaults"}
    additional_custom_channels = {}
    for el in context.custom_channels:
        if context.custom_channels[el].canonical_name not in RESERVED_NAMES:
            additional_custom_channels[el] = _get_base_url(
                context.custom_channels[el].url(with_credentials=True), el
            )
    libmamba_context.custom_channels = additional_custom_channels

    additional_custom_multichannels = {
        "local": list(context.conda_build_local_paths),
        "defaults": [channel.url(with_credentials=True) for channel in context.default_channels],
    }
    for el in context.custom_multichannels:
        if el not in RESERVED_NAMES:
            additional_custom_multichannels[el] = []
            for c in context.custom_multichannels[el]:
                additional_custom_multichannels[el].append(
                    _get_base_url(c.url(with_credentials=True))
                )
    libmamba_context.custom_multichannels = additional_custom_multichannels

    libmamba_context.default_channels = [
        _get_base_url(x.url(with_credentials=True)) for x in context.default_channels
    ]

    if context.channel_priority is ChannelPriority.STRICT:
        libmamba_context.channel_priority = libmambapy.ChannelPriority.Strict
    elif context.channel_priority is ChannelPriority.FLEXIBLE:
        libmamba_context.channel_priority = libmambapy.ChannelPriority.Flexible
    elif context.channel_priority is ChannelPriority.DISABLED:
        libmamba_context.channel_priority = libmambapy.ChannelPriority.Disabled

    return libmamba_context


def logger_callback(level: libmambapy.solver.libsolv.LogLevel, msg: str, logger: logging.Logger =_db_log) -> None:
    # from libmambapy.solver.libsolv import LogLevel
    # levels = {
    #     LogLevel.Debug: logging.DEBUG, # 0 -> 10
    #     LogLevel.Warning: logging.WARNING, # 1 -> 30
    #     LogLevel.Error: logging.ERROR, # 2 -> 40
    #     LogLevel.Fatal: logging.FATAL, # 3 -> 50
    # }
    if level.value == 0:
        # This incurs a large performance hit!
        logger.debug(msg)
    else:
        logger.log((level.value + 2) * 10, msg)


def palettes_and_formats() -> tuple[libmambapy.solver.ProblemsMessageFormat, libmambapy.solver.ProblemsMessageFormat]:
    # _indents = ["│  ", "   ", "├─ ", "└─ "]
    if os.getenv("NO_COLOR"):
        use_color = False
    elif os.getenv("FORCE_COLOR"):
        use_color = True
    else:
        use_color = all([sys.stdout.isatty(), sys.stdin.isatty()])
    palette_no_color = libmambapy.Palette.no_color()
    problems_format_nocolor = libmambapy.solver.ProblemsMessageFormat()
    problems_format_nocolor.unavailable = palette_no_color.failure
    problems_format_nocolor.available = palette_no_color.success
    problems_format_auto = (
        libmambapy.solver.ProblemsMessageFormat()
        if use_color 
        else problems_format_nocolor
    )

    return problems_format_auto, problems_format_nocolor


problems_format_auto, problems_format_nocolor = palettes_and_formats()
