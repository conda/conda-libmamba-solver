# Copyright (C) 2019, QuantStack
# SPDX-License-Identifier: BSD-3-Clause

# TODO: Temporarily vendored from mamba.utils v0.19 on 2021.12.02
# Decide what to do with it when we split into a plugin
# 2022.02.15: updated vendored parts to v0.21.2
# 2022.11.14: only keeping channel prioritization and context initialization logic now

import logging
import os
from importlib.metadata import version

import libmambapy as api
from conda.base.constants import ChannelPriority
from conda.base.context import context

log = logging.getLogger(f"conda.{__name__}")


def mamba_version():
    return version("libmambapy")


def set_channel_priorities(index, has_priority=None):
    """
    This function was part of load_channels originally.
    We just split it to reuse it a bit better.
    """
    if has_priority is None:
        has_priority = context.channel_priority in [
            ChannelPriority.STRICT,
            ChannelPriority.FLEXIBLE,
        ]

    subprio_index = len(index)
    if has_priority:
        # max channel priority value is the number of unique channels
        channel_prio = len({info.channel.canonical_name for info in index.values()})
        current_channel = next(iter(index.values())).channel.canonical_name

    for info in index.values():
        # add priority here
        if has_priority:
            if info.channel.canonical_name != current_channel:
                channel_prio -= 1
                current_channel = info.channel.canonical_name
            priority = channel_prio
        else:
            priority = 0
        if has_priority:
            # NOTE: -- this is the whole reason we are vendoring this file --
            # We are patching this from 0 to 1, starting with mamba 0.19
            # Otherwise, test_create::test_force_remove fails :shrug:
            subpriority = 1
        else:
            subpriority = subprio_index
            subprio_index -= 1

        if context.verbosity != 0 and not context.json:
            log.debug(
                "Channel: %s, platform: %s, prio: %s : %s",
                info.channel,
                info.channel.subdir,
                priority,
                subpriority,
            )
        info.repo.set_priority(priority, subpriority)

    return index


def init_api_context(use_mamba_experimental: bool = False):
    api_ctx = api.Context()

    api_ctx.json = context.json
    api_ctx.dry_run = context.dry_run
    if context.json:
        api.cancel_json_output()
        context.always_yes = True
        context.quiet = True
        if use_mamba_experimental:
            context.json = False

    api_ctx.verbosity = context.verbosity
    api_ctx.set_verbosity(context.verbosity)
    api_ctx.quiet = context.quiet
    api_ctx.offline = True
    api_ctx.local_repodata_ttl = context.local_repodata_ttl
    api_ctx.use_index_cache = True
    api_ctx.always_yes = context.always_yes
    api_ctx.channels = context.channels
    api_ctx.platform = context.subdir

    if "MAMBA_EXTRACT_THREADS" in os.environ:
        try:
            max_threads = int(os.environ["MAMBA_EXTRACT_THREADS"])
            api_ctx.extract_threads = max_threads
        except ValueError:
            v = os.environ["MAMBA_EXTRACT_THREADS"]
            raise ValueError(
                f"Invalid conversion of env variable 'MAMBA_EXTRACT_THREADS' from value '{v}'"
            )

    def get_base_url(url, name=None):
        tmp = url.rsplit("/", 1)[0]
        if name:
            if tmp.endswith(name):
                return tmp.rsplit("/", 1)[0]
        return tmp

    api_ctx.channel_alias = str(get_base_url(context.channel_alias.url(with_credentials=True)))

    additional_custom_channels = {}
    for el in context.custom_channels:
        if context.custom_channels[el].canonical_name not in ["local", "defaults"]:
            additional_custom_channels[el] = get_base_url(
                context.custom_channels[el].url(with_credentials=True), el
            )
    api_ctx.custom_channels = additional_custom_channels

    additional_custom_multichannels = {}
    for el in context.custom_multichannels:
        if el not in ["defaults", "local"]:
            additional_custom_multichannels[el] = []
            for c in context.custom_multichannels[el]:
                additional_custom_multichannels[el].append(
                    get_base_url(c.url(with_credentials=True))
                )
    api_ctx.custom_multichannels = additional_custom_multichannels

    api_ctx.default_channels = [
        get_base_url(x.url(with_credentials=True)) for x in context.default_channels
    ]

    if context.ssl_verify is False:
        api_ctx.ssl_verify = "<false>"
    elif context.ssl_verify is not True:
        api_ctx.ssl_verify = context.ssl_verify
    api_ctx.target_prefix = context.target_prefix
    api_ctx.root_prefix = context.root_prefix
    api_ctx.conda_prefix = context.conda_prefix
    api_ctx.pkgs_dirs = context.pkgs_dirs
    api_ctx.envs_dirs = context.envs_dirs

    api_ctx.connect_timeout_secs = int(round(context.remote_connect_timeout_secs))
    api_ctx.max_retries = context.remote_max_retries
    api_ctx.retry_backoff = context.remote_backoff_factor
    api_ctx.add_pip_as_python_dependency = context.add_pip_as_python_dependency
    api_ctx.use_only_tar_bz2 = context.use_only_tar_bz2

    if context.channel_priority is ChannelPriority.STRICT:
        api_ctx.channel_priority = api.ChannelPriority.kStrict
    elif context.channel_priority is ChannelPriority.FLEXIBLE:
        api_ctx.channel_priority = api.ChannelPriority.kFlexible
    elif context.channel_priority is ChannelPriority.DISABLED:
        api_ctx.channel_priority = api.ChannelPriority.kDisabled

    api_ctx.user_agent = context.user_agent
    api_ctx.use_lockfiles = False

    return api_ctx
