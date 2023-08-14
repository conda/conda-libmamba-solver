# Copyright (C) 2019, QuantStack
# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import json
import sys

from conda.base.context import context
from conda.cli import conda_argparse
from conda.common.io import Spinner
from conda.core.prefix_data import PrefixData
from libmambapy import QueryFormat

from .index import LibMambaIndexHelper


def configure_parser(parser: argparse.ArgumentParser):
    package_cmds = argparse.ArgumentParser(add_help=False)
    package_cmds.add_argument("package_query", help="the target package")
    package_cmds.add_argument(
        "-i",
        "--installed",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS,
    )

    package_cmds.add_argument("-p", "--platform", default=context.subdir)
    package_cmds.add_argument("--no-installed", action="store_true")
    package_cmds.add_argument("--pretty", action="store_true")

    package_cmds.add_argument(
        "-a",
        "--all-channels",
        action="store_true",
        help="Look at all channels (for depends / whoneeds)",
    )

    view_cmds = argparse.ArgumentParser(add_help=False)
    view_cmds.add_argument("-t", "--tree", action="store_true")
    view_cmds.add_argument("--recursive", action="store_true")

    subparser = parser.add_subparsers(dest="subcmd")

    c1 = subparser.add_parser(
        "whoneeds",
        help="shows packages that depend on this package",
        parents=[package_cmds, view_cmds],
    )

    c2 = subparser.add_parser(
        "depends",
        help="shows dependencies of this package",
        parents=[package_cmds, view_cmds],
    )

    c3 = subparser.add_parser(
        "search",
        help="shows all available package versions",
        parents=[package_cmds],
    )

    for cmd in (c1, c2, c3):
        conda_argparse.add_parser_channels(cmd)
        conda_argparse.add_parser_networking(cmd)
        conda_argparse.add_parser_known(cmd)
        conda_argparse.add_parser_json(cmd)


def repoquery(args):
    if not args.subcmd:
        print("repoquery needs a subcommand (search, depends or whoneeds)", file=sys.stderr)
        print("eg:", file=sys.stderr)
        print("    $ mamba repoquery search xtensor\n", file=sys.stderr)
        sys.exit(1)

    channels = None
    if hasattr(args, "channel"):
        channels = args.channel
    if args.all_channels or (channels is None and args.subcmd == "search"):
        if channels:
            print("WARNING: Using all channels instead of configured channels\n", file=sys.stderr)
        channels = context.channels

    use_installed = args.installed
    if args.no_installed:
        use_installed = False

    # if we're asking for depends and channels are given, disregard
    # installed packages to prevent weird mixing
    if args.subcmd in ("depends", "whoneeds") and use_installed and channels:
        use_installed = False

    only_installed = True
    if args.subcmd == "search" and not args.installed:
        only_installed = False
    elif args.all_channels or (channels and len(channels)):
        only_installed = False

    if only_installed and args.no_installed:
        print("No channels selected.", file=sys.stderr)
        print("Activate -a to search all channels.", file=sys.stderr)
        sys.exit(1)

    if use_installed:
        spinner_msg = f"Loading installed packages ({context.target_prefix})"
        prefix_data = PrefixData(context.target_prefix)
        prefix_data.load()
        installed_records = prefix_data.iter_records()
    else:
        installed_records = ()
        if channels:
            names = ",".join([getattr(c, "canonical_name", c) for c in channels])
            spinner_msg = f"Loading {args.platform} channels ({names})"

    if context.json:
        query_format = QueryFormat.JSON
    elif getattr(args, "tree", None):
        query_format = QueryFormat.TREE
    elif getattr(args, "recursive", None):
        query_format = QueryFormat.RECURSIVETABLE
    elif getattr(args, "pretty", None):
        query_format = QueryFormat.PRETTY
    else:
        query_format = QueryFormat.TABLE

    with Spinner(
        spinner_msg,
        enabled=not context.verbosity and not context.quiet,
        json=context.json,
    ):
        index = LibMambaIndexHelper(
            installed_records=installed_records,
            channels=channels,
            subdirs=(args.platform, "noarch"),
            repodata_fn=context.repodata_fns[-1],
            query_format=query_format,
        )

    result = getattr(index, args.subcmd)(args.package_query, records=False)
    if context.json:
        print(json.dumps(result, indent=2))
    else:
        print(result)
