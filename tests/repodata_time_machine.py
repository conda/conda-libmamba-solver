# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
conda repodata time machine

Given a date and a channel, this script will:

- Download a local copy of the (unpatched) repodata
- Trim to the closest timestamp
- Download the closest repodata patches for that channel
- Apply the patches
- Generate a ready-to-use local channel
"""

import bz2
import json
import os
import urllib.request
from argparse import ArgumentParser
from datetime import datetime

import requests
from conda.base.context import context
from conda.models.channel import Channel
from conda_index.index import _apply_instructions
from conda_package_handling.api import extract as cph_extract

PATCHED_CHANNELS = {"defaults", "main", "conda-forge"}


def cli():
    p = ArgumentParser()
    p.add_argument("channels", nargs="+", metavar="channel")
    p.add_argument("-t", "--timestamp", required=True, help="YYYY-MM-DD HH:MM:SS. Assumes UTC.")
    p.add_argument(
        "-s",
        "--subdirs",
        default=f"{context.subdir},noarch",
        help="Comma-separated list of subdirs to download. Include 'noarch' explicitly if needed.",
    )
    return p.parse_args()


def download_repodata(channel, subdirs=None):
    "Download remote repodata JSON payload to a temporary location in disk"
    c = Channel(channel)
    if c.canonical_name in PATCHED_CHANNELS:
        repodata_fn = "repodata_from_packages"
    else:
        repodata_fn = "repodata"
    subdirs = subdirs or context.subdirs
    for url in c.urls(with_credentials=True, subdirs=subdirs):
        subdir = url.strip("/").split("/")[-1]
        urllib.request.urlretrieve(f"{url}/{repodata_fn}.json.bz2", f"{repodata_fn}.json.bz2")

        with open(f"{repodata_fn}.json.bz2", "rb") as f:
            with open(f"{repodata_fn}.json", "wb") as g:
                g.write(bz2.decompress(f.read()))

        yield f"{repodata_fn}.json", subdir


def trim_to_timestamp(repodata, timestamp: float):
    trimmed_tar_pkgs = {}
    trimmed_conda_pkgs = {}
    with open(repodata) as f:
        data = json.load(f)
        for name, pkg in data["packages"].items():
            if pkg.get("timestamp", 0) <= timestamp:
                trimmed_tar_pkgs[name] = pkg
        for name, pkg in data["packages.conda"].items():
            if pkg.get("timestamp", 0) <= timestamp:
                trimmed_conda_pkgs[name] = pkg
    data["packages"] = trimmed_tar_pkgs
    data["packages.conda"] = trimmed_conda_pkgs
    fn = f"trimmed.{os.path.basename(repodata)}"
    with open(fn, "w") as f:
        json.dump(data, f)
    return fn


def download_patches(channel, timestamp: float):
    name = Channel(channel).canonical_name
    if name != "conda-forge":
        raise NotImplementedError("Only conda-forge is supported for now")

    url = "https://api.anaconda.org/package/conda-forge/conda-forge-repodata-patches/files"
    r = requests.get(url)
    r.raise_for_status()
    pkgs = r.json()
    closest_older = None
    for pkg in sorted(pkgs, key=lambda pkg: pkg["attrs"]["timestamp"]):
        if pkg["attrs"]["timestamp"] <= timestamp:
            closest_older = pkg
        else:
            break
    if closest_older is None:
        raise ValueError(f"No patch found for timestamp {timestamp}")

    fn = closest_older["basename"].split("/")[-1]
    urllib.request.urlretrieve(f"https:{closest_older['download_url']}", fn)

    extract_path = f"conda-forge-repodata-patches-{closest_older['version']}"
    cph_extract(fn, dest_dir=extract_path)
    return extract_path


def apply_patch(repodata_file, patch):
    with open(repodata_file) as f, open(patch) as g:
        repodata = json.load(f)
        instructions = json.load(g)
    fn = f"patched.{os.path.basename(repodata_file)}"
    with open(fn, "w") as f:
        patched = _apply_instructions(None, repodata, instructions)
        json.dump(patched, f, indent=2)
    return fn


def repodata_time_machine(channels, timestamp_str, subdirs=None):
    horizon = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    timestamp = horizon.timestamp() * 1000
    original_dir = os.getcwd()
    try:
        workdir = f"repodata-{timestamp_str.replace(' ', '-').replace(':', '-').replace('.', '-')}"
        os.makedirs(workdir, exist_ok=True)
        os.chdir(workdir)
        # Download repodata
        for channel in channels:
            print("Rolling back", channel, "to", horizon)
            channel_name = Channel(channel).canonical_name
            os.makedirs(channel_name, exist_ok=True)
            os.chdir(channel_name)
            must_patch = channel_name in PATCHED_CHANNELS
            if must_patch:
                print("  Getting patches")
                patch_dir = os.path.abspath(download_patches(channel, timestamp))
            for repodata, subdir in download_repodata(channel, subdirs=subdirs):
                print("  Downloaded", repodata, "for", subdir)
                print("    Trimming...")
                abs_repodata = os.path.abspath(repodata)
                os.makedirs(subdir, exist_ok=True)
                os.chdir(subdir)
                trimmed = trim_to_timestamp(abs_repodata, timestamp)
                if must_patch:
                    print("    Patching...")
                    instructions = f"{patch_dir}/{subdir}/patch_instructions.json"
                    patched = apply_patch(trimmed, instructions)
                    if not os.path.exists("repodata.json"):
                        os.symlink(patched, "repodata.json")
                else:
                    if not os.path.exists("repodata.json"):
                        os.symlink(trimmed, "repodata.json")
                os.chdir("..")
            os.chdir("..")
        return workdir
    finally:
        os.chdir(original_dir)


def main():
    args = cli()
    return repodata_time_machine(args.channels, args.timestamp, args.subdirs.split(","))


if __name__ == "__main__":
    main()
    print("Done!")
