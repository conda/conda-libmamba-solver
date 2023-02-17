# Copyright (C) 2022 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause
import json
import sys
from subprocess import check_output, STDOUT, CalledProcessError

import pytest
from conda.testing.integration import _get_temp_prefix, make_temp_env, run_command


def test_channel_matchspec():
    stdout, stderr, _ = run_command(
        "create",
        _get_temp_prefix(),
        "--solver=libmamba",
        "--json",
        "--override-channels",
        "-c",
        "defaults",
        "conda-forge::libblas=*=*openblas",
        "python=3.9",
    )
    result = json.loads(stdout)
    assert result["success"] is True
    for record in result["actions"]["LINK"]:
        if record["name"] == "numpy":
            assert record["channel"] == "conda-forge"
        elif record["name"] == "python":
            assert record["channel"] == "pkgs/main"


def test_channels_prefixdata():
    """
    Make sure libmamba does not complain about missing channels
    used in previous commands.

    See https://github.com/conda/conda/issues/11790
    """
    with make_temp_env("conda-forge::xz", "python", "--solver=libmamba", use_restricted_unicode=True) as prefix:
        try:
            output = check_output(
            [
                sys.executable,
                "-m",
                "conda",
                "install",
                "-yp",
                prefix,
                "pytest",
                "--solver=libmamba",
            ],
            stderr=STDOUT,
            text=True,
        )
        except CalledProcessError as exc:
            print(exc.output)
            raise exc
        print(output)
        assert (
            "Selected channel specific (or force-reinstall) job, "
            "but package is not available from channel. "
            "Solve job will fail." not in output
        )
