import json
import subprocess
import sys

from conda.testing.integration import _get_temp_prefix

from .channel_testing_utils import create_with_channel


def test_server_auth_none(server_auth_none):
    create_with_channel(server_auth_none)


def test_server_auth_basic(server_auth_basic):
    create_with_channel(server_auth_basic)


def test_server_auth_basic_email(server_auth_basic_email):
    create_with_channel(server_auth_basic_email)


def test_server_auth_token(server_auth_token):
    create_with_channel(server_auth_token)


def test_channel_matchspec():
    out = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "conda",
            "create",
            "-p",
            _get_temp_prefix(),
            "--experimental-solver=libmamba",
            "--json",
            "--override-channels",
            "-c",
            "defaults",
            "conda-forge::libblas=*=*openblas",
            "python=3.9",
        ]
    )
    print(out)
    result = json.loads(out)
    assert result["success"] is True
    for record in result["actions"]["LINK"]:
        if record["name"] == "numpy":
            assert record["channel"] == "conda-forge"
        elif record["name"] == "python":
            assert record["channel"] == "pkgs/main"
