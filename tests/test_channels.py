import json
import sys
from subprocess import check_output, STDOUT

from conda.testing.integration import _get_temp_prefix, run_command, make_temp_env

from .channel_testing_utils import (
    create_with_channel,
    create_with_channel_in_process,
    http_server_auth_basic,
    http_server_auth_basic_email,
    http_server_auth_none,
    http_server_auth_token,
)


def test_http_server_auth_none(http_server_auth_none):
    create_with_channel(http_server_auth_none)


def test_http_server_auth_basic(http_server_auth_basic):
    create_with_channel(http_server_auth_basic)


def test_http_server_auth_basic_email(http_server_auth_basic_email):
    create_with_channel(http_server_auth_basic_email)


def test_http_server_auth_token(http_server_auth_token):
    create_with_channel(http_server_auth_token)


def test_channel_matchspec():
    stdout, stderr, _ = run_command(
        "create",
        _get_temp_prefix(),
        "--experimental-solver=libmamba",
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
    with make_temp_env("conda-forge::xz", "python", use_restricted_unicode=True) as prefix:
        output = check_output(
            [
                sys.executable,
                "-m",
                "conda",
                "install",
                "-yp",
                prefix,
                "pytest",
                "--experimental-solver=libmamba",
            ],
            stderr=STDOUT,
            text=True,
        )
        print(output)
        assert (
            "Selected channel specific (or force-reinstall) job, "
            "but package is not available from channel. "
            "Solve job will fail." not in output
        )
