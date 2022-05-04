import json
import pathlib
import os
import socket
import subprocess
import sys

import pytest
from xprocess import ProcessStarter

from conda.testing.integration import _get_temp_prefix


def _mock_server(xprocess, name, port, auth="none", user=None, password=None, token=None):
    """
    Adapted from
    https://github.com/mamba-org/powerloader/blob/effe2b7e1/test/helpers.py#L11
    """
    curdir = pathlib.Path(__file__).parent
    print("Starting mock_server")

    class Starter(ProcessStarter):

        pattern = f"Server started at localhost:{port}"
        terminate_on_interrupt = True
        timeout = 10
        args = [
            sys.executable,
            "-u",  # unbuffered
            str(curdir / "_reposerver.py"),
            "-d",
            str(curdir / "data" / "mamba_repo"),
            "--port",
            str(port),
        ]
        if auth == "token":
            assert token
            args += ["--token", token]
        elif auth:
            args += ["--auth", auth]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if user and password:
            env["TESTPWD"] = f"{user}:{password}"

        def startup_check(self):
            s = socket.socket()
            address = "localhost"
            error = False
            try:
                s.connect((address, port))
            except Exception as e:
                print("something's wrong with %s:%d. Exception is %s" % (address, port, e))
                error = True
            finally:
                s.close()

            return not error

    logfile = xprocess.ensure(name, Starter)

    if user and password:
        yield f"http://{user}:{password}@localhost:{port}"
    elif token:
        yield f"http://localhost:{port}/t/{token}"
    else:
        yield f"http://localhost:{port}"

    xprocess.getinfo(name).terminate()


@pytest.fixture
def server_auth_none(xprocess):
    yield from _mock_server(xprocess, name="server_auth_none", port=8000, auth="none")


@pytest.fixture
def server_auth_basic(xprocess):
    yield from _mock_server(
        xprocess, name="server_auth_basic", port=8000, auth="basic", user="user", password="test"
    )


@pytest.fixture
def server_auth_basic_email(xprocess):
    yield from _mock_server(
        xprocess,
        name="server_auth_basic_email",
        port=8000,
        auth="basic",
        user="user@email.com",
        password="test",
    )


@pytest.fixture
def server_auth_token(xprocess):
    yield from _mock_server(
        xprocess,
        name="server_auth_token",
        port=8000,
        auth="token",
        token="xy-12345678-1234-1234-1234-123456789012",
    )


def create_with_channel(channel):
    return subprocess.check_call(
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
            "--download-only",
            "-c",
            channel,
            "test-package",
        ]
    )


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
