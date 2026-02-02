# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
import asyncio
import ctypes
import json
import signal
import subprocess as sp
import sys

import pytest
from conda.common.compat import on_win


def test_matchspec_star_version():
    """
    Specs like `libblas=*=*mkl` choked on `MatchSpec.conda_build_form()`.
    We work around that with `.utils.safe_conda_build_form()`.
    Reported in https://github.com/conda/conda/issues/11347
    """
    sp.check_call(
        [
            sys.executable,
            "-m",
            "conda",
            "create",
            "-p",
            "UNUSED",
            "--dry-run",
            "--override-channels",
            "-c",
            "conda-test",
            "--solver=libmamba",
            "activate_deactivate_package=*=*0",
        ]
    )


def test_build_string_filters():
    process = sp.run(
        [
            sys.executable,
            "-m",
            "conda",
            "create",
            "-p",
            "UNUSED",
            "--dry-run",
            "--solver=libmamba",
            "numpy=*=*py38*",
            "--json",
        ],
        stdout=sp.PIPE,
        text=True,
    )
    print(process.stdout)
    process.check_returncode()
    data = json.loads(process.stdout)
    assert data["success"]
    for pkg in data["actions"]["LINK"]:
        if pkg["name"] == "python":
            assert pkg["version"].startswith("3.8")
        if pkg["name"] == "numpy":
            assert "py38" in pkg["build_string"]


@pytest.mark.parametrize("shards", (True, False), ids=["shards", "noshards"])
@pytest.mark.parametrize("stage", ["Collecting package metadata", "Solving environment"])
def test_ctrl_c(stage, shards):
    TIMEOUT = 20

    async def run() -> tuple[str, str, int]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + TIMEOUT
        env = {
            "CONDA_PLUGINS_USE_SHARDED_REPODATA": str(shards),
            "PYTHONHASHSEED": str(0xAD792856),
        }
        p = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "conda",
            "create",
            "-p",
            "UNUSED",
            "--dry-run",
            "--solver=libmamba",
            "--override-channels",
            "--channel=conda-forge",
            "--quiet",
            "vaex",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        lines: list[str] = []
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                chunk = await asyncio.wait_for(p.stdout.readline(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if not chunk:
                break
            line = chunk.decode()
            lines.append(line)
            print(line.strip())
            if stage in line:
                break

        if loop.time() >= deadline:
            stdout_text = "".join(lines)
            raise RuntimeError(f"Timeout after {TIMEOUT} seconds\n{stdout_text}")

        # works around Windows' awkward CTRL-C signal handling
        # https://stackoverflow.com/a/64357453
        if on_win:
            try:
                kernel = ctypes.windll.kernel32
                kernel.FreeConsole()
                kernel.AttachConsole(p.pid)
                kernel.SetConsoleCtrlHandler(None, 1)
                kernel.GenerateConsoleCtrlEvent(0, 0)
                remaining = deadline - loop.time()
                await asyncio.wait_for(p.wait(), timeout=remaining)
            finally:
                kernel.SetConsoleCtrlHandler(None, 0)
        else:
            p.send_signal(signal.SIGINT)
            remaining = deadline - loop.time()
            await asyncio.wait_for(p.wait(), timeout=remaining)

        remaining = deadline - loop.time()
        if remaining <= 0:
            raise RuntimeError(f"Timeout after {TIMEOUT} seconds")
        stdout_tail = await asyncio.wait_for(p.stdout.read(), timeout=remaining)
        stderr_tail = await asyncio.wait_for(p.stderr.read(), timeout=remaining)
        stdout_text = "".join(lines) + stdout_tail.decode()
        stderr_text = stderr_tail.decode()
        return stdout_text, stderr_text, p.returncode

    stdout_text, stderr_text, returncode = asyncio.run(run())
    assert returncode != 0
    assert "KeyboardInterrupt" in stdout_text + stderr_text
