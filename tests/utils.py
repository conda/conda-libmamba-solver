# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
import sys
from pathlib import Path
from subprocess import CompletedProcess, run

from ruamel.yaml import YAML


def conda_subprocess(*args, explain=False, capture_output=True, **kwargs) -> CompletedProcess:
    cmd = [sys.executable, "-m", "conda", *[str(a) for a in args]]
    check = kwargs.pop("check", True)
    if explain:
        print("+", " ".join(cmd))
    p = run(
        cmd,
        capture_output=capture_output,
        text=kwargs.pop("text", capture_output),
        check=False,
        **kwargs,
    )
    if capture_output and (explain or p.returncode):
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
    if check:
        p.check_returncode()
    return p


def write_env_config(prefix, force=False, **kwargs):
    condarc = Path(prefix) / ".condarc"
    if condarc.is_file() and not force:
        raise RuntimeError(f"File {condarc} already exists. Use force=True to overwrite.")
    yaml = YAML(typ="full", pure=True)
    with open(condarc, "w") as f:
        yaml.dump(kwargs, f)
