import sys
from subprocess import run, CompletedProcess

from ruamel.yaml import YAML
from pathlib import Path

def conda_subprocess(*args, explain=False, **kwargs) -> CompletedProcess:
    cmd = [sys.executable, "-m", "conda", *[str(a) for a in args]]
    if explain:
        print("+", " ".join(cmd))
    p = run(
        cmd,
        capture_output=True,
        text=True,
        **kwargs,
    )
    if p.returncode:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        p.check_returncode()
    return p


def write_env_config(prefix, force=False, **kwargs):
    condarc = Path(prefix) / ".condarc"
    if condarc.is_file() and not force:
        raise RuntimeError(f"File {condarc} already exists. Use force=True to overwrite.")
    yaml = YAML(typ='unsafe', pure=True)
    with open(condarc, "w") as f:
        yaml.dump(kwargs, f)