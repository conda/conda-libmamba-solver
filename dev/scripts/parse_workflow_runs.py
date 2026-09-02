#!/usr/bin/env python3
# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""Parse a GitHub Actions workflow and emit a shell script of its ``run:`` steps
to ease local testing of simple github actions workflows.

Usage:
    python dev/scripts/parse_workflow_runs.py [WORKFLOW] [OUTPUT]

Defaults:
    WORKFLOW = .github/workflows/pytest-conda-solvers.yml
    OUTPUT   = pytest-conda-solvers-steps.sh
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

DEFAULT_WORKFLOW = ".github/workflows/pytest-conda-solvers.yml"
DEFAULT_OUTPUT = "pytest-conda-solvers-steps.sh"

# Fallback values for ``${{ matrix.* }}`` expressions that cannot be resolved
# from the workflow file alone (the matrix is only expanded by GitHub).
MATRIX_DEFAULTS = {
    "python-version": "3.13",
    "os": "ubuntu-latest",
}

EXPR_RE = re.compile(r"\$\{\{\s*(\w+)\.([\w-]+)\s*\}\}")


def shell_var(name: str) -> str:
    """Convert a GitHub matrix key into a valid shell variable name."""
    return name.replace("-", "_").upper()


def substitute(text: str, env: dict, matrix_vars: set) -> str:
    """Replace ``${{ env.X }}`` and ``${{ matrix.X }}`` expressions."""

    def repl(match: re.Match) -> str:
        kind, name = match.group(1), match.group(2)
        if kind == "env":
            return str(env.get(name, f"${{{shell_var(name)}}}"))
        if kind == "matrix":
            matrix_vars.add(name)
            return f"${{{shell_var(name)}}}"
        return match.group(0)

    return EXPR_RE.sub(repl, text)


def main(argv: list[str]) -> int:
    workflow_path = Path(argv[0]) if len(argv) > 0 else Path(DEFAULT_WORKFLOW)
    output_path = Path(argv[1]) if len(argv) > 1 else Path(DEFAULT_OUTPUT)

    workflow = yaml.safe_load(workflow_path.read_text())
    env = workflow.get("env") or {}
    matrix_vars: set = set()

    chunks: list[str] = []
    for job_name, job in workflow["jobs"].items():
        chunks.append(f"# ---- job: {job_name} ----")
        chunks.append("")
        for step in job.get("steps", []):
            run = step.get("run")
            if run is None:
                continue
            if isinstance(run, list):
                run = "\n".join(run)
            run = str(run).rstrip("\n")

            chunks.append(f"# {step.get('name', '(unnamed)')}")
            if "if" in step:
                raise ValueError(
                    f"step conditions are not supported: {step['if']}"
                )
            if step.get("working-directory"):
                chunks.append(f"cd {step['working-directory']}")
            chunks.append(substitute(run, env, matrix_vars))
            chunks.append("")

    header = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# generated from {workflow_path}",
        "# run from the workspace root (the parent of conda-libmamba-solver)",
        "",
    ]
    if matrix_vars:
        header.append("# matrix defaults (override before running)")
        for var in sorted(matrix_vars):
            default = MATRIX_DEFAULTS.get(var, "")
            header.append(f"{shell_var(var)}=${{{shell_var(var)}:-{default}}}")
        header.append("")

    output_path.write_text("\n".join(header + chunks) + "\n")
    output_path.chmod(0o755)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
