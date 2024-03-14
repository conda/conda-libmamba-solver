#!/bin/bash

# This script assumes we are running in a Miniconda container where:
# - /opt/conda is the Miniconda installation directory
# - https://github.com/conda/conda is mounted at /workspaces/conda
# - https://github.com/conda/conda-libmamba-solver is mounted at
#   /workspaces/conda-libmamba-solver

set -euo pipefail

CONDA_SRC=${CONDA_SRC:-/workspaces/conda}
CONDA_LIBMAMBA_SOLVER_SRC=${CONDA_LIBMAMBA_SOLVER_SRC:-/workspaces/conda-libmamba-solver}

# Clear history to avoid unneeded conflicts
echo '' > /opt/conda/conda-meta/history

# Install all dependencies
/opt/conda/bin/conda install -n base --yes --quiet \
    --override-channels --channel=defaults \
    --file="$CONDA_SRC/tests/requirements.txt" \
    --file="$CONDA_SRC/tests/requirements-ci.txt" \
    --file="$CONDA_SRC/tests/requirements-Linux.txt" \
    --file="$CONDA_SRC/tests/requirements-s3.txt" \
    --file="$CONDA_LIBMAMBA_SOLVER_SRC/dev/requirements.txt" \
    --file="$CONDA_LIBMAMBA_SOLVER_SRC/tests/requirements.txt"
