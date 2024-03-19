#!/bin/bash

# This script assumes we are running in a Miniconda container where:
# - /opt/conda is the Miniconda installation directory
# - https://github.com/conda/conda is mounted at /workspaces/conda
# - https://github.com/conda/conda-libmamba-solver is mounted at
#   /workspaces/conda-libmamba-solver
# - https://github.com/mamba-org/mamba is (optionally) mounted at
#   /workspaces/mamba

set -euo pipefail

CONDA_SRC=${CONDA_SRC:-/workspaces/conda}
CONDA_LIBMAMBA_SOLVER_SRC=${CONDA_LIBMAMBA_SOLVER_SRC:-/workspaces/conda-libmamba-solver}

# Clear history to avoid unneeded conflicts
echo "Clearing base history..."
echo '' > /opt/conda/conda-meta/history

if [ ! -f $CONDA_SRC/pyproject.toml ]; then
    echo "conda/conda not found, cloning..."
    git clone https://github.com/conda/conda $CONDA_SRC
fi

echo "Installing dev & test dependencies..."
/opt/conda/bin/conda install -n base --yes --quiet \
    --file="$CONDA_SRC/tests/requirements.txt" \
    --file="$CONDA_SRC/tests/requirements-ci.txt" \
    --file="$CONDA_SRC/tests/requirements-Linux.txt" \
    --file="$CONDA_SRC/tests/requirements-s3.txt" \
    --file="$CONDA_LIBMAMBA_SOLVER_SRC/dev/requirements.txt" \
    --file="$CONDA_LIBMAMBA_SOLVER_SRC/tests/requirements.txt"
