#!/bin/bash

# This script assumes we are running in a Miniconda container where:
# - /opt/conda is the Miniconda or Miniforge installation directory
# - https://github.com/conda/conda is mounted at /workspaces/conda
# - https://github.com/conda/conda-libmamba-solver is mounted at
#   /workspaces/conda-libmamba-solver
# - https://github.com/mamba-org/mamba is (optionally) mounted at
#   /workspaces/mamba

set -euo pipefail

HERE=$(dirname $0)
BASE_CONDA=${BASE_CONDA:-/opt/conda}
SRC_CONDA=${SRC_CONDA:-/workspaces/conda}
SRC_CONDA_LIBMAMBA_SOLVER=${SRC_CONDA_LIBMAMBA_SOLVER:-/workspaces/conda-libmamba-solver}

if which apt-get > /dev/null; then
    echo "Installing system dependencies"
    apt-get update
    DEBIAN_FRONTEND=noninteractive xargs -a "$HERE/apt-deps.txt" apt-get install -y
fi


if [ ! -f "$SRC_CONDA/pyproject.toml" ]; then
    echo "https://github.com/conda/conda not found! Please clone or mount to $SRC_CONDA"
    exit 1
fi

# Clear history to avoid unneeded conflicts
echo "Clearing base history..."
echo '' > "$BASE_CONDA/conda-meta/history"

echo "Installing dev & test dependencies..."
"$BASE_CONDA/bin/conda" install -n base --yes --quiet \
    --file="$SRC_CONDA/tests/requirements.txt" \
    --file="$SRC_CONDA/tests/requirements-ci.txt" \
    --file="$SRC_CONDA/tests/requirements-Linux.txt" \
    --file="$SRC_CONDA/tests/requirements-s3.txt" \
    --file="$SRC_CONDA_LIBMAMBA_SOLVER/dev/requirements.txt" \
    --file="$SRC_CONDA_LIBMAMBA_SOLVER/tests/requirements.txt"\
    pre-commit
