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
