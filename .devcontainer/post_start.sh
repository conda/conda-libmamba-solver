#!/bin/bash

# This script assumes we are running in a Miniconda container where:
# - /opt/conda is the Miniconda installation directory
# - https://github.com/conda/conda is mounted at /workspaces/conda
# - https://github.com/conda/conda-libmamba-solver is mounted at
#   /workspaces/conda-libmamba-solver

set -euo pipefail

MINICONDA=${MINICONDA:-/opt/conda}
CONDA_SRC=${CONDA_SRC:-/workspaces/conda}
CONDA_LIBMAMBA_SOLVER_SRC=${CONDA_LIBMAMBA_SOLVER_SRC:-/workspaces/conda-libmamba-solver}
MAMBA_SRC=${MAMBA_SRC:-/workspaces/mamba}

function recompile-mamba () {
  if [ -f "$MAMBA_SRC/mamba/setup.py" ]; then
    cd "$MAMBA_SRC"
    sudo "$MINICONDA/bin/cmake" -B build/ \
        -DBUILD_LIBMAMBA=ON \
        -DBUILD_SHARED=ON \
        -DCMAKE_INSTALL_PREFIX="$MINICONDA" \
        -DCMAKE_PREFIX_PATH="$MINICONDA" \
        -DBUILD_LIBMAMBAPY=ON
    sudo "$MINICONDA/bin/cmake" --build build/ -j2
    sudo make install -C build/
    cd -
  else
    echo "ERROR: path '$MAMBA_SRC' not mounted so no libmamba to recompile"
  fi
}

# We only support mamba 1.x for now; v2 there's no 'mamba' subdirectory
if [ -f "$MAMBA_SRC/mamba/setup.py" ]; then
  # remove "sel(win)" in environment yaml hack since conda does not understand
  # libmamba specific specs
  sed '/sel(.*)/d' "$MAMBA_SRC/mamba/environment-dev.yml" > /tmp/mamba-environment-dev.yml
  sudo "$MINICONDA/condabin/conda" env update -p "$MINICONDA" \
       --file /tmp/mamba-environment-dev.yml

  sudo "$MINICONDA/condabin/conda" remove -p "$MINICONDA" -y --force libmambapy libmamba
  recompile-mamba
  sudo "$MINICONDA/bin/pip" install -e "$MAMBA_SRC/libmambapy/" --no-deps
fi

cd "$CONDA_SRC"
"$MINICONDA/bin/python" -m conda init --dev bash
cd -

"$MINICONDA/bin/python" -m pip install -e "$CONDA_LIBMAMBA_SOLVER_SRC" --no-deps

set -x
conda list -p "$MINICONDA"
conda info
conda config --show-sources
