#!/bin/bash

# This script assumes we are running in a Miniconda container where:
# - /opt/conda is the Miniconda or Miniforge installation directory
# - https://github.com/conda/conda is mounted at /workspaces/conda
# - https://github.com/conda/conda-libmamba-solver is mounted at
#   /workspaces/conda-libmamba-solver
# - https://github.com/mamba-org/mamba is (optionally) mounted at
#   /workspaces/mamba

set -euo pipefail

MINICONDA=${MINICONDA:-/opt/conda}
CONDA_SRC=${CONDA_SRC:-/workspaces/conda}
CONDA_LIBMAMBA_SOLVER_SRC=${CONDA_LIBMAMBA_SOLVER_SRC:-/workspaces/conda-libmamba-solver}
MAMBA_SRC=${MAMBA_SRC:-/workspaces/mamba}

cat >> ~/.bashrc <<EOF
function develop-mamba() (
  # Runs in a subshell to avoid polluting the current shell
  set -euo pipefail
  if ! conda config --show channels | grep -q conda-forge; then
    echo "Miniconda not compatible with develop-mamba"
    exit 1
  fi
  if [ ! -f "$MAMBA_SRC/mamba/setup.py" ]; then
    echo "Mamba 1.x not found at $MAMBA_SRC"
    exit 1
  fi
  # Install mamba dev dependencies only once:
  if [ ! -f ~/.mamba-develop-installed ]; then
    # remove "sel(win)" in environment yaml hack since conda does not understand
    # libmamba specific specs
    echo "Installing mamba 1.x in dev mode..."
    sed '/sel(.*)/d' "$MAMBA_SRC/mamba/environment-dev.yml" > /tmp/mamba-environment-dev.yml
    CONDA_QUIET=1 "$MINICONDA/condabin/conda" env update -p "$MINICONDA" \
        --file /tmp/mamba-environment-dev.yml
    "$MINICONDA/condabin/conda" install make -yq  # missing in mamba's dev env
    # Clean build directory to avoid issues with stale build files
    test -f "$MAMBA_SRC/build/CMakeCache.txt" && rm -rf "$MAMBA_SRC/build"
  fi
  # Compile
  cd "$MAMBA_SRC"
  "$MINICONDA/bin/cmake" -B build/ \
      -DBUILD_LIBMAMBA=ON \
      -DBUILD_SHARED=ON \
      -DCMAKE_INSTALL_PREFIX="$MINICONDA" \
      -DCMAKE_PREFIX_PATH="$MINICONDA" \
      -DBUILD_LIBMAMBAPY=ON
  "$MINICONDA/bin/cmake" --build build/ -j\${NPROC:-2}
  if [ ! -f ~/.mamba-develop-installed ]; then
    "$MINICONDA/condabin/conda" remove -p "$MINICONDA" -yq --force libmambapy libmamba
  fi
  make install -C build/
  cd -
  "$MINICONDA/bin/pip" install -e "$MAMBA_SRC/libmambapy/" --no-deps
  test -f "$MINICONDA/conda-meta/mamba-"*".json" && "$MINICONDA/bin/pip" install -e "$MAMBA_SRC/mamba/" --no-deps
  touch ~/.mamba-develop-installed || true
)
EOF

cd "$CONDA_SRC"
echo "Initializing conda in dev mode..."
"$MINICONDA/bin/python" -m conda init --dev bash
cd -

echo "Installing conda-libmamba-solver in dev mode..."
"$MINICONDA/bin/python" -m pip install -e "$CONDA_LIBMAMBA_SOLVER_SRC" --no-deps

set -x
conda list -p "$MINICONDA"
conda info
conda config --show-sources
set +x
test -f "$MAMBA_SRC/mamba/setup.py" \
  && echo "Mamba mounted at $MAMBA_SRC; source ~/.bashrc and run develop-mamba() for dev-install"
