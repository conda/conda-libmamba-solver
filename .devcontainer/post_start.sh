#!/bin/bash

# This script assumes we are running in a Miniconda container where:
# - /opt/conda is the Miniconda or Miniforge installation directory
# - https://github.com/conda/conda is mounted at /workspaces/conda
# - https://github.com/conda/conda-libmamba-solver is mounted at
#   /workspaces/conda-libmamba-solver
# - https://github.com/mamba-org/mamba is (optionally) mounted at
#   /workspaces/mamba

set -euo pipefail

BASE_CONDA=${BASE_CONDA:-/opt/conda}
SRC_CONDA=${SRC_CONDA:-/workspaces/conda}
SRC_CONDA_LIBMAMBA_SOLVER=${SRC_CONDA_LIBMAMBA_SOLVER:-/workspaces/conda-libmamba-solver}
SRC_MAMBA=${SRC_MAMBA:-/workspaces/mamba}

cat >> ~/.bashrc <<EOF
function develop-mamba() (
  # Runs in a subshell to avoid polluting the current shell
  set -euo pipefail
  if ! conda config --show channels | grep -q conda-forge; then
    echo "Miniconda not compatible with develop-mamba"
    exit 1
  fi
  if [ ! -f "$SRC_MAMBA/libmamba/CMakeLists.txt" ]; then
    echo "Could not find mamba-org/mamba at $SRC_MAMBA"
    exit 1
  fi
  if [ -f "$SRC_MAMBA/mamba/setup.py" ]; then
    echo "Mamba 1.x found at $SRC_MAMBA"
    mamba_version=1
    environment_yaml="$SRC_MAMBA/mamba/environment-dev.yml"
  else
    echo "Mamba 2.x found at $SRC_MAMBA"
    environment_yaml="$SRC_MAMBA/dev/environment-dev.yml"
    mamba_version=2
  fi
  # Install mamba dev dependencies only once:
  if [ ! -f ~/.mamba-develop-installed ]; then
    # remove "sel(win)" in environment yaml hack since conda does not understand
    # libmamba specific specs
    sed '/sel(.*)/d' "\$environment_yaml" > /tmp/mamba-environment-dev.yml
    # Environment.yml is missing make
    echo "  - make" >> /tmp/mamba-environment-dev.yml
    "$BASE_CONDA/condabin/conda" env update \
      --quiet \
      --prefix "$BASE_CONDA" \
      --file /tmp/mamba-environment-dev.yml
    # Clean build directory to avoid issues with stale build files
    test -f "$SRC_MAMBA/build/CMakeCache.txt" && rm -rf "$SRC_MAMBA/build"
  fi
  # Compile
  cd "$SRC_MAMBA"
  "$BASE_CONDA/bin/cmake" -B build/ \
      -DBUILD_LIBMAMBA=ON \
      -DBUILD_SHARED=ON \
      -DCMAKE_INSTALL_PREFIX="$BASE_CONDA" \
      -DCMAKE_PREFIX_PATH="$BASE_CONDA" \
      -DBUILD_LIBMAMBAPY=ON
  "$BASE_CONDA/bin/cmake" --build build/ -j\${NPROC:-2}
  if [ ! -f ~/.mamba-develop-installed ]; then
    "$BASE_CONDA/condabin/conda" remove -p "$BASE_CONDA" -yq --force libmambapy libmamba
  fi
  make install -C build/
  cd -
  "$BASE_CONDA/bin/pip" install -e "$SRC_MAMBA/libmambapy/" --no-deps
  if [ "\$mamba_version" == "1" ]; then
    test -f "$BASE_CONDA/conda-meta/mamba-"*".json" && "$BASE_CONDA/bin/pip" install -e "$SRC_MAMBA/mamba/" --no-deps
  else
    echo "Mamba binary installation not supported yet"
  fi
  touch ~/.mamba-develop-installed || true
)
EOF

cd "$SRC_CONDA"
echo "Initializing conda in dev mode..."
"$BASE_CONDA/bin/python" -m conda init --dev bash
cd -

echo "Installing conda-libmamba-solver in dev mode..."
"$BASE_CONDA/bin/python" -m pip install -e "$SRC_CONDA_LIBMAMBA_SOLVER" --no-deps

set -x
conda list -p "$BASE_CONDA"
conda info
conda config --show-sources
set +x
test -f "$SRC_MAMBA/mamba/setup.py" \
  && echo "Mamba mounted at $SRC_MAMBA; source ~/.bashrc and run develop-mamba() for dev-install"
