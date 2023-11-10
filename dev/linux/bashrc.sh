#!/bin/bash
# To be used with the conda/conda CI Docker images, possibly while developing locally
# This script expects the following volumes to be present:
# - /opt/conda-src -> repo for conda/conda
# - /opt/conda-libmamba-src -> repo for conda/conda-libmamba-solver

set -e
restore_e() {
    set +e
}
trap restore_e EXIT

function recompile-mamba () {
  if [ -d "/opt/mamba-src" ]; then
    pushd /opt/mamba-src/
    sudo /opt/conda/bin/cmake -B build/ \
        -DBUILD_LIBMAMBA=ON \
        -DBUILD_SHARED=ON \
        -DCMAKE_INSTALL_PREFIX=/opt/conda \
        -DCMAKE_PREFIX_PATH=/opt/conda \
        -DBUILD_LIBMAMBAPY=ON
    sudo /opt/conda/bin/cmake --build build/ -j
    sudo make install -C build/
    popd
  else
    echo "ERROR: path '/opt/mamba-src' not mounted so no libmamba to recompile"
  fi
}

sudo /opt/conda/condabin/conda install -y -p /opt/conda --repodata-fn repodata.json \
    --file /opt/conda-libmamba-solver-src/dev/requirements.txt \
    --file /opt/conda-libmamba-solver-src/tests/requirements.txt

if [ -d "/opt/mamba-src" ]; then
  # remove "sel(win)" in environment yaml hack since conda does not understand
  # libmamba specific specs
  sed '/sel(.*)/d' /opt/mamba-src/mamba/environment-dev.yml > /tmp/mamba-environment-dev.yml
  sudo /opt/conda/condabin/conda env update -p /opt/conda \
       --file /tmp/mamba-environment-dev.yml

  sudo /opt/conda/condabin/conda remove -p /opt/conda -y --force libmambapy libmamba
  recompile-mamba
  sudo /opt/conda/bin/pip install -e /opt/mamba-src/libmambapy/ --no-deps
fi

cd /opt/conda-libmamba-solver-src
sudo /opt/conda/bin/python -m pip install -e . --no-deps

cd /opt/conda-src
export RUNNING_ON_DEVCONTAINER=${RUNNING_ON_DEVCONTAINER:-}
source /opt/conda-src/dev/linux/bashrc.sh

cd /opt/conda-libmamba-solver-src

set -x
conda list -p /opt/conda
conda info
conda config --show-sources
set +ex
