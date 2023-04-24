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

# remove "sel(win)" in environment yaml hack since conda does not understand
# libmamba specific specs
sed '/sel/d' /opt/mamba-src/mamba/environment-dev.yml > /tmp/mamba-environment-dev.yml
sudo /opt/conda/condabin/conda env update -p /opt/conda \
     --file /tmp/mamba-environment-dev.yml

cd /opt/mamba-src/
sudo /opt/conda/bin/cmake -B build/ \
    -DBUILD_LIBMAMBA=ON \
    -DBUILD_SHARED=ON \
    -DCMAKE_INSTALL_PREFIX=/opt/conda \
    -DCMAKE_PREFIX_PATH=/opt/conda \
    -DBUILD_LIBMAMBAPY=ON
sudo /opt/conda/bin/cmake --build build/ -j
sudo make install -C build/
sudo /opt/conda/bin/pip install -e libmambapy/ --no-deps

# remove libmamba in environment yaml since already installed in develop
sed '/libmamba/d' /opt/conda-libmamba-solver-src/dev/requirements.txt > /tmp/conda-libmamba-solver-requirements.txt
sudo /opt/conda/condabin/conda install -y -p /opt/conda \
    --file /tmp/conda-libmamba-solver-requirements.txt \
    --file /opt/conda-libmamba-solver-src/tests/requirements.txt

cd /opt/conda-libmamba-solver-src
sudo /opt/conda/bin/python -m pip install -e . --no-deps

cd /opt/conda-src
export RUNNING_ON_DEVCONTAINER=${RUNNING_ON_DEVCONTAINER:-}
source /opt/conda-src/dev/linux/bashrc.sh

cd /opt/conda-libmamba-solver-src

set +e
