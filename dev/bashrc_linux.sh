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

sudo /opt/conda/condabin/conda install -y -p /opt/conda \
    "flit-core>=3.2,<4" \
    --file /opt/conda-libmamba-solver-src/dev/requirements.txt \
    --file /opt/conda-libmamba-solver-src/tests/requirements.txt

cd /opt/conda-libmamba-solver-src
sudo env FLIT_ROOT_INSTALL=1 /opt/conda/bin/python -m flit install --symlink --deps=none

cd /opt/conda-src
export RUNNING_ON_DEVCONTAINER=${RUNNING_ON_DEVCONTAINER:-}
source /opt/conda-src/dev/linux/bashrc.sh

cd /opt/conda-libmamba-solver-src

set +e
