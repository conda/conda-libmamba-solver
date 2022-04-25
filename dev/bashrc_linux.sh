#!/bin/bash
# To be used with the conda/conda CI Docker images, possibly while developing locally
# This script expects the following volumes to be present:
# - /opt/conda-src -> repo for conda/conda
# - /opt/conda-libmamba-src -> repo for conda-incubator/conda-libmamba-solver

set -euo pipefail

sudo /opt/conda/condabin/conda install -y -p /opt/conda \
    --file /opt/conda-libmamba-solver-src/dev/requirements.txt \
    --file /opt/conda-libmamba-solver-src/tests/requirements.txt

cd /opt/conda-libmamba-solver-src
sudo env FLIT_ROOT_INSTALL=1 /opt/conda/bin/python -m flit install --symlink --deps=none

cd /opt/conda-src
source /opt/conda-src/dev/linux/bashrc.sh

cd /opt/conda-libmamba-solver-src
