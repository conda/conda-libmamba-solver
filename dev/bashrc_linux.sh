#!/bin/bash
# To be used with the conda/conda CI Docker images, possibly while developing locally

sudo /opt/conda/condabin/conda install -y -p /opt/conda \
    --file /opt/conda-libmamba-solver-src/dev/requirements.txt \
    --file /opt/conda-libmamba-solver-src/tests/requirements.txt && \
    /opt/conda/bin/python -m pip install /opt/conda-libmamba-solver-src --no-deps -vvv && \
    source /opt/conda-src/dev/linux/bashrc.sh &&
    cd /opt/conda-libmamba-solver-src
