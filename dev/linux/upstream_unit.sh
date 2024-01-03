#!/usr/bin/env bash

set -o errtrace -o pipefail -o errexit

# CONDA LIBMAMBA SOLVER CHANGES
CONDA_SRC=${CONDA_SRC:-/opt/conda-src}
CONDA_LIBMAMBA_SOLVER_SRC=${CONDA_LIBMAMBA_SOLVER_SRC:-/opt/conda-libmamba-solver-src}
# /CONDA LIBMAMBA SOLVER CHANGES

### Prevent git safety errors when mounting directories ###
git config --global --add safe.directory /opt/conda-src

TEST_SPLITS="${TEST_SPLITS:-1}"
TEST_GROUP="${TEST_GROUP:-1}"

eval "$(sudo /opt/conda/bin/python -m conda init --dev bash)"
# make sure all test requirements are installed
# CONDA LIBMAMBA SOLVER CHANGES
sudo /opt/conda/bin/conda install --quiet -y --solver=classic --repodata-fn repodata.json \
    --file "${CONDA_SRC}/tests/requirements.txt" \
    --file "${CONDA_SRC}/tests/requirements-s3.txt" \
    --file "${CONDA_LIBMAMBA_SOLVER_SRC}/dev/requirements.txt"
sudo /opt/conda/bin/python -m pip install "$CONDA_LIBMAMBA_SOLVER_SRC" --no-deps -vvv
# /CONDA LIBMAMBA SOLVER CHANGES

conda info
# remove the pkg cache.  We can't hardlink from here anyway.  Having it around causes log problems.
sudo rm -rf /opt/conda/pkgs/*-*-*
# put temporary files on same filesystem
export TMP=$HOME/pytesttmp
mkdir -p $TMP
python -m pytest \
    --cov=conda \
    --durations-path=./tools/durations/Linux.json \
    --basetemp=$TMP \
    -m "not integration" \
    --splits=${TEST_SPLITS} \
    --group=${TEST_GROUP}
