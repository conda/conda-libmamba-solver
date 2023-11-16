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

sudo su root -c "/opt/conda/bin/conda install -yq conda-build"
# make sure all test requirements are installed
# CONDA LIBMAMBA SOLVER CHANGES
sudo /opt/conda/bin/conda install --quiet -y --solver=classic --repodata-fn repodata.json \
    --file "${CONDA_SRC}/tests/requirements.txt" \
    --file "${CONDA_SRC}/tests/requirements-s3.txt" \
    --file "${CONDA_LIBMAMBA_SOLVER_SRC}/dev/requirements.txt"
sudo /opt/conda/bin/python -m pip install "$CONDA_LIBMAMBA_SOLVER_SRC" --no-deps -vvv
# /CONDA LIBMAMBA SOLVER CHANGES
eval "$(sudo /opt/conda/bin/python -m conda init --dev bash)"
conda-build tests/test-recipes/activate_deactivate_package tests/test-recipes/pre_link_messages_package
conda info
# put temporary files on same filesystem
# put temporary files on same filesystem
export TMP=$HOME/pytesttmp
mkdir -p $TMP
python -m pytest \
    --cov=conda \
    --durations-path=./tools/durations/Linux.json \
    --basetemp=$TMP \
    -m "integration" \
    --splits=${TEST_SPLITS} \
    --group=${TEST_GROUP}
