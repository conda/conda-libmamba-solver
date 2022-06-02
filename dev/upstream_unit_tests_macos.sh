#!/usr/bin/env bash

set -o errtrace -o pipefail -o errexit

TEST_SPLITS="${TEST_SPLITS:-1}"
TEST_GROUP="${TEST_GROUP:-1}"

eval "$(sudo ${CONDA_PREFIX}/bin/python -m conda init bash --dev)"
conda info
python -c "from importlib_metadata import version; print('libmambapy', version('libmambapy'))"
python -m pytest -v tests/core/test_solve.py