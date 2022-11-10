#!/usr/bin/env bash

set -o errtrace -o pipefail -o errexit

eval "$(sudo /opt/conda/bin/python -m conda init --dev bash)"
conda info
python -c "from importlib_metadata import version; print('libmambapy', version('libmambapy'))"
# remove the pkg cache.  We can't hardlink from here anyway.  Having it around causes log problems.
sudo rm -rf /opt/conda/pkgs/*-*-*
python -m pytest -v tests/core/test_solve.py
