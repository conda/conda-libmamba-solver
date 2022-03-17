#!/bin/bash
# File adapted from mamba-org/mamba, commit bff16c2bdc4103ba74c23ab4fdbf58849a55981c, on Mar 17 2022
# See data/mamba_repo/LICENSE for full details

set -xuo pipefail

rm -rf $CONDA_PREFIX/pkgs/test-package*
ENV_NAME=testauth

python _reposerver.py -d data/mamba_repo/ --auth none & PID=$!
conda create --experimental-solver=libmamba \
    -y -q -n $ENV_NAME --override-channels \
    --download-only \
    -c http://localhost:8000/ test-package --json
exit_code=$?
kill -TERM $PID
[[ $exit_code != "0" ]] && exit $exit_code


export TESTPWD="user:test"
python _reposerver.py -d data/mamba_repo/ --auth basic & PID=$!
conda create --experimental-solver=libmamba \
    -y -q -n $ENV_NAME --override-channels \
    --download-only \
    -c http://user:test@localhost:8000/ test-package --json
exit_code=$?
kill -TERM $PID
[[ $exit_code != "0" ]] && exit $exit_code

export TESTPWD="user@email.com:test"
python _reposerver.py -d data/mamba_repo/ --auth basic & PID=$!
conda create --experimental-solver=libmamba \
    -y -q -n $ENV_NAME --override-channels \
    --download-only \
    -c http://user@email.com:test@localhost:8000/ test-package --json
exit_code=$?
kill -TERM $PID
[[ $exit_code != "0" ]] && exit $exit_code

python _reposerver.py -d data/mamba_repo/ --auth token & PID=$!
conda create --experimental-solver=libmamba \
    -y -q -n $ENV_NAME --override-channels \
    --download-only \
    -c http://localhost:8000/t/xy-12345678-1234-1234-1234-123456789012 test-package --json
exit_code=$?
kill -TERM $PID
[[ $exit_code != "0" ]] && exit $exit_code

# if [[ "$(uname -s)" == "Linux" ]]; then
# 	export KEY1=$(gpg --fingerprint "MAMBA1")
# 	export KEY2=$(gpg --fingerprint "MAMBA2")
#     set -x
# 	python _reposerver.py -d data/repo/ --auth none --sign & PID=$!
# 	sleep 5s
#     kill -TERM $PID
# fi

exit 0
