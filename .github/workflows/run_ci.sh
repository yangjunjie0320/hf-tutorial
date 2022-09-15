#!/usr/bin/env bash

set -e

if [ "$RUNNER_OS" == "Linux" ]; then
    os='linux'
else
    echo "$RUNNER_OS not supported"
    exit 1
fi

#!/usr/bin/env bash
export OMP_NUM_THREADS=4
export PYTHONPATH=$(pwd):$PYTHONPATH 
ulimit -s 20000

pytest
