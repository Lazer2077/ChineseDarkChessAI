#!/bin/sh
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${JIEQI_NEURAL_PYTHON:-python3}" "$HERE/neural_policy_fixed_work_uci.py" --strategy halving --simulations "${JIEQI_FIXED_WORK_SIMULATIONS:-256}" --device "${JIEQI_FIXED_WORK_DEVICE:-cuda}" "$@"
