#!/bin/sh
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export JIEQI_FACTORIAL_VALUE=runs/observer-expert-policy-training-01/expert
exec "${JIEQI_NEURAL_PYTHON:-python3}" "$HERE/neural_policy_factorial_uci.py" --device cuda "$@"
