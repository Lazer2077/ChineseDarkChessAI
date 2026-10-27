#!/bin/sh
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export JIEQI_FACTORIAL_VALUE=runs/neural-selfplay-series-01/iter-001/trained
exec "${JIEQI_NEURAL_PYTHON:-python3}" "$HERE/neural_policy_factorial_uci.py" --device cuda "$@"
