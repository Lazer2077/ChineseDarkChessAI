#!/bin/sh
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${JIEQI_NEURAL_PYTHON:-python3}" "$HERE/spatial_policy_uci.py" --device cuda "$@"
