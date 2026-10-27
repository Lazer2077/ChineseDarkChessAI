#!/bin/sh
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${JIEQI_NEURAL_PYTHON:-python3}" "$HERE/neural_policy_graph_uci.py" --device cuda "$@"
