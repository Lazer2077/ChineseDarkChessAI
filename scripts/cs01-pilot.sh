#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root after selecting an available device.
: "${TRAIN_PYTHON:=python3}"
: "${TRAIN_DEVICE:=cpu}"
run="runs/pilot-$(date -u +%Y%m%dT%H%M%SZ)"
node --test
node src/run.js --games=4 --simulations=32 --horizon=4 --out="$run/data"
"$TRAIN_PYTHON" scripts/train.py --data="$run/data" --out="$run/model.json" --epochs=10 --device="$TRAIN_DEVICE"
node src/run.js --games=4 --seed=100001 --model="$run/model.json" --opponent=upstream --timeMs=100 --out="$run/eval"
