#!/usr/bin/env bash
set -euo pipefail
: "${TRAIN_PYTHON:=python3}"
: "${TRAIN_DEVICE:=cpu}"
: "${EXPERIMENT_DIR:=runs/first-experiment}"
node --test
node src/run.js --games=8 --seed=20261026 --simulations=128 --horizon=4 --out="$EXPERIMENT_DIR/gen0"
"$TRAIN_PYTHON" scripts/train.py --data="$EXPERIMENT_DIR/gen0" --out="$EXPERIMENT_DIR/model0.json" --epochs=20 --device="$TRAIN_DEVICE"
node src/run.js --games=8 --seed=20262026 --simulations=128 --horizon=4 --model="$EXPERIMENT_DIR/model0.json" --out="$EXPERIMENT_DIR/gen1"
"$TRAIN_PYTHON" scripts/train.py --data="$EXPERIMENT_DIR/gen1" --out="$EXPERIMENT_DIR/model1.json" --epochs=20 --device="$TRAIN_DEVICE"
node src/run.js --games=8 --seed=30361026 --simulations=128 --horizon=4 --model="$EXPERIMENT_DIR/model1.json" --opponent=upstream --timeMs=100 --out="$EXPERIMENT_DIR/eval"
