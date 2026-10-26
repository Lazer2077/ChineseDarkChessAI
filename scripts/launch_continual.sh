#!/usr/bin/env bash
# Copy this launcher to an immutable runs path before a background launch.
set -euo pipefail
: "${TRAIN_PYTHON:=python3}"
: "${REFEREE_PYTHON:=python3.11}"
: "${SERIES_DIR:=runs/continual-series-01}"
export OMP_NUM_THREADS=2
for seed in 20261026 20261027 20261028; do
  "$TRAIN_PYTHON" scripts/continual_learning.py --out="$SERIES_DIR/seed-$seed" --seed="$seed" --iterations=2 --games=8192 --val-games=512 --nodes=5000 --workers=4 --epochs=15 --batch=4096 --gate-games=200 --gate-tc=5+0.05 --referee-python="$REFEREE_PYTHON"
done
