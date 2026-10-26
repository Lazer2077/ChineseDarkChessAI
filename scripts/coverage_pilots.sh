#!/usr/bin/env bash
# Developer screen only. Exact binary, source and net hashes saved per match.
set -euo pipefail
: "${REFEREE_PYTHON:=python3.11}"
: "${JIEQI_PYTHON:=$REFEREE_PYTHON}"
export JIEQI_PYTHON
export MUSESFISH_DIR="$PWD/external/musesfish"
export UNIVERSE_JIEQI_DIR="$PWD/external/universe-jieqi"
: "${UNIVERSE_NODE:=node}"
export UNIVERSE_NODE
candidate=build/adaptive/bin/jqe
net=external/devinokeefe-jieqi/engine/nets/it014.jqn
"$REFEREE_PYTHON" scripts/native_match.py --candidate="$candidate" --net="$net" --baseline=external/devinokeefe-jieqi/adapters/musesfish_uci/run.sh --baseline-classical --baseline-strict --out=runs/coverage-musesfish-strict --games=16 --movetime=200 --seed=6300001
"$REFEREE_PYTHON" scripts/native_match.py --candidate="$candidate" --net="$net" --baseline=external/devinokeefe-jieqi/adapters/universe/run.sh --baseline-classical --baseline-options=Difficulty=hard,Seed=20261026 --out=runs/coverage-universe --games=16 --movetime=200 --seed=6400001
"$REFEREE_PYTHON" scripts/native_match.py --candidate="$candidate" --net="$net" --baseline=external/pikafish-jieqi-old/src/PikaJieQi --baseline-classical --out=runs/coverage-pika-old --games=16 --movetime=200 --seed=6500001
"$REFEREE_PYTHON" scripts/native_match.py --candidate="$candidate" --net="$net" --baseline=external/pikajieqi-enhanced/src/PikaJieQi --baseline-classical --out=runs/coverage-pika-enhanced --games=16 --movetime=200 --seed=6600001
"$REFEREE_PYTHON" scripts/native_match.py --candidate="$candidate" --net="$net" --baseline=external/pikafish-jieqi-wasm/src/PikaJieQi --baseline-classical --out=runs/coverage-pika-mistboard --games=16 --movetime=200 --seed=6700001
