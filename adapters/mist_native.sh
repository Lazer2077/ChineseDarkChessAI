#!/usr/bin/env bash
set -euo pipefail
: "${MIST_NODE:=node}"
exec "$MIST_NODE" "$(dirname "$0")/mist_native.mjs"
