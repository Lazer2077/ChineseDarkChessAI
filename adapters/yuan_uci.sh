#!/bin/sh
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${JIEQI_PYTHON:-python3}" "$HERE/yuan_uci.py"
