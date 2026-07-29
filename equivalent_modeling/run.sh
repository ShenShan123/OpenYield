#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

nohup "$PYTHON_BIN" -u "$PROJECT_ROOT/equivalent_modeling/main_sram.py" \
    >/dev/null 2>&1 &
