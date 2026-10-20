#!/usr/bin/env bash
# bash launcher of src/runner.py (Linux / macOS / Docker)
#   ./runner.sh ci --electron 0.3 0.5 --s --test
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${LDMX_PYTHON:-python3}" -X utf8 "$here/src/runner.py" "$@"
