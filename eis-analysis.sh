#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    echo "[eis-analysis] Python 3.10 or newer is required." >&2
    exit 1
  fi
  echo "[eis-analysis] First run: creating local Python environment..." >&2
  python3 -m venv .venv
fi

if [ ! -f ".venv/.eis_ecm_drt_installed" ]; then
  echo "[eis-analysis] First run: installing dependencies..." >&2
  .venv/bin/python -m pip install -r requirements.txt >/dev/null
  echo installed > .venv/.eis_ecm_drt_installed
fi

exec .venv/bin/python -m eis_ecm_drt.cli "$@"
