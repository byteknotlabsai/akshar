#!/usr/bin/env bash
# Start the akshar speech-to-text server with the settings in settings.env. The first start installs the Python
# packages into .venv, which needs internet.
#   ./run.sh            start the server
#   ./run.sh --setup    install the packages and stop
set -euo pipefail
cd "$(dirname "$0")"

# settings.env, except where the environment already sets a value
if [ -f settings.env ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%$'\r'}
    case "$line" in '' | '#'*) continue ;; esac
    key=${line%%=*}
    [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
    [ -n "${!key+set}" ] || export "$key=${line#*=}"
  done < settings.env
fi

export NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1  # the engine package reports usage unless these are set
# The engine build akshar's published scores were measured with, where it suits this machine. Elsewhere the engine
# package downloads its own build from Hugging Face on the first start.
if [ "$(uname -s)-$(uname -m)" = "Linux-x86_64" ] && [ -f engine/linux-x86_64/libneedle.so ]; then
  export NEEDLE3_LIB_PATH="$PWD/engine/linux-x86_64/libneedle.so"
fi

want=$(cksum requirements.txt | cut -d' ' -f1)
if [ ! -x .venv/bin/python ] || [ "$(cat .venv/.installed 2>/dev/null)" != "$want" ]; then
  echo "Installing the Python packages into .venv (once) ..."
  if command -v uv >/dev/null 2>&1; then
    [ -x .venv/bin/python ] || uv venv --quiet --python 3.12 .venv || uv venv --quiet .venv
    uv pip install --quiet --python .venv/bin/python -r requirements.txt
  else
    uv_hint="Or install uv, which needs no sudo and fetches Python itself: curl -LsSf https://astral.sh/uv/install.sh | sh"
    if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
      echo "akshar needs Python 3.9 or newer as python3. $uv_hint" >&2
      exit 1
    fi
    if [ ! -x .venv/bin/python ] && ! python3 -m venv .venv; then
      echo "python3 cannot make a virtual environment. On Debian or Ubuntu: sudo apt install python3-venv. $uv_hint" >&2
      rm -rf .venv
      exit 1
    fi
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install --quiet -r requirements.txt
  fi
  echo "$want" > .venv/.installed
fi
if [ "${1:-}" = "--setup" ]; then
  echo "Ready. Start the server with ./run.sh"
  exit 0
fi

command=(.venv/bin/python -m akshar.serve --host "${AKSHAR_HOST:-0.0.0.0}" --port "${AKSHAR_PORT:-8765}"
         --models "$PWD/models" "$@")
if [ -n "${AKSHAR_CPUS:-}" ]; then
  command=(taskset -c "$AKSHAR_CPUS" "${command[@]}")
fi
exec "${command[@]}"
