#!/usr/bin/env bash
# Label training sources with Shruti one at a time. One engine uses every core; several at once crowd each other out
# (three in parallel ran at about a third of one engine's speed). A source still being built is waited for first.
set -euo pipefail
export NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
for source in "$@"; do
  while systemctl --user is-active --quiet "akshar-build-$source"; do sleep 30; done
  echo "$(date -Is) labelling $source"
  .venv/bin/python -m akshar.data.label "$source"
done
echo "$(date -Is) labelling done"
