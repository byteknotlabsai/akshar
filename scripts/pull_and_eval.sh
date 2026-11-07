#!/usr/bin/env bash
# On the laptop: as each Phase 2 run finishes on the server, copy its model back and score it on the dev subsets.
#   scripts/pull_and_eval.sh user@gpu-server /home/user/akshar-work
# Then prints the dev table and scores the two main runs on the full test sets (the Phase 2 exit check).
set -euo pipefail
host=$1 dir=$2
cd "$(dirname "$0")/.."
export NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
PY=.venv/bin/python
[ -f runs/train/a0-mixed2-ptq/model-mixed2.cact ] || \
  $PY -c "from akshar.model import checkpoint, export; export.write(checkpoint.load('models/shruti/shruti.safetensors'), 'runs/train/a0-mixed2-ptq/model-mixed2.cact', export.cactus_mixed)"
nice -n 5 $PY -m akshar.eval dev runs/train/a0-mixed2-ptq/model-mixed2.cact --name a0-mixed2-ptq
RUNS="a1-mixed2-ce a2-mixed2-kd a3-mixed2-kd-l6 a4-mixed2-kd-l4"
for name in $RUNS; do
  until ssh -o BatchMode=yes "$host" "grep -q -E '(DONE|FAILED) $name' $dir/train.log"; do sleep 60; done
  if ssh -o BatchMode=yes "$host" "grep -q 'FAILED $name' $dir/train.log"; then echo "$(date -Is) $name failed"; continue; fi
  mkdir -p "runs/train/$name"
  rsync -a "$host:$dir/runs/train/$name/model-mixed2.cact" "$host:$dir/runs/train/$name/train.json" "runs/train/$name/"
  echo "$(date -Is) scoring $name"
  nice -n 5 $PY -m akshar.eval dev "runs/train/$name/model-mixed2.cact" --name "$name"
done
$PY -m akshar.eval dev-table a0-mixed2-ptq $RUNS | tee runs/train/phase2-dev-table.md
for name in a1-mixed2-ce a2-mixed2-kd; do  # with and without distillation, on the full test sets
  for s in monsoon fleurs_hi lahaja mucs svarah librispeech_clean silence; do
    [ -f "runs/train/$name/model-mixed2.cact" ] || continue
    out="runs/$s/$name"
    [ -f "$out/scores.json" ] || nice -n 5 $PY -m akshar.eval run --model "runs/train/$name/model-mixed2.cact" --set "$s" --out "$out"
  done
done
echo "$(date -Is) phase 2 evaluation done"
