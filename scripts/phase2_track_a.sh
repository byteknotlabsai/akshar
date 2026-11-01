#!/usr/bin/env bash
# Phase 2 (Track A): the 2-bit recipe, trained from Shruti and distilled from it. Research-only weights (D9).
# Runs unattended; start it as a user service so it survives session restarts:
#   systemd-run --user --collect --unit=akshar-phase2 --working-directory=$PWD -p MemoryMax=12G \
#     -p StandardOutput=append:runs/logs/phase2.log -p StandardError=append:runs/logs/phase2.log scripts/phase2_track_a.sh
# Every step is skipped if its output exists, so re-running continues where it stopped.
set -euo pipefail
export NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=.venv/bin/python
SOURCES="librispeech_100 rural_women_hi fleurs_hi_train indicvoices_hi kathbath_hi"
COMMON="--sources $SOURCES --recipe mixed2 --steps 1200 --budget 120 --accum 4 --lr 5e-5"

echo "$(date -Is) waiting for teacher labels"
while systemctl --user is-active --quiet akshar-labels akshar-build-indicvoices_hi akshar-build-kathbath_hi; do sleep 60; done
for s in $SOURCES; do  # every clip must be labelled before training starts
  m=$(wc -l < ~/akshar-data/train/$s/manifest.jsonl); l=$(wc -l < ~/akshar-data/train/$s/labels-shruti.jsonl)
  echo "$s: $l of $m clips labelled"; [ "$l" -eq "$m" ] || { echo "labelling incomplete for $s"; exit 1; }
done

dev() {  # decode a model's dev subsets on the CPU, in the background, while the GPU trains the next model
  nice -n 15 $PY -m akshar.eval dev "$1" --name "$2" > "runs/logs/dev-$2.log" 2>&1 &
}

# a0: Shruti's weights quantised to the mixed 2-bit recipe with no training (the cost of 2 bits by itself).
if [ ! -f runs/train/a0-mixed2-ptq/model-mixed2.cact ]; then
  mkdir -p runs/train/a0-mixed2-ptq
  $PY -c "from akshar.model import checkpoint, export; print(export.write(checkpoint.load('models/shruti/shruti.safetensors'), 'runs/train/a0-mixed2-ptq/model-mixed2.cact', export.cactus_mixed))"
fi
dev runs/train/a0-mixed2-ptq/model-mixed2.cact a0-mixed2-ptq

train() {  # name, extra arguments
  local name=$1; shift
  if [ ! -f "runs/train/$name/model-mixed2.cact" ]; then
    echo "$(date -Is) training $name"
    if ! $PY -m akshar.train.distill $COMMON "$@" --out "runs/train/$name" $([ -f "runs/train/$name/state.msgpack" ] && echo --resume); then
      echo "$(date -Is) FAILED $name; going on with the next run"
      return 0
    fi
  fi
  dev "runs/train/$name/model-mixed2.cact" "$name"
}

train a1-mixed2-ce --alpha 0            # quantisation-aware training on Shruti's transcripts, no distillation
train a2-mixed2-kd --alpha 0.5          # the same plus distillation from Shruti's token distributions
train a3-mixed2-kd-l6 --alpha 0.5 --layers 6   # smaller tiers for low-end phones
train a4-mixed2-kd-l4 --alpha 0.5 --layers 4

wait  # the last dev decodes
$PY -m akshar.eval dev-table a0-mixed2-ptq a1-mixed2-ce a2-mixed2-kd a3-mixed2-kd-l6 a4-mixed2-kd-l4 | tee runs/train/phase2-dev-table.md
echo "$(date -Is) phase 2 schedule done"
