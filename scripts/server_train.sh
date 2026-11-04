#!/usr/bin/env bash
# Phase 2's four training runs on the server's GPU, inside DIR. Start it only after the GPU's usual service is stopped:
#   tmux new-session -d -s akshar-train 'bash DIR/repo/scripts/server_train.sh DIR >> DIR/train.log 2>&1'
# Refuses to start while any other process uses the GPU, takes only the GPU memory it needs, and keeps its
# data loader at low priority on cores 12 to 23. Each run writes DIR/runs/train/<name>/ (model, .cact, train.json);
# its resume file is deleted once the run finishes. The last line says when the GPU is free again.
set -euo pipefail
dir=$(cd "$1" && pwd)
others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$others" -gt 0 ] || [ "$used" -gt 1024 ]; then
  echo "$(date -Is) not starting: $others other GPU processes, $used MiB of GPU memory in use"; exit 1
fi
export HF_HOME=$dir/hf XDG_CACHE_HOME=$dir/cache NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
export AKSHAR_DATA=$dir/data AKSHAR_RUNS=$dir/runs XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$dir/repo"
SOURCES="librispeech_100 rural_women_hi fleurs_hi_train indicvoices_hi kathbath_hi nonspeech"  # nonspeech: D22
# 240 s of padded audio per batch and 2 batches per update: the laptop's 120 x 4, in fewer, larger steps.
COMMON="--sources $SOURCES --recipe mixed2 --steps 1200 --budget 240 --accum 2 --lr 5e-5"
train() {  # name, extra arguments
  local name=$1; shift
  local out=$dir/runs/train/$name
  if [ -f "$out/model-mixed2.cact" ]; then echo "$(date -Is) DONE $name (finished earlier)"; return 0; fi
  echo "$(date -Is) training $name"
  if nice -n 10 taskset -c 12-23 .venv/bin/python -m akshar.train.distill $COMMON "$@" --out "$out" \
       $([ -f "$out/state.msgpack" ] && echo --resume); then
    rm -f "$out/state.msgpack"
    echo "$(date -Is) DONE $name"
  else
    echo "$(date -Is) FAILED $name; going on with the next run"
  fi
}
train a1-mixed2-ce --alpha 0
train a2-mixed2-kd --alpha 0.5
train a3-mixed2-kd-l6 --alpha 0.5 --layers 6
train a4-mixed2-kd-l4 --alpha 0.5 --layers 4
echo "$(date -Is) ALL DONE: the GPU is free"
