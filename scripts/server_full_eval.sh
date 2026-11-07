#!/usr/bin/env bash
# Score one model on all seven full test sets on the server's CPU, at the lowest priority on the given CPUs.
#   tmux new-session -d -s NAME 'bash DIR/repo/scripts/server_full_eval.sh DIR NAME MODEL CPUS [SET...] >> DIR/night/full-NAME.log 2>&1'
# Results: DIR/runs/<set>/<NAME>/. Finished sets are skipped and half-finished ones resume.
# Server and laptop transcripts differ on a few clips (CPU numerics), so compare only runs scored on the same machine.
set -uo pipefail
dir=$(cd "$1" && pwd); name=$2; model=$3; cpus=$4; shift 4
sets=${*:-svarah lahaja mucs librispeech_clean monsoon fleurs_hi silence}
export HF_HOME=$dir/hf XDG_CACHE_HOME=$dir/cache NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
export AKSHAR_DATA=$dir/data AKSHAR_RUNS=$dir/runs JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
cd "$dir/repo"
for s in $sets; do
  out=$dir/runs/$s/$name
  if [ -f "$out/scores.json" ] && grep -q '"complete": true' "$out/scores.json"; then echo "$(date -Is) $s already scored"; continue; fi
  echo "$(date -Is) scoring $name on $s"
  nice -n 19 taskset -c "$cpus" .venv/bin/python -m akshar.eval run --model "$model" --set "$s" --out "$out" \
    | grep -v "x real time" || echo "$(date -Is) FAILED $s"
done
echo "$(date -Is) full evaluation of $name done"
