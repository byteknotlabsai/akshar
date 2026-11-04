#!/usr/bin/env bash
# Score each finished run on the dev subsets on the server's CPU, while the GPU trains the next ones.
#   tmux new-session -d -s akshar-dev 'bash DIR/repo/scripts/server_dev_loop.sh DIR NAME_REGEX >> DIR/night/dev.log 2>&1'
# Needs the dev subsets' clips in DIR/data/eval (copied from the laptop; full manifests, first clips' audio only).
# Takes runs whose name matches NAME_REGEX, oldest model first, at the lowest CPU priority on cores 0 to 11 (training's loaders use 12 to 23).
# Ends once DIR/night/eval-stop exists and every matching model is scored. Results: DIR/runs/dev/<run>/<set>/.
set -uo pipefail
dir=$(cd "$1" && pwd); pattern=$2
export HF_HOME=$dir/hf XDG_CACHE_HOME=$dir/cache NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
export AKSHAR_DATA=$dir/data AKSHAR_RUNS=$dir/runs JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
cd "$dir/repo"
scored() {  # every dev set of this run has complete scores
  [ "$(grep -l '"complete": true' "$dir/runs/dev/$1"/*/scores.json 2>/dev/null | wc -l)" -ge 7 ]
}
while :; do
  next=""
  for model in $(ls -tr "$dir"/runs/train/*/model-*.cact 2>/dev/null); do
    name=$(basename "$(dirname "$model")")
    [[ $name =~ $pattern ]] || continue
    scored "$name" && continue
    next=$model; break
  done
  if [ -n "$next" ]; then
    name=$(basename "$(dirname "$next")")
    echo "$(date -Is) scoring $name"
    nice -n 19 taskset -c 0-11 .venv/bin/python -m akshar.eval dev "$next" --name "$name" > "$dir/night/dev-$name.log" 2>&1 \
      && echo "$(date -Is) scored $name" || echo "$(date -Is) FAILED scoring $name"
  elif [ -f "$dir/night/eval-stop" ]; then
    echo "$(date -Is) everything scored; eval loop ends"; exit 0
  else
    sleep 60
  fi
done
