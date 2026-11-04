#!/usr/bin/env bash
# A night of Phase 2 runs on the server's GPU while the owner's llama.cpp container, coot-moe, is stopped.
#   tmux new-session -d -s akshar-night 'bash DIR/repo/scripts/server_night.sh DIR "YYYY-MM-DD HH:MM" "YYYY-MM-DD HH:MM" >> DIR/night/night.log 2>&1'
# The two times are the cutoff (no run starts after it) and the deadline (runs still going are stopped; they resume).
# Two streams, A and B, train in parallel. Each takes runs from DIR/night/queue-A.txt or queue-B.txt, one per line:
# a run name, then akshar.train.distill arguments that override COMMON. Lines can be appended while it runs.
# A stream ends at the cutoff, or when its queue is empty and DIR/night/finish exists.
# When both streams have ended, for whatever reason, it starts coot-moe again; nothing else on the server is touched.
# It refuses to start while anything uses the GPU, and a run starts only while no other program holds the GPU.
set -uo pipefail
dir=$(cd "$1" && pwd)
cutoff_s=$(date -d "$2" +%s)
deadline_s=$(date -d "$3" +%s)
night=$dir/night
mkdir -p "$night"
export HF_HOME=$dir/hf XDG_CACHE_HOME=$dir/cache NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
export AKSHAR_DATA=$dir/data AKSHAR_RUNS=$dir/runs XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$dir/repo"
SOURCES="librispeech_100 rural_women_hi fleurs_hi_train indicvoices_hi kathbath_hi nonspeech"
COMMON="--sources $SOURCES --recipe mixed2 --steps 1200 --budget 240 --accum 2 --lr 5e-5 --alpha 0.5"
me=$(id -un)

log() { echo "$(date -Is) $*"; }

foreign_gpu() {  # how many GPU processes are not our own training
  local pid n=0
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    ps -o args= -p "$pid" 2>/dev/null | grep -q "akshar.train.distill" || n=$((n + 1))
  done
  echo "$n"
}

train() {  # stream, run name, extra arguments
  local s=$1 name=$2; shift 2
  local out=$dir/runs/train/$name
  if ls "$out"/model-*.cact >/dev/null 2>&1; then log "[$s] DONE $name (finished earlier)"; return 0; fi
  while [ "$(foreign_gpu)" -gt 0 ]; do
    if [ "$(date +%s)" -ge "$cutoff_s" ]; then log "[$s] another program holds the GPU past the cutoff; skipping $name"; return 0; fi
    log "[$s] another program holds the GPU; waiting"; sleep 60
  done
  log "[$s] training $name: $*"
  if nice -n 10 taskset -c 12-23 .venv/bin/python -m akshar.train.distill $COMMON "$@" --out "$out" \
       $([ -f "$out/state.msgpack" ] && echo --resume) > "$night/$name.log" 2>&1; then
    rm -f "$out/state.msgpack"
    log "[$s] DONE $name: $(grep -E '^val ' "$night/$name.log" | tail -1)"
  else
    log "[$s] FAILED $name: $(grep -v cuda_timer "$night/$name.log" | tail -1 | cut -c1-200)"
  fi
}

stream() {  # stream name
  local s=$1 q=$night/queue-$1.txt n=0 line
  touch "$q"
  while :; do
    if [ "$(date +%s)" -ge "$cutoff_s" ]; then log "[$s] past the cutoff; stream ends"; return; fi
    line=$(grep -v -E '^[[:space:]]*(#|$)' "$q" | sed -n "$((n + 1))p")
    if [ -n "$line" ]; then
      n=$((n + 1))
      train "$s" $line
    elif [ -f "$night/finish" ]; then
      log "[$s] queue empty and finish requested; stream ends"; return
    else
      sleep 30
    fi
  done
}

finish() {
  trap - EXIT INT TERM HUP
  kill "${timer:-}" "${pa:-}" "${pb:-}" 2>/dev/null
  pkill -u "$me" -f "akshar.train.distill" && log "stopped the runs still going"
  for _ in $(seq 24); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && break; sleep 5; done
  log "starting coot-moe"
  docker start coot-moe >/dev/null && log "coot-moe: $(docker ps -a --filter name=coot-moe --format '{{.Status}}')"
  log "NIGHT DONE: the GPU is back with coot-moe"
}

log "night starts; cutoff $(date -d "@$cutoff_s" '+%F %R'), deadline $(date -d "@$deadline_s" '+%F %R')"
if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; then
  log "not starting: something is using the GPU"; exit 1
fi
trap finish EXIT
trap 'exit 1' INT TERM HUP
( sleep $((deadline_s - $(date +%s))); log "deadline reached"; touch "$night/finish"
  pkill -u "$me" -f "akshar.train.distill" ) &
timer=$!
stream A & pa=$!
stream B & pb=$!
wait "$pa" "$pb"
log "both streams ended"
