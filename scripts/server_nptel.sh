#!/usr/bin/env bash
# Build and label AI4Bharat NPTEL for Phase 2 on the server, into DIR/data/train/nptel_en and nptel_hi.
#   tmux new-session -d -s akshar-nptel 'bash DIR/repo/scripts/server_nptel.sh DIR [HOURS] >> DIR/night/nptel.log 2>&1'
# For each part it downloads the first shard, reads its duration column to choose the share of clips that gives about
# HOURS hours (default 30), builds that share with the blocklist check (akshar.data.build), then labels every clip
# with Shruti (akshar.data.label). The two parts are labelled in parallel on CPUs 0-11 and 12-23 at the lowest
# priority. The dataset is gated: the server's Hugging Face account must have been granted access.
set -euo pipefail
dir=$(cd "$1" && pwd); hours=${2:-30}
export HF_HOME=$dir/hf XDG_CACHE_HOME=$dir/cache NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
export AKSHAR_DATA=$dir/data JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
cd "$dir/repo"
share() {  # source name -> fraction of its first shard that gives about $hours hours
  nice -n 19 .venv/bin/python - "$1" "$hours" <<'PY'
import sys
import pyarrow.parquet as pq
from akshar.data import build, sources
src, hours = sources.get(sys.argv[1]), float(sys.argv[2])
path = build._download(src, src.shards[sources.spread(len(src.shards))[0]])
seconds = sum(pq.read_table(path, columns=["duration"]).column("duration").to_pylist())
print(min(1.0, round(hours * 3600 * 1.1 / seconds, 3)))
PY
}
for s in nptel_en nptel_hi; do
  if [ -f "$dir/data/train/$s/manifest.jsonl" ] && grep -q '"done"' "$dir/data/train/$s/progress.json" 2>/dev/null \
     && python3 -c "import json,sys; p=json.load(open('$dir/data/train/$s/progress.json')); sys.exit(0 if p['seconds'] >= $hours*3600*0.9 else 1)"; then
    echo "$(date -Is) $s already built"; continue
  fi
  f=$(share $s)
  echo "$(date -Is) building $s: $hours h at fraction $f"
  nice -n 19 taskset -c 0-23 .venv/bin/python -m akshar.data.build $s --hours "$hours" --fraction "$f"
done
echo "$(date -Is) labelling both parts with Shruti"
nice -n 19 taskset -c 0-11 .venv/bin/python -m akshar.data.label nptel_en > "$dir/night/label-nptel_en.log" 2>&1 &
a=$!
nice -n 19 taskset -c 12-23 .venv/bin/python -m akshar.data.label nptel_hi > "$dir/night/label-nptel_hi.log" 2>&1 &
b=$!
wait $a && wait $b
for s in nptel_en nptel_hi; do
  echo "$s: $(wc -l < $dir/data/train/$s/manifest.jsonl) clips, $(wc -l < $dir/data/train/$s/labels-shruti.jsonl) labelled"
done
echo "$(date -Is) NPTEL READY"
