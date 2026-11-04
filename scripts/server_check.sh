#!/usr/bin/env bash
# Check, on the CPU with the GPU hidden, that everything Phase 2's training reads is in DIR and loads.
#   bash DIR/repo/scripts/server_check.sh DIR
set -euo pipefail
dir=$(cd "$1" && pwd)
export HF_HOME=$dir/hf XDG_CACHE_HOME=$dir/cache NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
export AKSHAR_DATA=$dir/data JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
cd "$dir/repo"
nice -n 15 .venv/bin/python - <<'PY'
from akshar.model import checkpoint
from akshar.train.distill import load_clips
ck = checkpoint.load("models/shruti/shruti.safetensors")
sources = ["librispeech_100", "rural_women_hi", "fleurs_hi_train", "indicvoices_hi", "kathbath_hi"]
train, val, stats = load_clips(sources, "shruti", ck.tokenizer(), ck.cfg, 0.01)
print(f"{len(train)} training clips, {sum(c.seconds for c in train) / 3600:.1f} h; {len(val)} validation; {stats}")
missing = [c.path for c in train + val if not c.path.exists()]
print("every clip's audio is present" if not missing else f"{len(missing)} clips MISSING audio, e.g. {missing[:3]}")
PY
