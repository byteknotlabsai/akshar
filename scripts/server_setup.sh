#!/usr/bin/env bash
# Build Phase 2's training environment inside DIR on the server, after scripts/push_to_server.sh.
#   bash DIR/repo/scripts/server_setup.sh DIR
# Writes only inside DIR (environment, caches) and never uses the GPU: every check runs on the CPU with the GPU
# hidden, so a service already on the GPU is undisturbed. Starts no training.
set -euo pipefail
dir=$(cd "$1" && pwd)
export UV_CACHE_DIR=$dir/uv-cache UV_PYTHON_DOWNLOADS=never HF_HOME=$dir/hf XDG_CACHE_HOME=$dir/cache
export NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 AKSHAR_DATA=$dir/data AKSHAR_RUNS=$dir/runs
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
cd "$dir/repo"
nice -n 10 "$dir/bin/uv" sync --frozen --extra gpu --python /usr/bin/python3.12
cp "$dir/tokenizer/tokenizer.model" "$dir/tokenizer/tokenizer.vocab" .venv/lib/python3.12/site-packages/needle/model/
nice -n 10 .venv/bin/python - <<'PY'
import jax
from akshar.model import checkpoint
from akshar.train.distill import load_clips
print("jax", jax.__version__, "devices", jax.devices())
ck = checkpoint.load("models/shruti/shruti.safetensors")
tokenizer = ck.tokenizer()
sources = ["librispeech_100", "rural_women_hi", "fleurs_hi_train", "indicvoices_hi", "kathbath_hi"]
train, val, stats = load_clips(sources, "shruti", tokenizer, ck.cfg, 0.01)
print(f"{len(train)} training clips, {sum(c.seconds for c in train) / 3600:.1f} h; {len(val)} validation")
missing = [c.path for c in train[::500] if not c.path.exists()]
print("sampled audio files present" if not missing else f"MISSING audio, e.g. {missing[:3]}")
PY
du -sh "$dir"
echo "ready; training starts only when asked"
