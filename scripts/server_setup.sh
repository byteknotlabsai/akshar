#!/usr/bin/env bash
# Build Phase 2's training environment inside DIR on the server, after scripts/push_to_server.sh.
# Then scripts/server_fetch.sh fetches the weights and audio, and scripts/server_check.sh checks it all.
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
nice -n 10 .venv/bin/python -c "import jax, akshar.train.distill; print('jax', jax.__version__, 'devices', jax.devices())"
du -sh "$dir"
echo "environment ready; next: scripts/server_fetch.sh $dir"
