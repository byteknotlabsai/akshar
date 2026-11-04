#!/usr/bin/env bash
# Fetch Phase 2's big inputs on the server itself, into DIR: Shruti's weights and the training audio.
#   bash DIR/repo/scripts/server_fetch.sh DIR [source...]      (default: all five training sets)
# The audio is re-created from the same pinned Hugging Face revisions as on the laptop, exactly the clips each
# copied manifest lists (akshar.data.build --from-manifest). IndicVoices and Kathbath are gated: log in first with
#   HF_HOME=DIR/hf DIR/repo/.venv/bin/hf auth login
# Runs at low priority on cores 12 to 23, the idle half, and never uses the GPU. Resumable: finished shards are skipped.
set -euo pipefail
dir=$(cd "$1" && pwd); shift
sources=${*:-librispeech_100 rural_women_hi fleurs_hi_train indicvoices_hi kathbath_hi}
export HF_HOME=$dir/hf XDG_CACHE_HOME=$dir/cache NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
export AKSHAR_DATA=$dir/data JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
cd "$dir/repo"
run() { nice -n 15 taskset -c 12-23 .venv/bin/python "$@"; }  # the idle half of the cores
if [ ! -f models/shruti/shruti.safetensors ]; then
  run -c "from huggingface_hub import hf_hub_download as d; [d('udgamlabs/Shruti', f, local_dir='models/shruti') for f in ('shruti.cact', 'shruti.safetensors')]"
fi
( cd models/shruti && sha256sum -c <<'SUMS'
b189e954aaed4a9fd41acbad3ac05e486381e96325b22a4c695b9ea5153d7065  shruti.cact
5c43105db153a47d02c47131d37d066343f04f1c3ff82315ff5da1ce264fbfa7  shruti.safetensors
SUMS
)
for s in $sources; do
  echo "$(date -Is) fetching $s"
  if [ "$s" = fleurs_hi_train ]; then
    run -m akshar.data.fleurs --from-manifest "$dir/data/train/$s/manifest.jsonl" --out "$dir/data/train/$s"
  else
    run -m akshar.data.build "$s" --from-manifest "$dir/data/train/$s/manifest.jsonl"
  fi
done
du -sh "$dir/data/train"
echo "$(date -Is) fetched: $sources"
