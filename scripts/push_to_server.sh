#!/usr/bin/env bash
# Copy what Phase 2's training needs into one directory on a remote machine, over SSH. Starts nothing there.
#   scripts/push_to_server.sh user@gpu-server /path/on/server
# Layout on the server, all inside that directory:
#   repo/        the code at this commit, plus models/shruti (weights)
#   data/train/  the five training sets: manifest, Shruti's labels and audio
#   bin/uv       the uv binary, so no installer touches shell profiles
#   tokenizer/   Whistle's base tokenizer, copied into the environment by server_setup.sh
set -euo pipefail
host=$1 dir=$2
cd "$(dirname "$0")/.."
ssh "$host" "mkdir -p '$dir/repo/models/shruti' '$dir/data/train' '$dir/bin' '$dir/tokenizer'"
git archive --format=tar HEAD | ssh "$host" "tar -x -C '$dir/repo'"
rsync -a models/shruti/shruti.safetensors models/shruti/shruti.cact "$host:$dir/repo/models/shruti/"
rsync -a ~/.local/bin/uv "$host:$dir/bin/uv"
rsync -a .venv/lib/python3.12/site-packages/needle/model/tokenizer.model \
         .venv/lib/python3.12/site-packages/needle/model/tokenizer.vocab "$host:$dir/tokenizer/"
for s in librispeech_100 rural_women_hi fleurs_hi_train indicvoices_hi kathbath_hi; do
  echo "$(date -Is) copying $s"
  rsync -a --info=stats1 --include='manifest.jsonl' --include='labels-shruti.json*' --include='clips/***' \
        --exclude='*' ~/akshar-data/train/$s/ "$host:$dir/data/train/$s/"
done
echo "$(date -Is) copied; next, on the server: bash $dir/repo/scripts/server_setup.sh $dir"
