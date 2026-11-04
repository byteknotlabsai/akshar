#!/usr/bin/env bash
# Copy the small things Phase 2's training needs into one directory on the server, over SSH. Starts nothing there.
#   scripts/push_to_server.sh user@gpu-server /home/user/akshar-work
# Only small files travel this way (about 80 MB): the link between this laptop and the server runs through a
# Tailscale relay at about 0.4 MB/s. Big files are fetched by the server itself from Hugging Face, at about
# 14 MB/s (scripts/server_fetch.sh), as D14 asks.
# Layout on the server, all inside that directory:
#   repo/        the code at this commit
#   data/train/  each training set's manifest and Shruti's labels; server_fetch.sh adds the audio
#   bin/uv       the uv binary, so no installer touches shell profiles
#   tokenizer/   Whistle's base tokenizer, copied into the environment by server_setup.sh
set -euo pipefail
host=$1 dir=$2
cd "$(dirname "$0")/.."
ssh "$host" "mkdir -p '$dir/repo' '$dir/data/train' '$dir/bin' '$dir/tokenizer'"
git archive --format=tar HEAD | ssh "$host" "tar -x -C '$dir/repo'"
rsync -a ~/.local/bin/uv "$host:$dir/bin/uv"
rsync -a .venv/lib/python3.12/site-packages/needle/model/tokenizer.model \
         .venv/lib/python3.12/site-packages/needle/model/tokenizer.vocab "$host:$dir/tokenizer/"
for s in librispeech_100 rural_women_hi fleurs_hi_train indicvoices_hi kathbath_hi; do
  rsync -a --include='manifest.jsonl' --include='labels-shruti.json*' --exclude='*' \
        ~/akshar-data/train/$s/ "$host:$dir/data/train/$s/"
done
echo "$(date -Is) copied; next, on the server: scripts/server_setup.sh, then scripts/server_fetch.sh"
