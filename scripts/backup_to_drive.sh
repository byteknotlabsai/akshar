#!/usr/bin/env bash
# Back up the akshar working folder to Google Drive: an exact mirror of every file except datasets and caches.
#
#   scripts/backup_to_drive.sh
#
# First, if ~/.config/akshar/backup.env names a GPU server (AKSHAR_SERVER=user@host, AKSHAR_SERVER_DIR=its work
# folder) and it answers, its work files are pulled into ~/akshar-data/server: run outputs, labels, test results,
# logs and scripts. Datasets, caches, login tokens (hf/, rclone/), the code checkout and resume states stay behind.
# The server is rented, so nothing made there may exist only there. The file is kept out of the repository because
# it names a private machine.
#
# Mirror:  gdrive:akshar/laptop-backup/current/Projects-akshar  this repository, with .git, models/, runs/ and
#                                                              uncommitted changes
#          gdrive:akshar/laptop-backup/current/akshar-data      ~/akshar-data without its datasets
# Archive: a file that a run changes or deletes on Drive is first moved to
#          gdrive:akshar/laptop-backup/archive/<date_time>/..., so a sync never loses anything.
# Left out: .venv (rebuilt by `uv sync`), Python caches, and the datasets in ~/akshar-data (raw/, eval/, train/),
#          which come from Hugging Face and gdrive:akshar/p3.
# Daily:   a systemd user timer runs it (~/.config/systemd/user/akshar-backup.timer, 21:00, catching up after the
#          laptop was off). Log: ~/.local/state/akshar-backup.log.
set -euo pipefail
RCLONE=${RCLONE:-$HOME/.local/bin/rclone}
DEST=gdrive:akshar/laptop-backup
STAMP=$(date +%Y-%m-%d_%H%M)
LOG=$HOME/.local/state/akshar-backup.log
mkdir -p "$(dirname "$LOG")" "$HOME/.cache"
exec 9>"$HOME/.cache/akshar-backup.lock"
flock -n 9 || { echo "$(date -Is) another backup is still running" >> "$LOG"; exit 0; }

mirror() {  # source folder, name on Drive, extra rclone filters
  local src=$1 name=$2; shift 2
  "$RCLONE" sync "$src" "$DEST/current/$name" --backup-dir "$DEST/archive/$STAMP/$name" \
    --links --fast-list --transfers 4 --log-file "$LOG" --log-level NOTICE "$@"
}

echo "$(date -Is) backup $STAMP starts" >> "$LOG"
CONFIG=$HOME/.config/akshar/backup.env
[ -f "$CONFIG" ] && . "$CONFIG"
if [ -n "${AKSHAR_SERVER:-}" ]; then
  if ssh -o BatchMode=yes -o ConnectTimeout=15 "$AKSHAR_SERVER" true 2>/dev/null; then
    if rsync -a --partial --timeout=300 \
         --exclude "/repo/" --exclude "/hf/" --exclude "/rclone/" --exclude "/bin/" --exclude "/uv-cache/" \
         --exclude "/cache/" --exclude "/data/eval/" --exclude "/data/raw/" --exclude "/data3/eval" --exclude "/data3/raw/" \
         --exclude "/data3/train/*/manifest.jsonl" --exclude "/data3/train/*/blocked.jsonl" \
         --exclude "clips/" --exclude "staging/" --exclude "*.flac" --exclude "*.tar" --exclude "*.parquet" \
         --exclude "*.msgpack" "$AKSHAR_SERVER:$AKSHAR_SERVER_DIR/" "$HOME/akshar-data/server/" >> "$LOG" 2>&1; then
      echo "$(date -Is) pulled the server's work files" >> "$LOG"
    else
      echo "$(date -Is) pulling the server's work files FAILED; backing up the rest" >> "$LOG"
    fi
  else
    echo "$(date -Is) the server did not answer; its files were not pulled" >> "$LOG"
  fi
fi
mirror "$HOME/Projects/akshar" Projects-akshar \
  --exclude "/.venv/**" --exclude "__pycache__/**" --exclude ".pytest_cache/**" --exclude ".ruff_cache/**"
mirror "$HOME/akshar-data" akshar-data --exclude "/raw/**" --exclude "/eval/**" --exclude "/train/**"
echo "$(date -Is) backup $STAMP done" >> "$LOG"
