#!/usr/bin/env bash
# Back up the akshar working folder to Google Drive: an exact mirror of every file except datasets and caches.
#
#   scripts/backup_to_drive.sh
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
mirror "$HOME/Projects/akshar" Projects-akshar \
  --exclude "/.venv/**" --exclude "__pycache__/**" --exclude ".pytest_cache/**" --exclude ".ruff_cache/**"
mirror "$HOME/akshar-data" akshar-data --exclude "/raw/**" --exclude "/eval/**" --exclude "/train/**"
echo "$(date -Is) backup $STAMP done" >> "$LOG"
