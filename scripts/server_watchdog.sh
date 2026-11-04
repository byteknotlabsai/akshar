#!/usr/bin/env bash
# Safety net for a night on the server's GPU: at TIME, if coot-moe is not running, stop our training and start it.
#   tmux new-session -d -s akshar-watchdog 'bash DIR/repo/scripts/server_watchdog.sh "YYYY-MM-DD HH:MM" >> DIR/night/watchdog.log 2>&1'
set -uo pipefail
sleep $(( $(date -d "$1" +%s) - $(date +%s) ))
if [ -z "$(docker ps -q --filter name=^coot-moe$ --filter status=running)" ]; then
  echo "$(date -Is) coot-moe is not running at the deadline; stopping our training and starting it"
  pkill -u "$(id -un)" -f "akshar.train.distill"
  for _ in $(seq 24); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && break; sleep 5; done
  docker start coot-moe && echo "$(date -Is) coot-moe: $(docker ps -a --filter name=^coot-moe$ --format '{{.Status}}')"
else
  echo "$(date -Is) coot-moe is already running; nothing to do"
fi
