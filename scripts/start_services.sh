#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh
for role in api ui; do
  if tmux has-session -t "proj6405-$role" 2>/dev/null; then
    echo "Already running: proj6405-$role"
  else
    tmux new-session -d -s "proj6405-$role" -c "$PWD" \
      -e "UI_HOST=${UI_HOST:-127.0.0.1}" -e "PROJ6405_CONDA_EXE=$PROJ6405_CONDA_EXE" \
      -e "RUN_PREFIX=${RUN_PREFIX:-}" \
      bash "scripts/run_$role.sh"
    echo "Started: proj6405-$role; inspect logs/$role.log"
  fi
done
echo "Check readiness: curl http://127.0.0.1:8000/health"
