#!/usr/bin/env bash
set -euo pipefail
for role in api ui; do
  if tmux has-session -t "proj6405-$role" 2>/dev/null; then
    tmux kill-session -t "proj6405-$role"
  fi
done
