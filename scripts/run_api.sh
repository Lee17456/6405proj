#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
exec python -m uvicorn proj6405.api:app --host 127.0.0.1 --port 8000 --workers 1 >> logs/api.log 2>&1
