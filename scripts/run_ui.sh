#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh
export API_BASE_URL=http://127.0.0.1:8000
exec python -m streamlit run app.py --server.address "${UI_HOST:-127.0.0.1}" --server.port 8501 --server.headless true >> logs/ui.log 2>&1
