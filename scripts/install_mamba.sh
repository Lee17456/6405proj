#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh
export MAX_JOBS="${MAX_JOBS:-2}"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is absent. Matching prebuilt wheels may work; source builds need a CUDA toolkit."
fi
python -m pip install --no-build-isolation -r requirements-mamba.txt
python -m pip check
python -m pip freeze > reports/environment_full.txt
python -m proj6405.doctor
