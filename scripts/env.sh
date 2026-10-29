#!/usr/bin/env bash
# Source from bash. Paths are derived from this file, not the caller's working directory.
PROJ6405_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="$PROJ6405_ROOT/.cache/huggingface"
export MPLCONFIGDIR="$PROJ6405_ROOT/.cache/matplotlib"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
mkdir -p "$PROJ6405_ROOT/logs" "$PROJ6405_ROOT/reports" "$MPLCONFIGDIR"
