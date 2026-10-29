#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "Run this installer on the Linux x86_64 GPU server." >&2
  exit 1
fi
source scripts/conda_helpers.sh
source scripts/env.sh
PROJ6405_CONDA_EXE="$(proj6405_find_conda)"
export PROJ6405_CONDA_EXE
# Keep the package cache on the same persistent filesystem as the environment.
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$PROJ6405_ROOT/.cache/conda/pkgs}"
if [[ -f "$PROJ6405_ENV_PREFIX/conda-meta/history" ]]; then
  echo "Reusing project Conda environment: $PROJ6405_ENV_PREFIX"
elif [[ -e "$PROJ6405_ENV_PREFIX" ]]; then
  echo "Existing path is not a completed Conda environment: $PROJ6405_ENV_PREFIX. Inspect it before retrying." >&2
  exit 1
else
  "$PROJ6405_CONDA_EXE" create --yes --prefix "$PROJ6405_ENV_PREFIX" \
    --override-channels --channel conda-forge python=3.11 pip
fi
printf '%s\n' "$PROJ6405_CONDA_EXE" > .conda-executable
source scripts/activate.sh
python -m pip install --upgrade 'pip==25.0.1' 'setuptools==75.8.0' 'wheel==0.45.1' 'packaging==24.2' 'ninja==1.11.1.3'
python -m pip install 'torch==2.5.1+cu121' --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
python -m pip freeze > reports/environment_base.txt
"$PROJ6405_CONDA_EXE" list --prefix "$PROJ6405_ENV_PREFIX" --explicit > reports/conda-explicit-linux64.txt
python -m proj6405.doctor
echo "Conda environment ready. Next: bash scripts/install_mamba.sh"
echo "In every new terminal: source scripts/activate.sh"
