#!/usr/bin/env bash
# Use: source scripts/activate.sh (also works in non-interactive tmux shells).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Use 'source scripts/activate.sh' so activation affects the current shell." >&2
  exit 1
fi
source "$(dirname "${BASH_SOURCE[0]}")/conda_helpers.sh" || return 1
if [[ ! -f "$PROJ6405_ENV_PREFIX/conda-meta/history" || ! -x "$PROJ6405_ENV_PREFIX/bin/python" ]]; then
  echo "Project Conda environment missing. Run: bash scripts/setup_conda.sh" >&2
  return 1
fi
proj6405_init_conda || return 1
conda activate "$PROJ6405_ENV_PREFIX" || return 1
python -c 'import pathlib,sys; assert sys.version_info[:2] == (3,11), "Project requires Python 3.11"; assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(sys.argv[1]).resolve(), "Wrong Python environment"' "$PROJ6405_ENV_PREFIX" || return 1
source "$PROJ6405_ROOT/scripts/env.sh"
