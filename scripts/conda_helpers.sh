#!/usr/bin/env bash
# Shared functions; source this file from Bash.
PROJ6405_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJ6405_ENV_PREFIX="$PROJ6405_ROOT/.conda-env"

proj6405_find_conda() {
  local candidate=""
  if [[ -n "${PROJ6405_CONDA_EXE:-}" ]]; then
    if [[ ! -x "$PROJ6405_CONDA_EXE" ]]; then
      echo "PROJ6405_CONDA_EXE is not executable: $PROJ6405_CONDA_EXE" >&2
      return 1
    fi
    printf '%s\n' "$PROJ6405_CONDA_EXE"
    return 0
  fi
  if [[ -f "$PROJ6405_ROOT/.conda-executable" ]]; then
    IFS= read -r candidate < "$PROJ6405_ROOT/.conda-executable" || true
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi
  if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    printf '%s\n' "$CONDA_EXE"
    return 0
  fi
  if candidate="$(type -P conda)" && [[ -x "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  for candidate in /workspace/miniforge3/bin/conda /opt/conda/bin/conda /opt/miniconda3/bin/conda \
                   "${HOME}/miniforge3/bin/conda" "${HOME}/miniconda3/bin/conda" "${HOME}/anaconda3/bin/conda"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  echo "Conda not found. See docs/CONDA_RUNPOD_ZH.md, or set PROJ6405_CONDA_EXE to its absolute executable path." >&2
  return 1
}

proj6405_init_conda() {
  local conda_base
  PROJ6405_CONDA_EXE="$(proj6405_find_conda)" || return 1
  conda_base="$("$PROJ6405_CONDA_EXE" info --base)" || return 1
  if [[ ! -f "$conda_base/etc/profile.d/conda.sh" ]]; then
    echo "Conda shell hook not found under: $conda_base" >&2
    return 1
  fi
  source "$conda_base/etc/profile.d/conda.sh" || return 1
  export PROJ6405_CONDA_EXE PROJ6405_ENV_PREFIX
}
