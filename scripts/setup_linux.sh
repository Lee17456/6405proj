#!/usr/bin/env bash
set -euo pipefail
# Compatibility entry point: the project now uses Conda by default.
exec bash "$(dirname "$0")/setup_conda.sh" "$@"
