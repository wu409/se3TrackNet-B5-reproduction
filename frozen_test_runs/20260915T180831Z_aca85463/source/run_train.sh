#!/usr/bin/env bash
# User entry point: build the nine-sequence reference, fit three bases, then freeze.
# Usage: bash run_train.sh [--manifest-only | --check-only] [runner options]
# run_train.py implements this workflow; users do not need to launch it separately.
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export CONDA_SH=${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}
export CONDA_ENV=${CONDA_ENV:-b5-main}
if [[ -z "${TRAIN_PYTHON:-}" ]]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    TRAIN_PYTHON=$(command -v python)
fi
export TRAIN_PYTHON

# The runner creates a NEW reference_manifest.csv covering nine sequences.
# It forwards that same CSV to final_fit, which selects only the three bases.
# No train_manifest.csv/test_manifest.csv is created. Errors propagate unchanged.
exec "$TRAIN_PYTHON" -u -B "$SCRIPT_DIR/run_train.py" "$@"
