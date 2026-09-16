#!/usr/bin/env bash
# Four-mode v2: nine-sequence reference, four development sequences, then freeze.
# SE3_PYTHON must support predict.Tracker; SE3_WEIGHT_ROOT/SE3_DATA_ROOT locate assets.
# Usage: bash run_train.sh [--manifest-only | --check-only] [runner options]
# run_train.py implements this workflow; users do not need to launch it separately.
set -Eeuo pipefail
# New releases snapshot b5_revision.py and recovery_gate.py automatically.
# Their development settings are recorded in model config and FROZEN.json.
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
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
[[ "$OMP_NUM_THREADS" =~ ^[1-9][0-9]*$ ]] || { echo 'OMP_NUM_THREADS must be positive' >&2; exit 2; }

# The runner creates a NEW reference_manifest.csv covering nine sequences.
# It forwards that same CSV to final_fit, which selects only the four bases.
# No train_manifest.csv/test_manifest.csv is created. Errors propagate unchanged.
exec "$TRAIN_PYTHON" -u -B "$SCRIPT_DIR/run_train.py" "$@"
