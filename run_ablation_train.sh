#!/usr/bin/env bash
# Four-mode v2 only: derive q0 with identical observer restart/assets, zero refits.
# Train and freeze q0 from an exact completed full release; never change that release.
# bash run_ablation_train.sh --release /absolute/new/full/release [--check-only]
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
[[ "$OMP_NUM_THREADS" =~ ^[1-9][0-9]*$ ]] || { echo 'OMP_NUM_THREADS must be positive' >&2; exit 2; }
if [[ -z "${TRAIN_PYTHON:-}" ]]; then
    source "${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV:-b5-main}"
    TRAIN_PYTHON=$(command -v python)
fi
export TRAIN_PYTHON
exec "$TRAIN_PYTHON" -u -B "$SCRIPT_DIR/ablation_release.py" train-no-rollout "$@"
