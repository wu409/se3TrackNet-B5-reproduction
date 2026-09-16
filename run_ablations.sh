#!/usr/bin/env bash
# Evaluate controls and q0; optionally reuse a completed matched full/simple run.
# bash run_ablations.sh --release /full/release --no-rollout-release /q0/release \
#   --full-results /completed/full-simple/run [--check-only]
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
[[ "$OMP_NUM_THREADS" =~ ^[1-9][0-9]*$ ]] || { echo 'OMP_NUM_THREADS must be positive' >&2; exit 2; }
if [[ -z "${TEST_PYTHON:-}" ]]; then
    source "${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV:-b5-main}"
    TEST_PYTHON=$(command -v python)
fi
export TEST_PYTHON
exec "$TEST_PYTHON" -u -B "$SCRIPT_DIR/ablation_release.py" evaluate "$@"
