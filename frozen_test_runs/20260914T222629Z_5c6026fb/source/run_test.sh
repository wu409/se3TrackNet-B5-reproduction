#!/usr/bin/env bash
# Frozen test entry point. Never trains or regenerates the reference manifest.
# bash run_test.sh --release /absolute/path/to/frozen/release [--check-only]
set -Eeuo pipefail
SCRIPT_DIR=${BASH_SOURCE[0]%/*}
[[ "$SCRIPT_DIR" != "${BASH_SOURCE[0]}" ]] || SCRIPT_DIR=.
SCRIPT_DIR=$(cd -- "$SCRIPT_DIR" && pwd)
cd "$SCRIPT_DIR"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
[[ "$OMP_NUM_THREADS" =~ ^[1-9][0-9]*$ ]] || { echo 'OMP_NUM_THREADS must be a positive integer' >&2; exit 2; }
if [[ -z "${TEST_PYTHON:-}" ]]; then
    source "${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV:-b5-main}"
    TEST_PYTHON=$(command -v python)
fi
export TEST_PYTHON
exec "$TEST_PYTHON" -u -B "$SCRIPT_DIR/test_release.py" "$@"
