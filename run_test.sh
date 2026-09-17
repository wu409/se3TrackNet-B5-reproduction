#!/usr/bin/env bash
# Frozen test entry point. Never trains or regenerates the reference manifest.
# Requires matching persistent-perception source/config in the new training release.
# bash run_test.sh --release /absolute/path/to/frozen/release [--check-only]
# Requires a NEW four-mode v2 release; old policy transplants are rejected.
# Default: full, then simple with full's MODE3 trigger schedule.
# Standalone simple: --reference-full-results /completed/matching/v2/run
set -Eeuo pipefail
SCRIPT_DIR=${BASH_SOURCE[0]%/*}
[[ "$SCRIPT_DIR" != "${BASH_SOURCE[0]}" ]] || SCRIPT_DIR=.
SCRIPT_DIR=$(cd -- "$SCRIPT_DIR" && pwd)
cd "$SCRIPT_DIR"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export B5_NUM_THREADS=${B5_NUM_THREADS:-${OMP_NUM_THREADS:-4}}
export OMP_NUM_THREADS=$B5_NUM_THREADS
[[ "$OMP_NUM_THREADS" =~ ^[1-9][0-9]*$ ]] || { echo 'OMP_NUM_THREADS must be a positive integer' >&2; exit 2; }
if [[ -z "${TEST_PYTHON:-}" ]]; then
    source "${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV:-b5-main}"
    TEST_PYTHON=$(command -v python)
fi
export TEST_PYTHON
exec "$TEST_PYTHON" -u -B "$SCRIPT_DIR/test_release.py" "$@"
