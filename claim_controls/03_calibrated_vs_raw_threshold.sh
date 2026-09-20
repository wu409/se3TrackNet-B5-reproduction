#!/usr/bin/env bash
# Raw cm threshold tuned on development calibration frames; no test-set tuning.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
RUN=${EXPERIMENT_RUN_ROOT:-"$ROOT/all_runs/run_20260917T094822Z_236992_24438"}
Q0=${Q0_RELEASE:-"$RUN/04_q0_training/no_rollout_release"}
Q1=${FULL_RELEASE:-"$RUN/01_full_release"}
OUT=${CLAIM_OUTPUT:-"$ROOT/claim_control_runs/calibration_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}"}
exec "${TEST_PYTHON:-python}" -B "$ROOT/claim_controls/offline_controls.py" calibration --q0 "$Q0" --q1 "$Q1" --output "$OUT"
