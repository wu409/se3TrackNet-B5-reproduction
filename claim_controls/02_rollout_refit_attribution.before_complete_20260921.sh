#!/usr/bin/env bash
# q0/q1 same-candidate diagnostics. Optional fixed-policy cache enables prior-only control.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
RUN=${EXPERIMENT_RUN_ROOT:-"$ROOT/all_runs/run_20260917T094822Z_236992_24438"}
Q0=${Q0_RELEASE:-"$RUN/04_q0_training/no_rollout_release"}
Q1=${FULL_RELEASE:-"$RUN/01_full_release"}
OUT=${CLAIM_OUTPUT:-"$ROOT/claim_control_runs/refit_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}"}
EXTRA=()
if [[ -n "${FIXED_POLICY_SAMPLES:-}" ]]; then EXTRA=(--fixed-policy-samples "$FIXED_POLICY_SAMPLES"); fi
exec "${TEST_PYTHON:-python}" -B "$ROOT/claim_controls/offline_controls.py" refit --q0 "$Q0" --q1 "$Q1" --output "$OUT" "${EXTRA[@]}"
