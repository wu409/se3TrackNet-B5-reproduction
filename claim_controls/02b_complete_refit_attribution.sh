#!/usr/bin/env bash
# Generate genuine observation-only-policy pairs, then fit matched lightweight controls.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
RUN=${EXPERIMENT_RUN_ROOT:-"$ROOT/all_runs/run_20260917T094822Z_236992_24438"}
Q0=${Q0_RELEASE:-"$RUN/04_q0_training/no_rollout_release"}
Q1=${FULL_RELEASE:-"$RUN/01_full_release"}
OUT=${CLAIM_OUTPUT:-"$ROOT/claim_control_runs/refit_complete_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}"}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
EXTRA=()
if [[ -n "${FIXED_POLICY_CACHE:-}" ]]; then EXTRA=(--fixed-policy-cache "$FIXED_POLICY_CACHE"); fi
exec "${TEST_PYTHON:-python}" -B "$ROOT/claim_controls/fixed_policy_control.py" \
    --q0 "$Q0" --q1 "$Q1" --output "$OUT" "${EXTRA[@]}" "$@"
