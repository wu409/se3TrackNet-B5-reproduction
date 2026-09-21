#!/usr/bin/env bash
# Default: complete fixed-policy collection and matched refit controls.
# Set REFIT_SAME_CANDIDATE_ONLY=1 to reproduce the original partial diagnostic.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ "${REFIT_SAME_CANDIDATE_ONLY:-0}" != 1 ]]; then
    [[ -z "${FIXED_POLICY_SAMPLES:-}" ]] || { echo 'Use FIXED_POLICY_CACHE with its receipt, not a bare unverified CSV.' >&2; exit 2; }
    exec bash "$ROOT/claim_controls/02b_complete_refit_attribution.sh" "$@"
fi
[[ $# == 0 ]] || { echo 'Legacy same-candidate-only mode accepts no extra flags.' >&2; exit 2; }
RUN=${EXPERIMENT_RUN_ROOT:-"$ROOT/all_runs/run_20260917T094822Z_236992_24438"}
Q0=${Q0_RELEASE:-"$RUN/04_q0_training/no_rollout_release"}
Q1=${FULL_RELEASE:-"$RUN/01_full_release"}
OUT=${CLAIM_OUTPUT:-"$ROOT/claim_control_runs/refit_partial_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}"}
exec "${TEST_PYTHON:-python}" -B "$ROOT/claim_controls/offline_controls.py" refit --q0 "$Q0" --q1 "$Q1" --output "$OUT"
