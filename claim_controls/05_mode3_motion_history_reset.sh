#!/usr/bin/env bash
# Reset velocity history only after a used MODE3 registration; blackout reset stays unchanged.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
RUN=${EXPERIMENT_RUN_ROOT:-"$ROOT/all_runs/run_20260917T094822Z_236992_24438"}
Q1=${FULL_RELEASE:-"$RUN/01_full_release"}
OUT=${CLAIM_OUTPUT:-"$ROOT/claim_control_runs/mode3_reset_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}"}
exec bash "$ROOT/run_test.sh" --release "$Q1" --variants mode3_history_reset --output "$OUT" "$@"
