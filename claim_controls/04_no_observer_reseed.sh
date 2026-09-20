#!/usr/bin/env bash
# Same frozen q1 and cache, but a used registration no longer restarts the SE3 observer.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
RUN=${EXPERIMENT_RUN_ROOT:-"$ROOT/all_runs/run_20260917T094822Z_236992_24438"}
Q1=${FULL_RELEASE:-"$RUN/01_full_release"}
OUT=${CLAIM_OUTPUT:-"$ROOT/claim_control_runs/no_reseed_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}"}
exec bash "$ROOT/run_test.sh" --release "$Q1" --variants no_observer_reseed --output "$OUT" "$@"
