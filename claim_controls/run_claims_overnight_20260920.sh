#!/usr/bin/env bash
# Sequential execution of the five uploaded controls; independent failures retained.
set -uo pipefail
ROOT=/root/autodl-tmp/se3TrackNet-B5-reproduction
cd "$ROOT" || exit 2
export TEST_PYTHON=/root/autodl-tmp/conda-envs/b5-main/bin/python3.8
export EXPERIMENT_RUN_ROOT="$ROOT/all_runs/run_20260917T094822Z_236992_24438"
export FULL_RELEASE="$EXPERIMENT_RUN_ROOT/01_full_release"
export Q0_RELEASE="$EXPERIMENT_RUN_ROOT/04_q0_training/no_rollout_release"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN_DIR=${1:?Supply a new output directory}
[[ "$RUN_DIR" == "$ROOT/claim_control_runs/overnight_"* ]] || exit 2
mkdir -p -- "$ROOT/claim_control_runs"
mkdir -- "$RUN_DIR" || exit 2
mkdir -- "$RUN_DIR/logs"
printf '%s\n' "$$" > "$RUN_DIR/runner.pid"
date -Is > "$RUN_DIR/STARTED.txt"
sha256sum claim_controls/*.sh claim_controls/offline_controls.py test_release.py 3-train_evaluation.py > "$RUN_DIR/source.sha256"
printf 'stage\tstarted_utc\tfinished_utc\texit_code\n' > "$RUN_DIR/stages.tsv"
failed=0
stage() {
    local name=$1 script=$2
    shift 2
    local start finish code
    start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf '%s\n' "$name" > "$RUN_DIR/CURRENT_STAGE.txt"
    printf 'Starting %s at %s\n' "$name" "$start"
    CLAIM_OUTPUT="$RUN_DIR/$name" bash "$ROOT/claim_controls/$script" "$@" > "$RUN_DIR/logs/$name.log" 2>&1
    code=$?
    finish=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf '%s\t%s\t%s\t%s\n' "$name" "$start" "$finish" "$code" >> "$RUN_DIR/stages.tsv"
    printf 'Finished %s at %s, exit %s\n' "$name" "$finish" "$code"
    if [[ "$code" != 0 ]]; then failed=1; fi
    return "$code"
}
stage 01_shared_source 01_shared_vs_source_specific.sh
stage 02_refit 02_rollout_refit_attribution.sh
stage 03_calibration 03_calibrated_vs_raw_threshold.sh
if stage 04_preflight 04_no_observer_reseed.sh --check-only; then
    stage 04_no_reseed 04_no_observer_reseed.sh
fi
if stage 05_preflight 05_mode3_motion_history_reset.sh --check-only; then
    stage 05_mode3_reset 05_mode3_motion_history_reset.sh
fi
date -Is > "$RUN_DIR/FINISHED.txt"
if [[ "$failed" == 0 ]]; then
    printf 'All requested scripts completed. Check provenance for missing fixed-policy control.\n' > "$RUN_DIR/COMPLETE.txt"
else
    printf 'One or more stages failed; see stages.tsv and logs.\n' > "$RUN_DIR/FAILED.txt"
fi
exit "$failed"
