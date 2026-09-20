#!/usr/bin/env bash
# Four-mode v2: checks -> full training -> full/simple testing -> q0 -> ablations.
# All new stages use cached perception v2; do not overlay a running release.
# Only orchestrates existing entry points. Never patches code or reuses old runs.
# Usage:
#   export SE3_PYTHON=/absolute/path/to/compatible/se3/environment/bin/python
#   bash run_all.sh --dry-run
#   bash run_all.sh
# Optional: TRAIN_PYTHON, CONDA_SH, CONDA_ENV, SE3_WEIGHT_ROOT, SE3_DATA_ROOT,
#           B5_NUM_THREADS (16), B5_IO_WORKERS (8), SAM2_CACHE_ROOT,
#           --run-dir /absolute/NEW/output/directory.
# If SE3_PYTHON is unset, use the selected training Python (must support Tracker).
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd "$SCRIPT_DIR"
DRY_RUN=0
SKIP_STANDALONE_CHECKS=0
RUN_DIR=""
CURRENT_STAGE="initialization"

usage() {
    printf '%s\n' \
        'Usage: bash run_all.sh [--dry-run] [--skip-standalone-checks] [--run-dir NEW_DIRECTORY]' \
        'Runs all six steps sequentially; full/simple is one testing step.' \
        'Default output: <project>/all_runs/run_<UTC>_<PID>_<random>/' \
        '--dry-run prints paths/commands only: no environment activation or file creation.' \
        '--skip-standalone-checks omits duplicate check-only runs; real runs retain all built-in checks.' \
        'Use SE3_PYTHON for the compatible SE3 environment; otherwise TRAIN_PYTHON is used.' \
        'A failed run is preserved, never overwritten or automatically resumed.'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --skip-standalone-checks) SKIP_STANDALONE_CHECKS=1; shift ;;
        --run-dir)
            [[ $# -ge 2 && -n "$2" ]] || { echo '--run-dir needs a NEW directory' >&2; exit 2; }
            RUN_DIR=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

RUN_DIR=${RUN_DIR:-"$SCRIPT_DIR/all_runs/run_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}"}
[[ "$RUN_DIR" == /* ]] || RUN_DIR="$SCRIPT_DIR/$RUN_DIR"
[[ ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]] || { echo "Output already exists; choose a NEW directory: $RUN_DIR" >&2; exit 2; }

export B5_NUM_THREADS=${B5_NUM_THREADS:-16}
export B5_IO_WORKERS=${B5_IO_WORKERS:-8}
export SAM2_CACHE_ROOT=${SAM2_CACHE_ROOT:-"$SCRIPT_DIR/all_run/sam2_masks"}
export OMP_NUM_THREADS=$B5_NUM_THREADS
[[ "$OMP_NUM_THREADS" =~ ^[1-9][0-9]*$ ]] || { echo 'OMP_NUM_THREADS must be positive' >&2; exit 2; }
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export SE3_WEIGHT_ROOT=${SE3_WEIGHT_ROOT:-"$SCRIPT_DIR/YCBInEOAT_weights"}
export SE3_DATA_ROOT=${SE3_DATA_ROOT:-"$SCRIPT_DIR/datasets/YCBInEOAT_data"}
for entry in run_train.sh run_test.sh run_ablation_train.sh run_ablations.sh; do
    [[ -f "$SCRIPT_DIR/$entry" ]] || { echo "Missing entry point: $entry" >&2; exit 2; }
done
for entry in perception_runtime.py perception_workers.py runtime_settings.py ordered_prefetch.py sam2_episode_cache.py prepare_sam_cache.py; do
    [[ -f "$SCRIPT_DIR/$entry" ]] || { echo "Missing persistent perception implementation: $entry" >&2; exit 2; }
done

failed() {
    local code=$1 line=$2
    trap - ERR
    printf '\nFAILED: %s (exit %s, line %s)\nOutput preserved: %s\n' \
        "$CURRENT_STAGE" "$code" "$line" "$RUN_DIR" >&2
    printf 'No pipeline success is claimed. Inspect logs and paths.env; no later stages were launched.\n' >&2
    exit "$code"
}
trap 'failed "$?" "$LINENO"' ERR

if [[ "$DRY_RUN" == 0 ]]; then
    # Resolve once, so training, frozen evaluation and q0 use the same Python.
    if [[ -z "${TRAIN_PYTHON:-}" ]]; then
        source "${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
        conda activate "${CONDA_ENV:-b5-main}"
        TRAIN_PYTHON=$(command -v python)
    fi
    TRAIN_PYTHON=$("$TRAIN_PYTHON" -c 'import os,sys; print(os.path.realpath(sys.executable))')
    if [[ -n "${TEST_PYTHON:-}" ]]; then
        TEST_PYTHON=$("$TEST_PYTHON" -c 'import os,sys; print(os.path.realpath(sys.executable))')
        [[ "$TEST_PYTHON" == "$TRAIN_PYTHON" ]] || {
            echo 'TEST_PYTHON must match TRAIN_PYTHON for these frozen runs.' >&2; exit 2;
        }
    fi
    export TRAIN_PYTHON
    export TEST_PYTHON="$TRAIN_PYTHON"
    SE3_PYTHON=${SE3_PYTHON:-"$TRAIN_PYTHON"}
    SE3_PYTHON=$("$SE3_PYTHON" -c 'import os,sys; print(os.path.realpath(sys.executable))')
    export SE3_PYTHON
    [[ -d "$SE3_WEIGHT_ROOT" && -d "$SE3_DATA_ROOT" ]] || {
        echo 'SE3_WEIGHT_ROOT / SE3_DATA_ROOT is missing; set the actual asset directories.' >&2; exit 2;
    }
    SE3_WEIGHT_ROOT=$(cd -- "$SE3_WEIGHT_ROOT" && pwd -P)
    SE3_DATA_ROOT=$(cd -- "$SE3_DATA_ROOT" && pwd -P)
    mkdir -p -- "$(dirname -- "$RUN_DIR")"
    # Atomic reservation. All child output directories remain NEW until used.
    mkdir -- "$RUN_DIR"
    RUN_DIR=$(cd -- "$RUN_DIR" && pwd -P)
    mkdir -- "$RUN_DIR/logs"
    cp -- "$SCRIPT_DIR/run_all.sh" "$RUN_DIR/run_all.sh"
fi

TRAIN_CHECK="$RUN_DIR/00_train_check"
export FULL_RELEASE="$RUN_DIR/01_full_release"
TEST_CHECK="$RUN_DIR/02_test_check"
export FULL_RESULTS="$RUN_DIR/03_full_simple"
Q0_TRAIN_OUTPUT="$RUN_DIR/04_q0_training"
export Q0_RELEASE="$Q0_TRAIN_OUTPUT/no_rollout_release"
ABLATION_RESULTS="$RUN_DIR/05_ablations"

printf 'Pipeline output: %s\nOMP_NUM_THREADS=%s\n' "$RUN_DIR" "$OMP_NUM_THREADS"
printf 'FULL_RELEASE=%s\nFULL_RESULTS=%s\nQ0_RELEASE=%s\nABLATION_RESULTS=%s\n' \
    "$FULL_RELEASE" "$FULL_RESULTS" "$Q0_RELEASE" "$ABLATION_RESULTS"

if [[ "$DRY_RUN" == 0 ]]; then
    # Shell-escaped exact paths; source this file later to reuse explicit outputs.
    for key in TRAIN_PYTHON TEST_PYTHON SE3_PYTHON SE3_WEIGHT_ROOT SE3_DATA_ROOT \
               OMP_NUM_THREADS B5_NUM_THREADS B5_IO_WORKERS SKIP_STANDALONE_CHECKS RUN_DIR TRAIN_CHECK FULL_RELEASE TEST_CHECK \
               FULL_RESULTS Q0_TRAIN_OUTPUT Q0_RELEASE ABLATION_RESULTS SAM2_CACHE_ROOT; do
        printf 'export %s=%q\n' "$key" "${!key}"
    done > "$RUN_DIR/paths.env"
    printf 'stage\tstarted_utc\tfinished_utc\telapsed_seconds\texit_code\n' > "$RUN_DIR/stage_timings.tsv"
fi

run_stage() {
    CURRENT_STAGE=$1
    shift
    printf '\n[%s]\n' "$CURRENT_STAGE"
    printf '  %q' "$@"
    printf '\n'
    if [[ "$DRY_RUN" == 0 ]]; then
        local started_utc started_seconds finished_utc elapsed stage_exit=0
        started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        started_seconds=$SECONDS
        printf 'Started: %s\n' "$started_utc"
        # pipefail propagates child failure even when tee succeeds.
        "$@" 2>&1 | tee "$RUN_DIR/logs/$CURRENT_STAGE.log" || stage_exit=$?
        finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        elapsed=$((SECONDS - started_seconds))
        printf '%s\t%s\t%s\t%s\t%s\n' "$CURRENT_STAGE" "$started_utc" \
            "$finished_utc" "$elapsed" "$stage_exit" >> "$RUN_DIR/stage_timings.tsv"
        printf 'Finished: %s | elapsed: %s seconds | exit: %s\n' "$finished_utc" "$elapsed" "$stage_exit"
        return "$stage_exit"
    fi
}

receipt() {
    [[ "$DRY_RUN" == 0 ]] || return 0
    "$TRAIN_PYTHON" - "$@" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("status") not in sys.argv[2:]:
    raise SystemExit("Unexpected completion status in " + str(path) + ": " + str(data.get("status")))
print("Verified receipt:", path)
PY
}

if [[ "$SKIP_STANDALONE_CHECKS" == 0 ]]; then
    run_stage 01_train_check bash "$SCRIPT_DIR/run_train.sh" --check-only --release-dir "$TRAIN_CHECK"
    if [[ "$DRY_RUN" == 0 ]]; then
        [[ -f "$TRAIN_CHECK/reference_manifest.csv" && -f "$TRAIN_CHECK/effective_config.json" && ! -e "$TRAIN_CHECK/FROZEN.json" ]]
    fi
else
    printf '\nSkipping standalone training check; run_train retains manifest generation and prepare-generated validation.\n'
fi

run_stage 02_train_full bash "$SCRIPT_DIR/run_train.sh" --release-dir "$FULL_RELEASE"
receipt "$FULL_RELEASE/FROZEN.json" final_development_model_frozen

if [[ "$SKIP_STANDALONE_CHECKS" == 0 ]]; then
    run_stage 03_test_check bash "$SCRIPT_DIR/run_test.sh" --release "$FULL_RELEASE" --check-only --output "$TEST_CHECK"
    receipt "$TEST_CHECK/CHECKED.json" preflight_passed_no_inference
else
    printf '\nSkipping standalone testing check; run_test retains release, data and inference preflight validation.\n'
fi

run_stage 04_test_full_simple bash "$SCRIPT_DIR/run_test.sh" --release "$FULL_RELEASE" --variants full simple --output "$FULL_RESULTS"
receipt "$FULL_RESULTS/COMPLETE.json" frozen_test_complete post_test_diagnostic_complete

run_stage 05_train_q0 bash "$SCRIPT_DIR/run_ablation_train.sh" --release "$FULL_RELEASE" --output "$Q0_TRAIN_OUTPUT"
receipt "$Q0_RELEASE/FROZEN.json" final_development_model_frozen

run_stage 06_test_ablations bash "$SCRIPT_DIR/run_ablations.sh" --release "$FULL_RELEASE" \
    --no-rollout-release "$Q0_RELEASE" --full-results "$FULL_RESULTS" --output "$ABLATION_RESULTS"
receipt "$ABLATION_RESULTS/COMPLETE.json" post_test_ablation_complete

if [[ "$DRY_RUN" == 1 ]]; then
    printf '\nDRY RUN ONLY: no directories created, no training/testing executed.\n'
else
    printf 'All requested stages completed successfully at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/PIPELINE_COMPLETE.txt"
    printf '\nAll stages completed. Exact paths: %s\nFinal ablations: %s\n' "$RUN_DIR/paths.env" "$ABLATION_RESULTS"
fi
