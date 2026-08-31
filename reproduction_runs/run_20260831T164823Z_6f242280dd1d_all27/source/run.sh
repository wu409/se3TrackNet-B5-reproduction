#!/bin/bash
set -Eeuo pipefail

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# ============================================================
# 0. Global experiment configuration
# ============================================================
SEED=${SEED:-42}
RISK_THRESHOLD=${RISK_THRESHOLD:-1.0}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-10000}

DATASET_ROOT=${DATASET_ROOT:-./datasets/YCBInEOAT_Corrupted}
GT_ROOT=${GT_ROOT:-./datasets/YCBInEOAT}
RESULT_ROOT=${RESULT_ROOT:-./results_collection}
CAD_MODEL_ROOT=${CAD_MODEL_ROOT:-./datasets/YCB_Video_Models/CADmodels}
MANIFEST_CONFIG=${MANIFEST_CONFIG:-"$SCRIPT_DIR/manifest_config.json"}


# This manifest is dedicated to the complete 3 x 9 experiment matrix.
REFERENCE_MANIFEST=${REFERENCE_MANIFEST:-"$SCRIPT_DIR/reference_manifest_all27.csv"}
REBUILD_REFERENCE_MANIFEST=${REBUILD_REFERENCE_MANIFEST:-0}

SE3TRACKNET_WEIGHTS_ROOT=${SE3TRACKNET_WEIGHTS_ROOT:-"$SCRIPT_DIR/YCBInEOAT_weights"}

CONDA_SH=${CONDA_SH:-/home/wyg/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-yolov5}

# FoundationPose remains in a separate conda environment. b5_policy.py launches
# it through this absolute Python executable.
FOUNDATIONPOSE_PYTHON=${FOUNDATIONPOSE_PYTHON:-/home/wyg/anaconda3/envs/foundationpose/bin/python}
FOUNDATIONPOSE_DIR=${FOUNDATIONPOSE_DIR:-/home/wyg/FoundationPose}
FOUNDATIONPOSE_REFINER_WEIGHT=${FOUNDATIONPOSE_REFINER_WEIGHT:-"$FOUNDATIONPOSE_DIR/weights/2023-10-28-18-33-37/model_best.pth"}
FOUNDATIONPOSE_SCORER_WEIGHT=${FOUNDATIONPOSE_SCORER_WEIGHT:-"$FOUNDATIONPOSE_DIR/weights/2024-01-11-20-02-45/model_best.pth"}
FOUNDATIONPOSE_REFINE_ITER=${FOUNDATIONPOSE_REFINE_ITER:-5}

# Optional explicit overrides. If empty, run.sh resolves the mesh using the same
# preference as 2-risk_label.py: textured_simple.obj -> textured.obj -> textured.ply.
FOUNDATIONPOSE_MUSTARD_MESH=${FOUNDATIONPOSE_MUSTARD_MESH:-}
FOUNDATIONPOSE_BLEACH_MESH=${FOUNDATIONPOSE_BLEACH_MESH:-}

# Low-VRAM settings used by the modified FoundationPose predictor code.
export FP_REFINE_CHUNK_SIZE=${FP_REFINE_CHUNK_SIZE:-2}
export FP_SCORE_CHUNK_SIZE=${FP_SCORE_CHUNK_SIZE:-2}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Three base sequences. Keep label-generation order equal to the existing
# 2-risk_label.py / old run.sh protocol so that only ci_object changes between
# the three label-generation passes.
BASE_SEQUENCES=(
    mustard0
    bleach0
    bleach_hard_00_03_chaitanya
)
LABEL_TARGET_SEQS=(
    mustard0
    bleach_hard_00_03_chaitanya
    bleach0
)
LABEL_CAD_MODELS=(
    006_mustard_bottle
    021_bleach_cleanser
    021_bleach_cleanser
)

# These five conditions are the common predictor-building conditions used by
# 2-risk_label.py. The four additional blackout episodes remain CI/evaluation
# episodes and are intentionally NOT added to --corruption_lists.
COMMON_CONDITIONS=(
    _occ40
    _black10
    _clean
    _drop60
    _occ60
)
CI_CONDITIONS=(
    _black10_2
    _black10_3
    _black10_4
    _black10_5
)

# Evaluation order: clean first so one compatible threshold pair is established
# before the remaining eight conditions in the same 3-train_evaluation.py run.
ALL_CONDITIONS=(
    _clean
    _occ40
    _occ60
    _drop60
    _black10
    _black10_2
    _black10_3
    _black10_4
    _black10_5
)


fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || fail "$2 does not exist: $1"
}

require_dir() {
    [[ -d "$1" ]] || fail "$2 does not exist: $1"
}

require_file "$MANIFEST_CONFIG" "Manifest configuration"
copy_if_exists() {
    local src=$1
    local dst_dir=$2
    if [[ -f "$src" ]]; then
        cp "$src" "$dst_dir/"
    fi
}

resolve_fp_mesh() {
    local model_dir=$1
    local override=${2:-}
    local candidate

    if [[ -n "$override" ]]; then
        require_file "$override" "FoundationPose mesh override"
        printf '%s\n' "$override"
        return 0
    fi

    for candidate in \
        "$model_dir/textured_simple.obj" \
        "$model_dir/textured.obj" \
        "$model_dir/textured.ply"; do
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    fail "No FoundationPose mesh found in $model_dir (expected textured_simple.obj, textured.obj, or textured.ply)"
}


# ============================================================
# 1. Required source files and clean Git state
# ============================================================
CODE_FILES=(
    "1-build_dataset_manifest_all.py"
    "2-risk_label.py"
    "3-train_evaluation.py"
    "b5_policy.py"
    "run.sh"
    "tests/test_manifest_builder.py"
    "tests/test_manifest_consumers.py"
    "tests/test_b5_shared_policy.py"
    "tests/test_b5_policy_state.py"
    "tests/test_reproduction_script.py"
)

for file in "${CODE_FILES[@]}"; do
    require_file "$file" "Versioned code file"
done

if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    GIT_AVAILABLE=true
    GIT_COMMIT=$(git rev-parse HEAD)
    GIT_STATUS=$(git status --porcelain=v1 --untracked-files=all)
    if [[ -n "$GIT_STATUS" ]]; then
        echo "$GIT_STATUS" >&2
        fail "Working tree is dirty. Commit/stash all tracked and untracked changes before a formal run."
    fi
    REPO_PREFIX=$(git rev-parse --show-prefix)
    for file in "${CODE_FILES[@]}"; do
        git cat-file -e "${GIT_COMMIT}:${REPO_PREFIX}${file}" 2>/dev/null || \
            fail "$file is not present in commit $GIT_COMMIT"
    done
    VERSION_TAG=${GIT_COMMIT:0:12}
else
    GIT_AVAILABLE=false
    GIT_COMMIT=not_available
    GIT_STATUS=""
    REPO_PREFIX=""
    VERSION_TAG=sourcehash
    echo "[WARNING] No Git repository detected; using immutable source-file SHA-256 evidence."
fi


# ============================================================
# 2. Runtime prerequisites
# ============================================================
require_file "$CONDA_SH" "Conda shell initialization"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

require_dir "$DATASET_ROOT" "Corrupted dataset root"
require_dir "$GT_ROOT" "GT dataset root"
require_dir "$RESULT_ROOT" "Prediction/result root"
require_dir "$CAD_MODEL_ROOT/006_mustard_bottle" "Mustard CAD model directory"
require_dir "$CAD_MODEL_ROOT/021_bleach_cleanser" "Bleach CAD model directory"

[[ -n "$SE3TRACKNET_WEIGHTS_ROOT" ]] || \
    fail "Set SE3TRACKNET_WEIGHTS_ROOT to the YCBInEOAT_weights directory."
require_dir "$SE3TRACKNET_WEIGHTS_ROOT" "SE3TrackNet YCBInEOAT weights root"
for object_name in mustard_bottle bleach_cleanser; do
    require_file "$SE3TRACKNET_WEIGHTS_ROOT/$object_name/model_best_val.pth.tar" \
        "$object_name SE3TrackNet checkpoint"
    require_file "$SE3TRACKNET_WEIGHTS_ROOT/$object_name/mean.npy" \
        "$object_name normalization mean"
    require_file "$SE3TRACKNET_WEIGHTS_ROOT/$object_name/std.npy" \
        "$object_name normalization standard deviation"
done

# All three base sequences use their own official first-frame mask.
for base in "${BASE_SEQUENCES[@]}"; do
    require_dir "$GT_ROOT/$base/annotated_poses" "$base GT pose directory"
    require_file "$GT_ROOT/$base/init_mask.png" "$base init_mask.png"
done

# FoundationPose is intentionally kept in a separate conda environment.
require_file "$FOUNDATIONPOSE_PYTHON" "FoundationPose environment Python"
require_dir "$FOUNDATIONPOSE_DIR" "FoundationPose repository"
require_file "$FOUNDATIONPOSE_REFINER_WEIGHT" "FoundationPose frozen refiner weight"
require_file "$FOUNDATIONPOSE_SCORER_WEIGHT" "FoundationPose frozen scorer weight"
require_file "$FOUNDATIONPOSE_DIR/learning/training/predict_pose_refine.py" \
    "FoundationPose refiner predictor source"
require_file "$FOUNDATIONPOSE_DIR/learning/training/predict_score.py" \
    "FoundationPose scorer predictor source"

FOUNDATIONPOSE_MUSTARD_MESH=$(resolve_fp_mesh \
    "$CAD_MODEL_ROOT/006_mustard_bottle" \
    "$FOUNDATIONPOSE_MUSTARD_MESH")
FOUNDATIONPOSE_BLEACH_MESH=$(resolve_fp_mesh \
    "$CAD_MODEL_ROOT/021_bleach_cleanser" \
    "$FOUNDATIONPOSE_BLEACH_MESH")

require_file "$CAD_MODEL_ROOT/006_mustard_bottle/points.xyz" "Mustard model points"
require_file "$CAD_MODEL_ROOT/021_bleach_cleanser/points.xyz" "Bleach model points"


# ============================================================
# 3. Create run directory and start complete logging
# ============================================================
RUN_TIME=$(date -u +"%Y%m%dT%H%M%SZ")
RUN_DIR="./reproduction_runs/run_${RUN_TIME}_${VERSION_TAG}_all27"
mkdir -p "$RUN_DIR"

if [[ "$GIT_AVAILABLE" == true ]]; then
    printf '%s\n' "$GIT_COMMIT" > "$RUN_DIR/git_commit.txt"
    git status --short > "$RUN_DIR/git_status.txt"
else
    printf '%s\n' "not_available" > "$RUN_DIR/git_commit.txt"
    : > "$RUN_DIR/git_status.txt"
fi

LOG_FILE="$RUN_DIR/full_run.log"
exec 3>&1 4>&2
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "Formal all-27 reproduction run started"
echo "UTC time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Git available: $GIT_AVAILABLE"
echo "Git commit: $GIT_COMMIT"
echo "Git dirty state: $([[ "$GIT_AVAILABLE" == true ]] && echo clean || echo not_applicable)"
echo "Run directory: $RUN_DIR"
echo "Experiment matrix: 3 base sequences x 9 conditions = 27 episodes"
echo "=========================================="


# ============================================================
# 4. Generate the exact 3 x 9 manifest configuration
# ============================================================


validate_reference_manifest_matrix() {
    python - "$REFERENCE_MANIFEST" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path)

bases = [
    "mustard0",
    "bleach0",
    "bleach_hard_00_03_chaitanya",
]
conditions = [
    "_clean",
    "_occ40",
    "_occ60",
    "_drop60",
    "_black10",
    "_black10_2",
    "_black10_3",
    "_black10_4",
    "_black10_5",
]
expected = {base + cond for base in bases for cond in conditions}
actual = set(df["sequence"].astype(str).unique())

missing = sorted(expected - actual)
extra = sorted(actual - expected)
if missing or extra or len(actual) != 27:
    raise SystemExit(
        "reference manifest is not the exact all-27 matrix; "
        f"missing={missing}, extra={extra}, unique={len(actual)}"
    )

if df.duplicated(["sequence", "frame_id"]).any():
    raise SystemExit("reference manifest has duplicate (sequence, frame_id)")
if df.duplicated(["sequence", "sequence_index"]).any():
    raise SystemExit("reference manifest has duplicate (sequence, sequence_index)")

print(f"Validated all-27 reference manifest: {path}")
print(f"Rows={len(df)}, episodes={len(actual)}")
PY
}

if [[ "$REBUILD_REFERENCE_MANIFEST" == "1" ]]; then
    echo "REBUILD_REFERENCE_MANIFEST=1: rebuilding all-27 frozen reference manifest..."
    rm -f "$REFERENCE_MANIFEST"
fi

if [[ ! -f "$REFERENCE_MANIFEST" ]]; then
    echo "Reference manifest not found. Building the all-27 frozen reference manifest..."
    python 1-build_dataset_manifest_all.py \
        --mode build-reference \
        --dataset_root "$DATASET_ROOT" \
        --gt_root "$GT_ROOT" \
        --result_root "$RESULT_ROOT" \
        --config "$MANIFEST_CONFIG" \
        --output "$REFERENCE_MANIFEST"
else
    echo "Using existing frozen reference manifest:"
    echo "$REFERENCE_MANIFEST"
fi

if ! validate_reference_manifest_matrix; then
    fail "Existing REFERENCE_MANIFEST is not the required 27-episode matrix. Re-run with REBUILD_REFERENCE_MANIFEST=1 after confirming the dataset/results are final."
fi


# ============================================================
# 5. Exact code version, source bundle, and environment provenance
# ============================================================
{
    echo "experiment_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "git_available=$GIT_AVAILABLE"
    echo "git_commit=$GIT_COMMIT"
    echo "experiment_matrix=3_base_sequences_x_9_conditions"
    echo "reference_manifest=$REFERENCE_MANIFEST"
    if [[ "$GIT_AVAILABLE" == true ]]; then
        echo "git_dirty=false"
        echo "git_branch=$(git branch --show-current || true)"
        echo "git_describe=$(git describe --always --dirty --tags 2>/dev/null || echo unavailable)"
    else
        echo "git_dirty=not_applicable"
        echo "git_branch=not_available"
        echo "git_describe=not_available"
        echo "version_evidence=source_file_sha256_and_source_bundle_sha256"
    fi
} > "$RUN_DIR/code_version.txt"
: > "$RUN_DIR/git_status_porcelain.txt"

{
    echo "working_sha256 commit_blob_sha256 path"
    for file in "${CODE_FILES[@]}"; do
        working_hash=$(sha256sum "$file" | awk '{print $1}')
        if [[ "$GIT_AVAILABLE" == true ]]; then
            commit_hash=$(git show "${GIT_COMMIT}:${REPO_PREFIX}${file}" | sha256sum | awk '{print $1}')
            [[ "$working_hash" == "$commit_hash" ]] || \
                fail "Working file differs from the cited commit despite clean-state check: $file"
        else
            commit_hash=not_available
        fi
        echo "$working_hash $commit_hash $file"
    done
} > "$RUN_DIR/code_sha256.txt"

mkdir -p "$RUN_DIR/source/tests"
cp \
    1-build_dataset_manifest_all.py \
    2-risk_label.py \
    3-train_evaluation.py \
    b5_policy.py \
    run.sh \
    "$RUN_DIR/source/"
cp "$MANIFEST_CONFIG" "$RUN_DIR/source/manifest_config.json"

cp \
    tests/test_manifest_builder.py \
    tests/test_manifest_consumers.py \
    tests/test_b5_shared_policy.py \
    tests/test_b5_policy_state.py \
    tests/test_reproduction_script.py \
    "$RUN_DIR/source/tests/"

(
    cd "$RUN_DIR/source"
    find . -type f -print0 | sort -z | xargs -0 sha256sum > ../source_file_sha256.txt
    sha256sum ../source_file_sha256.txt > ../source_bundle_sha256.txt
)

python --version 2>&1 | tee "$RUN_DIR/python_version.txt"
pip freeze > "$RUN_DIR/pip_freeze.txt"
conda env export --no-builds > "$RUN_DIR/environment.yml"
cp "$RUN_DIR/environment.yml" "$RUN_DIR/source/environment.yml"

"$FOUNDATIONPOSE_PYTHON" --version > "$RUN_DIR/foundationpose_python_version.txt" 2>&1
"$FOUNDATIONPOSE_PYTHON" -m pip freeze > "$RUN_DIR/foundationpose_pip_freeze.txt"

FOUNDATIONPOSE_ENV_PREFIX=$(dirname "$(dirname "$FOUNDATIONPOSE_PYTHON")")
if conda env export -p "$FOUNDATIONPOSE_ENV_PREFIX" --no-builds \
    > "$RUN_DIR/foundationpose_environment.yml" \
    2> "$RUN_DIR/foundationpose_environment_export.stderr"; then
    cp "$RUN_DIR/foundationpose_environment.yml" \
        "$RUN_DIR/source/foundationpose_environment.yml"
else
    echo "[WARNING] conda env export for FoundationPose failed; pip freeze was still recorded."
fi

mkdir -p "$RUN_DIR/source/foundationpose_overrides/learning/training"
cp "$FOUNDATIONPOSE_DIR/learning/training/predict_pose_refine.py" \
    "$RUN_DIR/source/foundationpose_overrides/learning/training/"
cp "$FOUNDATIONPOSE_DIR/learning/training/predict_score.py" \
    "$RUN_DIR/source/foundationpose_overrides/learning/training/"

{
    echo "foundationpose_python=$FOUNDATIONPOSE_PYTHON"
    echo "foundationpose_env_prefix=$FOUNDATIONPOSE_ENV_PREFIX"
    echo "foundationpose_dir=$FOUNDATIONPOSE_DIR"
    echo "foundationpose_refiner_weight=$FOUNDATIONPOSE_REFINER_WEIGHT"
    echo "foundationpose_scorer_weight=$FOUNDATIONPOSE_SCORER_WEIGHT"
    echo "foundationpose_refine_iter=$FOUNDATIONPOSE_REFINE_ITER"
    echo "foundationpose_mustard_mesh=$FOUNDATIONPOSE_MUSTARD_MESH"
    echo "foundationpose_bleach_mesh=$FOUNDATIONPOSE_BLEACH_MESH"
    echo "fp_refine_chunk_size=$FP_REFINE_CHUNK_SIZE"
    echo "fp_score_chunk_size=$FP_SCORE_CHUNK_SIZE"
    if command -v git >/dev/null 2>&1 && \
       git -C "$FOUNDATIONPOSE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "foundationpose_git_commit=$(git -C "$FOUNDATIONPOSE_DIR" rev-parse HEAD)"
        echo "foundationpose_git_status_begin"
        git -C "$FOUNDATIONPOSE_DIR" status --porcelain=v1 || true
        echo "foundationpose_git_status_end"
    else
        echo "foundationpose_git_commit=not_available"
    fi
} > "$RUN_DIR/foundationpose_provenance.txt"

sha256sum \
    "$FOUNDATIONPOSE_REFINER_WEIGHT" \
    "$FOUNDATIONPOSE_SCORER_WEIGHT" \
    "$FOUNDATIONPOSE_MUSTARD_MESH" \
    "$FOUNDATIONPOSE_BLEACH_MESH" \
    "$FOUNDATIONPOSE_DIR/learning/training/predict_pose_refine.py" \
    "$FOUNDATIONPOSE_DIR/learning/training/predict_score.py" \
    > "$RUN_DIR/foundationpose_artifact_sha256.txt"

uname -a > "$RUN_DIR/system_info.txt"
echo "SEED=$SEED" > "$RUN_DIR/seed.txt"
export PYTHONHASHSEED=$SEED

cp "$REFERENCE_MANIFEST" "$RUN_DIR/reference_manifest.csv"
sha256sum "$REFERENCE_MANIFEST" > "$RUN_DIR/reference_manifest_sha256.txt"

find \
    "$SE3TRACKNET_WEIGHTS_ROOT/mustard_bottle" \
    "$SE3TRACKNET_WEIGHTS_ROOT/bleach_cleanser" \
    -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$RUN_DIR/se3tracknet_weight_artifact_sha256.txt"

{
    echo "weights_root=$SE3TRACKNET_WEIGHTS_ROOT"
    echo "mustard0=mustard_bottle/model_best_val.pth.tar"
    echo "bleach0=bleach_cleanser/model_best_val.pth.tar"
    echo "bleach_hard_00_03_chaitanya=bleach_cleanser/model_best_val.pth.tar"
} > "$RUN_DIR/checkpoint_identifier.txt"

{
    echo "sequence,object_name,checkpoint,mean,std"
    echo "mustard0,mustard_bottle,mustard_bottle/model_best_val.pth.tar,mustard_bottle/mean.npy,mustard_bottle/std.npy"
    echo "bleach0,bleach_cleanser,bleach_cleanser/model_best_val.pth.tar,bleach_cleanser/mean.npy,bleach_cleanser/std.npy"
    echo "bleach_hard_00_03_chaitanya,bleach_cleanser,bleach_cleanser/model_best_val.pth.tar,bleach_cleanser/mean.npy,bleach_cleanser/std.npy"
} > "$RUN_DIR/prediction_sequence_checkpoint_mapping.csv"

find \
    "$CAD_MODEL_ROOT/006_mustard_bottle" \
    "$CAD_MODEL_ROOT/021_bleach_cleanser" \
    -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$RUN_DIR/cad_artifact_sha256.txt"


# ============================================================
# 6. Automated tests and frozen-manifest runtime verification
# ============================================================
echo "========== Automated Tests =========="
python -m unittest discover -s tests -p "test_*.py" -v 2>&1 | tee "$RUN_DIR/automated_tests.log"

echo "========== Step 1: Verify Frozen Reference Against Runtime =========="
python 1-build_dataset_manifest_all.py \
    --mode verify-runtime \
    --reference_manifest "$REFERENCE_MANIFEST" \
    --dataset_root "$DATASET_ROOT" \
    --gt_root "$GT_ROOT" \
    --result_root "$RESULT_ROOT" \
    --runtime_inventory "$RUN_DIR/runtime_inventory.csv" \
    --verification_report "$RUN_DIR/input_verification.json" \
    --hash_inventory_dir "$RUN_DIR"

python - "$RUN_DIR/reference_manifest.csv" "$RUN_DIR/association_protocol.txt" <<'PY'
import sys
import pandas as pd

manifest_path, output_path = sys.argv[1], sys.argv[2]
df = pd.read_csv(manifest_path)
fields = ["association_method", "association_reference", "association_description"]
for field in fields:
    values = df[field].dropna().unique().tolist()
    if len(values) != 1:
        raise SystemExit(f"expected exactly one {field}, found: {values}")
with open(output_path, "w", encoding="utf-8") as f:
    for field in fields:
        f.write(f"{field}={df[field].dropna().unique()[0]}\n")
PY

sha256sum \
    "$RUN_DIR/reference_manifest.csv" \
    "$RUN_DIR/runtime_inventory.csv" \
    "$RUN_DIR/input_verification.json" \
    "$RUN_DIR/dataset_artifact_sha256.csv" \
    "$RUN_DIR/prediction_artifact_sha256.csv" \
    > "$RUN_DIR/input_inventory_sha256.txt"


# ============================================================
# 7. Generate labels for all 27 episodes without modifying 2-risk_label.py
#
# 2-risk_label.py supports only one ci_object per invocation. Therefore:
#   pass 1 -> common 15 + mustard0 black10_2..5
#   pass 2 -> common 15 + bleach0 black10_2..5
#   pass 3 -> common 15 + bleach_hard... black10_2..5
# The common rows must be identical; run.sh verifies that before merging.
# ============================================================
echo "========== Step 2: Risk Label Generation for all 27 episodes =========="
LABEL_RUN_ROOT="$RUN_DIR/labels/by_ci_object"
mkdir -p "$LABEL_RUN_ROOT"

RAW_LABEL_NAME="per_frame_label_threshold${RISK_THRESHOLD}.csv"
RAW_BALANCE_NAME="class_balance_summary_threshold${RISK_THRESHOLD}.csv"
RAW_QUADRANT_NAME="risk_quadrant_summary_threshold${RISK_THRESHOLD}.csv"

for ci_object in "${BASE_SEQUENCES[@]}"; do
    echo "------------------------------------------------------------"
    echo "Risk-label pass: ci_object=$ci_object"
    echo "------------------------------------------------------------"

    rm -f \
        "$RAW_LABEL_NAME" \
        "$RAW_BALANCE_NAME" \
        "$RAW_QUADRANT_NAME" \
        label_p_obs_threshold.json \
        label_p_prior_threshold.json

    python 2-risk_label.py \
        --manifest_path "$REFERENCE_MANIFEST" \
        --ycb_dir "$GT_ROOT" \
        --data_dir "$DATASET_ROOT" \
        --res_dir "$RESULT_ROOT" \
        --mesh_path_root "$CAD_MODEL_ROOT" \
        --target_seqs "${LABEL_TARGET_SEQS[@]}" \
        --corruption_lists "${COMMON_CONDITIONS[@]}" \
        --ci_object "$ci_object" \
        --ci_episode "${CI_CONDITIONS[@]}" \
        --cad_models_seq "${LABEL_CAD_MODELS[@]}" \
        --risk_threshold "$RISK_THRESHOLD" \
        --foundationpose_python "$FOUNDATIONPOSE_PYTHON" \
        --foundationpose_dir "$FOUNDATIONPOSE_DIR" \
        --foundationpose_refiner_weight "$FOUNDATIONPOSE_REFINER_WEIGHT" \
        --foundationpose_refine_iter "$FOUNDATIONPOSE_REFINE_ITER"

    require_file "$RAW_LABEL_NAME" "Per-frame label CSV for ci_object=$ci_object"
    require_file "$RAW_BALANCE_NAME" "Class-balance CSV for ci_object=$ci_object"
    require_file "$RAW_QUADRANT_NAME" "Risk-quadrant CSV for ci_object=$ci_object"
    require_file "label_p_obs_threshold.json" "Label observation threshold JSON for ci_object=$ci_object"
    require_file "label_p_prior_threshold.json" "Label prior threshold JSON for ci_object=$ci_object"

    ci_dir="$LABEL_RUN_ROOT/$ci_object"
    mkdir -p "$ci_dir"
    cp "$RAW_LABEL_NAME" "$ci_dir/"
    cp "$RAW_BALANCE_NAME" "$ci_dir/"
    cp "$RAW_QUADRANT_NAME" "$ci_dir/"
    cp label_p_obs_threshold.json "$ci_dir/"
    cp label_p_prior_threshold.json "$ci_dir/"
done

# ci_object is used only for the four extra final-rollout episodes, so the
# learned label probability thresholds should be semantically identical across
# the three runs. Compare parsed JSON with a small numeric tolerance rather than
# raw file hashes, because CUDA inference can introduce harmless last-bit noise.
python - \
    "$LABEL_RUN_ROOT/mustard0/label_p_obs_threshold.json" \
    "$LABEL_RUN_ROOT/bleach0/label_p_obs_threshold.json" \
    "$LABEL_RUN_ROOT/bleach_hard_00_03_chaitanya/label_p_obs_threshold.json" \
    "$LABEL_RUN_ROOT/mustard0/label_p_prior_threshold.json" \
    "$LABEL_RUN_ROOT/bleach0/label_p_prior_threshold.json" \
    "$LABEL_RUN_ROOT/bleach_hard_00_03_chaitanya/label_p_prior_threshold.json" <<'PY'
import json
import math
import sys

obs_paths = sys.argv[1:4]
prior_paths = sys.argv[4:7]

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def semantically_equal(a, b, path="root"):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise SystemExit(f"threshold JSON keys differ at {path}: {set(a)} vs {set(b)}")
        for key in a:
            semantically_equal(a[key], b[key], f"{path}.{key}")
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise SystemExit(f"threshold JSON list length differs at {path}")
        for i, (x, y) in enumerate(zip(a, b)):
            semantically_equal(x, y, f"{path}[{i}]")
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-6):
            raise SystemExit(f"threshold JSON numeric value differs at {path}: {a} vs {b}")
        return
    if a != b:
        raise SystemExit(f"threshold JSON value differs at {path}: {a!r} vs {b!r}")

for paths, label in [(obs_paths, "obs"), (prior_paths, "prior")]:
    ref = load(paths[0])
    for path in paths[1:]:
        semantically_equal(ref, load(path), label)
    print(f"Validated semantically identical label {label} threshold JSON across 3 ci_object runs")
PY

cp "$LABEL_RUN_ROOT/mustard0/label_p_obs_threshold.json" label_p_obs_threshold.json
cp "$LABEL_RUN_ROOT/mustard0/label_p_prior_threshold.json" label_p_prior_threshold.json
cp label_p_obs_threshold.json "$RUN_DIR/label_p_obs_threshold.json"
cp label_p_prior_threshold.json "$RUN_DIR/label_p_prior_threshold.json"
sha256sum \
    "$LABEL_RUN_ROOT/mustard0/label_p_obs_threshold.json" \
    "$LABEL_RUN_ROOT/bleach0/label_p_obs_threshold.json" \
    "$LABEL_RUN_ROOT/bleach_hard_00_03_chaitanya/label_p_obs_threshold.json" \
    "$LABEL_RUN_ROOT/mustard0/label_p_prior_threshold.json" \
    "$LABEL_RUN_ROOT/bleach0/label_p_prior_threshold.json" \
    "$LABEL_RUN_ROOT/bleach_hard_00_03_chaitanya/label_p_prior_threshold.json" \
    > "$RUN_DIR/label_probability_threshold_sha256.txt"

MASTER_LABEL_CSV="./per_frame_label_threshold${RISK_THRESHOLD}.csv"
MASTER_BALANCE_CSV="./class_balance_summary_threshold${RISK_THRESHOLD}.csv"
MASTER_QUADRANT_CSV="./risk_quadrant_summary_threshold${RISK_THRESHOLD}.csv"
MERGE_REPORT="$RUN_DIR/labels/master_label_merge_report.txt"

python - \
    "$LABEL_RUN_ROOT/mustard0/$RAW_LABEL_NAME" \
    "$LABEL_RUN_ROOT/bleach0/$RAW_LABEL_NAME" \
    "$LABEL_RUN_ROOT/bleach_hard_00_03_chaitanya/$RAW_LABEL_NAME" \
    "$MASTER_LABEL_CSV" \
    "$MASTER_BALANCE_CSV" \
    "$MASTER_QUADRANT_CSV" \
    "$MERGE_REPORT" <<'PY'
import sys
import numpy as np
import pandas as pd

(
    mustard_path,
    bleach0_path,
    bleach_hard_path,
    output_path,
    balance_path,
    quadrant_path,
    report_path,
) = sys.argv[1:]

sources = [
    ("mustard0", mustard_path),
    ("bleach0", bleach0_path),
    ("bleach_hard_00_03_chaitanya", bleach_hard_path),
]

frames = []
canonical_columns = None
for source_name, path in sources:
    df = pd.read_csv(path)
    if canonical_columns is None:
        canonical_columns = list(df.columns)
    elif list(df.columns) != canonical_columns:
        raise SystemExit(
            f"label CSV columns differ for {source_name}: "
            f"expected={canonical_columns}, actual={list(df.columns)}"
        )
    if df.duplicated(["sequence", "frame_id"]).any():
        raise SystemExit(f"{source_name} label CSV contains duplicate (sequence, frame_id)")
    df = df.copy()
    df["__source_ci_object"] = source_name
    frames.append(df)

combined = pd.concat(frames, ignore_index=True)
value_columns = [c for c in canonical_columns if c not in {"sequence", "frame_id"}]

# Every duplicated common-condition row must be numerically/string identical.
def equal_value(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    if isinstance(a, (int, float, np.integer, np.floating)) and \
       isinstance(b, (int, float, np.integer, np.floating)):
        return bool(np.isclose(float(a), float(b), rtol=1e-7, atol=1e-5, equal_nan=True))
    return str(a) == str(b)

conflicts = []
for key, group in combined.groupby(["sequence", "frame_id"], sort=False, dropna=False):
    if len(group) <= 1:
        continue
    ref = group.iloc[0]
    for row_idx in range(1, len(group)):
        row = group.iloc[row_idx]
        bad_cols = [c for c in value_columns if not equal_value(ref[c], row[c])]
        if bad_cols:
            conflicts.append((key, bad_cols, ref["__source_ci_object"], row["__source_ci_object"]))
            if len(conflicts) >= 10:
                break
    if len(conflicts) >= 10:
        break

if conflicts:
    raise SystemExit(f"common label rows differ across ci_object runs; examples={conflicts}")

# Input order makes the mustard0 pass canonical for duplicated common rows.
master = combined.drop_duplicates(["sequence", "frame_id"], keep="first").copy()
master = master.drop(columns=["__source_ci_object"])

bases = ["mustard0", "bleach0", "bleach_hard_00_03_chaitanya"]
conditions = [
    "_clean", "_occ40", "_occ60", "_drop60",
    "_black10", "_black10_2", "_black10_3", "_black10_4", "_black10_5",
]
expected = {base + cond for base in bases for cond in conditions}
actual = set(master["sequence"].astype(str).unique())
missing = sorted(expected - actual)
extra = sorted(actual - expected)
if missing or extra or len(actual) != 27:
    raise SystemExit(
        f"merged label CSV is not exact all-27 matrix: missing={missing}, extra={extra}, unique={len(actual)}"
    )

if master.duplicated(["sequence", "frame_id"]).any():
    raise SystemExit("merged master label CSV contains duplicate (sequence, frame_id)")

sort_cols = ["sequence"]
if "sequence_index" in master.columns:
    sort_cols.append("sequence_index")
else:
    sort_cols.append("frame_id")
master = master.sort_values(sort_cols, kind="stable").reset_index(drop=True)
master.to_csv(output_path, index=False)

balance = master.groupby("sequence").agg(
    Total_Frames=("obs_risk_label", "count"),
    Obs_Risk_Positive_Ratio=("obs_risk_label", lambda x: f"{x.mean()*100:.2f}%"),
    Prior_Risk_Positive_Ratio=("prior_risk_label", lambda x: f"{x.mean()*100:.2f}%"),
).reset_index()
balance.to_csv(balance_path, index=False)

quadrant_rows = []
for obs_state in [0, 1]:
    for prior_state in [0, 1]:
        count = int(
            ((master["obs_risk_label"] == obs_state) &
             (master["prior_risk_label"] == prior_state)).sum()
        )
        quadrant_rows.append({
            "obs_risk_label": obs_state,
            "prior_risk_label": prior_state,
            "count": count,
        })
pd.DataFrame(quadrant_rows).to_csv(quadrant_path, index=False)

with open(report_path, "w", encoding="utf-8") as f:
    f.write("All duplicate common-condition rows were verified consistent across three ci_object runs within numeric tolerance.\n")
    f.write(f"master_rows={len(master)}\n")
    f.write(f"master_episodes={master['sequence'].nunique()}\n")
    f.write("expected_episodes=27\n")

print(f"Merged master labels: {output_path}")
print(f"Rows={len(master)}, episodes={master['sequence'].nunique()}")
PY

require_file "$MASTER_LABEL_CSV" "Merged all-27 label CSV"
require_file "$MASTER_BALANCE_CSV" "Merged all-27 class-balance CSV"
require_file "$MASTER_QUADRANT_CSV" "Merged all-27 risk-quadrant CSV"
cp "$MASTER_LABEL_CSV" "$RUN_DIR/"
cp "$MASTER_BALANCE_CSV" "$RUN_DIR/"
cp "$MASTER_QUADRANT_CSV" "$RUN_DIR/"


# ============================================================
# 8. Build three evaluation CSVs in run.sh only
#
# For each target base:
#   target base       -> all 9 conditions retained
#   the other 2 bases -> only the 5 common predictor-building conditions
# This prevents black10_2..5 of the training bases from leaking into the
# 3-train_evaluation.py train/cal split while leaving all 9 target conditions
# available for test_df/test_ALL_df.
# ============================================================
EVAL_LABEL_DIR="$RUN_DIR/labels/evaluation_csv"
mkdir -p "$EVAL_LABEL_DIR"

python - "$MASTER_LABEL_CSV" "$EVAL_LABEL_DIR" "$RISK_THRESHOLD" <<'PY'
import os
import sys
import pandas as pd

master_path, output_dir, risk_threshold = sys.argv[1:]
df = pd.read_csv(master_path)

bases = ["mustard0", "bleach0", "bleach_hard_00_03_chaitanya"]
common_conditions = ["_occ40", "_black10", "_clean", "_drop60", "_occ60"]
all_conditions = [
    "_clean", "_occ40", "_occ60", "_drop60",
    "_black10", "_black10_2", "_black10_3", "_black10_4", "_black10_5",
]

os.makedirs(output_dir, exist_ok=True)
for target in bases:
    train_bases = [base for base in bases if base != target]
    keep = {target + cond for cond in all_conditions}
    for train_base in train_bases:
        keep.update(train_base + cond for cond in common_conditions)

    sub = df[df["sequence"].astype(str).isin(keep)].copy()
    actual = set(sub["sequence"].astype(str).unique())
    missing = sorted(keep - actual)
    extra = sorted(actual - keep)
    if missing or extra or len(actual) != 19:
        raise SystemExit(
            f"{target}: evaluation CSV must contain 19 sequences "
            f"(9 target + 5 + 5 train); missing={missing}, extra={extra}, unique={len(actual)}"
        )
    if sub.duplicated(["sequence", "frame_id"]).any():
        raise SystemExit(f"{target}: duplicate (sequence, frame_id) in evaluation CSV")

    sort_cols = ["sequence", "sequence_index"] if "sequence_index" in sub.columns else ["sequence", "frame_id"]
    sub = sub.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    out = os.path.join(output_dir, f"per_frame_label_eval_{target}_threshold{risk_threshold}.csv")
    sub.to_csv(out, index=False)
    print(f"Built {out}: rows={len(sub)}, sequences={sub['sequence'].nunique()}")
PY


# ============================================================
# 9. Write the full 27-episode execution matrix
# ============================================================
python - "$RUN_DIR/experiment_matrix.csv" "$RESULT_ROOT" "$GT_ROOT" "$CAD_MODEL_ROOT" \
    "$FOUNDATIONPOSE_MUSTARD_MESH" "$FOUNDATIONPOSE_BLEACH_MESH" <<'PY'
import os
import sys
import pandas as pd

output, result_root, gt_root, cad_root, mustard_mesh, bleach_mesh = sys.argv[1:]

bases = ["mustard0", "bleach0", "bleach_hard_00_03_chaitanya"]
conditions = [
    "_clean", "_occ40", "_occ60", "_drop60",
    "_black10", "_black10_2", "_black10_3", "_black10_4", "_black10_5",
]
train_map = {
    "mustard0": ["bleach0", "bleach_hard_00_03_chaitanya"],
    "bleach0": ["bleach_hard_00_03_chaitanya", "mustard0"],
    "bleach_hard_00_03_chaitanya": ["bleach0", "mustard0"],
}
rows = []
for base in bases:
    if base == "mustard0":
        point_path = os.path.join(cad_root, "006_mustard_bottle", "points.xyz")
        fp_mesh = mustard_mesh
    else:
        point_path = os.path.join(cad_root, "021_bleach_cleanser", "points.xyz")
        fp_mesh = bleach_mesh
    for cond in conditions:
        sequence = base + cond
        rows.append({
            "base_sequence": base,
            "condition": cond.lstrip("_"),
            "sequence": sequence,
            "result_dir": os.path.join(result_root, base, sequence),
            "gt_dir": os.path.join(gt_root, base, "annotated_poses"),
            "point_path": point_path,
            "foundationpose_mesh_file": fp_mesh,
            "train_seqs": " ".join(train_map[base]),
        })
pd.DataFrame(rows).to_csv(output, index=False)
print(f"Execution matrix saved: {output} (27 rows)")
PY


# ============================================================
# 10. Evaluate one base at a time, each with all 9 conditions
# ============================================================
echo "========== Step 3: Three base-sequence evaluations, 9 conditions each =========="
mkdir -p "$RUN_DIR/evaluation"

for base in "${BASE_SEQUENCES[@]}"; do
    case "$base" in
        mustard0)
            TRAIN_SEQS=(bleach0 bleach_hard_00_03_chaitanya)
            POINT_PATH="$CAD_MODEL_ROOT/006_mustard_bottle/points.xyz"
            FP_MESH="$FOUNDATIONPOSE_MUSTARD_MESH"
            ;;
        bleach0)
            TRAIN_SEQS=(bleach_hard_00_03_chaitanya mustard0)
            POINT_PATH="$CAD_MODEL_ROOT/021_bleach_cleanser/points.xyz"
            FP_MESH="$FOUNDATIONPOSE_BLEACH_MESH"
            ;;
        bleach_hard_00_03_chaitanya)
            TRAIN_SEQS=(bleach0 mustard0)
            POINT_PATH="$CAD_MODEL_ROOT/021_bleach_cleanser/points.xyz"
            FP_MESH="$FOUNDATIONPOSE_BLEACH_MESH"
            ;;
        *)
            fail "Unknown base sequence: $base"
            ;;
    esac

    GT_DIR="$GT_ROOT/$base/annotated_poses"
    EVAL_LABEL_CSV="$EVAL_LABEL_DIR/per_frame_label_eval_${base}_threshold${RISK_THRESHOLD}.csv"
    EVAL_DIR="$RUN_DIR/evaluation/$base"
    mkdir -p "$EVAL_DIR"

    require_file "$EVAL_LABEL_CSV" "$base evaluation label CSV"
    require_file "$POINT_PATH" "$base point cloud model"
    require_file "$FP_MESH" "$base FoundationPose mesh"
    require_dir "$GT_DIR" "$base GT directory"

    RESULT_DIRS=()
    for condition in "${ALL_CONDITIONS[@]}"; do
        sequence="${base}${condition}"
        result_dir="$RESULT_ROOT/$base/$sequence"
        require_dir "$result_dir" "$sequence prediction directory"
        RESULT_DIRS+=("$result_dir")
    done

    echo "------------------------------------------------------------"
    echo "Evaluating base: $base"
    echo "Train sequences: ${TRAIN_SEQS[*]}"
    echo "Point model: $POINT_PATH"
    echo "FoundationPose mesh: $FP_MESH"
    echo "Conditions: ${ALL_CONDITIONS[*]}"
    echo "------------------------------------------------------------"

    # Threshold files are global fixed names in 3-train_evaluation.py. Remove the
    # previous base's files so this base establishes its own context first.
    rm -f p_obs_threshold.json p_prior_threshold.json

    # Fixed-name outputs are also global. Remove them before the new base so a
    # failed/incomplete run can never be mistaken for current output.
    rm -f \
        "checkpoint2_blackout_frame_intervals_threshold${RISK_THRESHOLD}.csv" \
        "checkpoint2_paired_auc_B5_vs_B1_threshold${RISK_THRESHOLD}.csv" \
        "checkpoint2_paired_recovery_B5_vs_B1_threshold${RISK_THRESHOLD}.csv" \
        "checkpoint2_probability_calibration_metrics_threshold${RISK_THRESHOLD}.csv" \
        "reliability_diagram_observation_risk_threshold${RISK_THRESHOLD}.png" \
        "reliability_diagram_prior_risk_threshold${RISK_THRESHOLD}.png" \
        "trajectory_recovery_plot_threshold${RISK_THRESHOLD}.png"
    rm -f checkpoint2_full_metrics_episode_summary_*_threshold"${RISK_THRESHOLD}".csv

    python 3-train_evaluation.py \
        --csv_path "$EVAL_LABEL_CSV" \
        --manifest_path "$REFERENCE_MANIFEST" \
        --result_dir "${RESULT_DIRS[@]}" \
        --gt_dir "$GT_DIR" \
        --point_path "$POINT_PATH" \
        --train_seqs "${TRAIN_SEQS[@]}" \
        --test_base_seq "$base" \
        --data_dir "$DATASET_ROOT" \
        --risk_threshold "$RISK_THRESHOLD" \
        --blackout_min_frames 10 \
        --ycbineoat_root "$GT_ROOT" \
        --foundationpose_python "$FOUNDATIONPOSE_PYTHON" \
        --foundationpose_dir "$FOUNDATIONPOSE_DIR" \
        --foundationpose_mesh_file "$FP_MESH" \
        --foundationpose_refiner_weight "$FOUNDATIONPOSE_REFINER_WEIGHT" \
        --foundationpose_refine_iter "$FOUNDATIONPOSE_REFINE_ITER" \
        --bootstrap_samples "$BOOTSTRAP_SAMPLES" \
        --seed "$SEED"

    require_file p_obs_threshold.json "$base frozen observation threshold JSON"
    require_file p_prior_threshold.json "$base frozen prior threshold JSON"

    # Archive the object-specific threshold files immediately.
    cp p_obs_threshold.json "$EVAL_DIR/p_obs_threshold.json"
    cp p_prior_threshold.json "$EVAL_DIR/p_prior_threshold.json"
    sha256sum p_obs_threshold.json p_prior_threshold.json \
        > "$EVAL_DIR/frozen_probability_threshold_sha256.txt"

    # Every per-frame log has the episode name in its filename, so all 9 can be
    # archived losslessly even though the Python evaluator is unchanged.
    for condition in "${ALL_CONDITIONS[@]}"; do
        episode="${base}${condition}"
        log_file="checkpoint2_per_frame_${episode}_log_threshold${RISK_THRESHOLD}.csv"
        require_file "$log_file" "$episode per-frame evaluation log"
        cp "$log_file" "$EVAL_DIR/"
    done

    # Archive fixed-name per-base outputs before the next base overwrites them.
    FIXED_OUTPUTS=(
        "checkpoint2_blackout_frame_intervals_threshold${RISK_THRESHOLD}.csv"
        "checkpoint2_paired_auc_B5_vs_B1_threshold${RISK_THRESHOLD}.csv"
        "checkpoint2_paired_recovery_B5_vs_B1_threshold${RISK_THRESHOLD}.csv"
        "checkpoint2_probability_calibration_metrics_threshold${RISK_THRESHOLD}.csv"
        "reliability_diagram_observation_risk_threshold${RISK_THRESHOLD}.png"
        "reliability_diagram_prior_risk_threshold${RISK_THRESHOLD}.png"
        "trajectory_recovery_plot_threshold${RISK_THRESHOLD}.png"
    )
    for output in "${FIXED_OUTPUTS[@]}"; do
        require_file "$output" "$base evaluation output"
        cp "$output" "$EVAL_DIR/"
    done

    SUMMARY_FILE="checkpoint2_full_metrics_episode_summary_${base}_clean_threshold${RISK_THRESHOLD}.csv"
    require_file "$SUMMARY_FILE" "$base full metrics summary"
    cp "$SUMMARY_FILE" "$EVAL_DIR/"

    # Validate that the paired-blackout logic saw exactly the five intended
    # blackout episodes for this base.
    BASE_INTERVAL_CSV="$EVAL_DIR/checkpoint2_blackout_frame_intervals_threshold${RISK_THRESHOLD}.csv"
    python - "$BASE_INTERVAL_CSV" "$base" <<'PY'
import sys
import pandas as pd

path, base = sys.argv[1:]
df = pd.read_csv(path)
required = {
    "episode", "blackout_start_index", "blackout_end_index", "recovery_index",
    "blackout_start_frame", "blackout_end_frame", "recovery_frame",
}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"{base}: blackout interval table missing columns: {sorted(missing)}")

expected = {
    base + "_black10",
    base + "_black10_2",
    base + "_black10_3",
    base + "_black10_4",
    base + "_black10_5",
}
actual = set(df["episode"].astype(str))
if len(df) != 5 or df["episode"].nunique() != 5 or actual != expected:
    raise SystemExit(
        f"{base}: expected exactly five blackout intervals {sorted(expected)}, "
        f"found rows={len(df)}, episodes={sorted(actual)}"
    )
if (df["blackout_start_index"] > df["blackout_end_index"]).any():
    raise SystemExit(f"{base}: blackout interval has start_index > end_index")
if (df["recovery_index"] != df["blackout_end_index"] + 1).any():
    raise SystemExit(f"{base}: recovery_index must immediately follow blackout_end_index")
print(f"Validated five blackout intervals for {base}: {path}")
PY

    echo "Completed and archived all 9 conditions for $base -> $EVAL_DIR"
done


# ============================================================
# 11. Aggregate the three per-base blackout interval tables
# ============================================================
ALL_INTERVAL_CSV="$RUN_DIR/checkpoint2_blackout_frame_intervals_all_bases_threshold${RISK_THRESHOLD}.csv"
python - \
    "$RUN_DIR/evaluation/mustard0/checkpoint2_blackout_frame_intervals_threshold${RISK_THRESHOLD}.csv" \
    "$RUN_DIR/evaluation/bleach0/checkpoint2_blackout_frame_intervals_threshold${RISK_THRESHOLD}.csv" \
    "$RUN_DIR/evaluation/bleach_hard_00_03_chaitanya/checkpoint2_blackout_frame_intervals_threshold${RISK_THRESHOLD}.csv" \
    "$ALL_INTERVAL_CSV" <<'PY'
import sys
import pandas as pd

*inputs, output = sys.argv[1:]
dfs = [pd.read_csv(path) for path in inputs]
df = pd.concat(dfs, ignore_index=True)
if len(df) != 15 or df["episode"].nunique() != 15:
    raise SystemExit(
        f"expected 15 blackout intervals across 3 bases, found rows={len(df)}, "
        f"unique episodes={df['episode'].nunique()}"
    )
if df["episode"].duplicated().any():
    raise SystemExit("duplicate episode found in combined blackout interval table")
df.to_csv(output, index=False)
print(f"Validated combined blackout interval table: {output} (15 episodes)")
PY

# Validate the complete per-frame-log inventory: 3 bases x 9 conditions = 27.
python - "$RUN_DIR/evaluation" "$RISK_THRESHOLD" <<'PY'
import os
import sys

evaluation_root, risk_threshold = sys.argv[1:]
bases = ["mustard0", "bleach0", "bleach_hard_00_03_chaitanya"]
conditions = [
    "_clean", "_occ40", "_occ60", "_drop60",
    "_black10", "_black10_2", "_black10_3", "_black10_4", "_black10_5",
]
missing = []
for base in bases:
    for cond in conditions:
        episode = base + cond
        path = os.path.join(
            evaluation_root,
            base,
            f"checkpoint2_per_frame_{episode}_log_threshold{risk_threshold}.csv",
        )
        if not os.path.isfile(path):
            missing.append(path)
if missing:
    raise SystemExit(f"missing per-frame logs: {missing[:10]}")
print("Validated per-frame logs: 27/27 episodes")
PY


echo "VERIFY_COMMAND=sha256sum -c sha256.txt" > "$RUN_DIR/VERIFY_COMMAND.txt"

# ============================================================
# 12. Final summary, portable hashes, and ZIP
# ============================================================
echo "=========================================="
echo "All-27 reproduction completed successfully."
echo "Git commit: $GIT_COMMIT"
echo "Master label CSV: $MASTER_LABEL_CSV"
echo "Evaluated bases: ${BASE_SEQUENCES[*]}"
echo "Total evaluated episodes: 27"
echo "Total validated blackout episodes: 15"
echo "Outputs saved to: $RUN_DIR"
echo "The portable SHA-256 manifest will be generated after full_run.log is closed."
echo "=========================================="

# Stop all writes to full_run.log, then wait for tee to flush before hashing it.
exec 1>&3 2>&4
exec 3>&- 4>&-
wait

(
    cd "$RUN_DIR"
    find . -type f ! -name "sha256.txt" -print0 | sort -z | xargs -0 sha256sum > sha256.txt
    sha256sum -c sha256.txt
)

RUN_ZIP="${RUN_DIR}.zip"
python - "$RUN_DIR" "$RUN_ZIP" <<'PY'
import os
import sys
import zipfile

source, destination = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for root, _, files in os.walk(source):
        for filename in sorted(files):
            path = os.path.join(root, filename)
            archive.write(path, os.path.relpath(path, source))
PY
sha256sum "$RUN_ZIP" > "${RUN_ZIP}.sha256"

echo "Portable bundle verification passed: $RUN_DIR/sha256.txt"
echo "Reproduction ZIP created: $RUN_ZIP"
