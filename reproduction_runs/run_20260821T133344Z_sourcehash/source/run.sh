#!/bin/bash
set -Eeuo pipefail

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# ==============================
# 0. Formal-run inputs and clean Git state
# ==============================
SEED=${SEED:-42}
RISK_THRESHOLD=${RISK_THRESHOLD:-1.0}
P_RISK_THRESHOLD=${P_RISK_THRESHOLD:-0.8}

DATASET_ROOT=${DATASET_ROOT:-./datasets/YCBInEOAT_Corrupted}
GT_ROOT=${GT_ROOT:-./datasets/YCBInEOAT}
RESULT_ROOT=${RESULT_ROOT:-./results_collection}
CAD_MODEL_ROOT=${CAD_MODEL_ROOT:-./datasets/YCB_Video_Models/CADmodels}
REFERENCE_MANIFEST=${REFERENCE_MANIFEST:-"$SCRIPT_DIR/reference_manifest.csv"}
SE3TRACKNET_WEIGHTS_ROOT=${SE3TRACKNET_WEIGHTS_ROOT:-"$SCRIPT_DIR/YCBInEOAT_weights"}

CONDA_SH=${CONDA_SH:-/c/anaconda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-yolov5}

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

# Git is preferred for commit-level traceability, but source-hash mode is supported
# for a project directory that has never been initialized as a Git repository.
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

[[ -n "$REFERENCE_MANIFEST" ]] || \
    fail "Set REFERENCE_MANIFEST to the frozen manifest built once from a trusted copy."
require_file "$REFERENCE_MANIFEST" "Frozen reference manifest"
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
require_dir "$DATASET_ROOT" "Corrupted dataset root"
require_dir "$GT_ROOT" "GT dataset root"
require_dir "$RESULT_ROOT" "Prediction/result root"
require_dir "$CAD_MODEL_ROOT/006_mustard_bottle" "Mustard CAD model directory"
require_dir "$CAD_MODEL_ROOT/021_bleach_cleanser" "Bleach CAD model directory"
require_file "$CONDA_SH" "Conda shell initialization"

# Environment activation is recorded after all formal-run prerequisites pass.
source "$CONDA_SH"
conda activate "$CONDA_ENV"

RUN_TIME=$(date -u +"%Y%m%dT%H%M%SZ")
RUN_DIR="./reproduction_runs/run_${RUN_TIME}_${VERSION_TAG}"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/full_run.log"

# Keep the original terminal descriptors so logging can be closed before hashing.
exec 3>&1 4>&2
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "Formal reproduction run started"
echo "UTC time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Git available: $GIT_AVAILABLE"
echo "Git commit: $GIT_COMMIT"
echo "Git dirty state: $([[ "$GIT_AVAILABLE" == true ]] && echo clean || echo not_applicable)"
echo "Run directory: $RUN_DIR"
echo "=========================================="

# ==============================
# 1. Exact code version and hashes
# ==============================
{
    echo "experiment_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "git_available=$GIT_AVAILABLE"
    echo "git_commit=$GIT_COMMIT"
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

# Documentation/templates are useful bundle metadata, but they are not runtime
# dependencies and must never stop the experiment when absent.
for optional_file in \
    REPRODUCTION_README.md \
    manifest_config.example.json \
    .gitattributes \
    .gitignore.example; do
    if [[ -f "$optional_file" ]]; then
        cp "$optional_file" "$RUN_DIR/source/"
    else
        echo "[INFO] Optional bundle file not present; skipped: $optional_file"
    fi
done
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

# ==============================
# 2. Environment and input provenance
# ==============================
python --version 2>&1 | tee "$RUN_DIR/python_version.txt"
pip freeze > "$RUN_DIR/pip_freeze.txt"
conda env export --no-builds > "$RUN_DIR/environment.yml"
cp "$RUN_DIR/environment.yml" "$RUN_DIR/source/environment.yml"
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
    -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_DIR/cad_artifact_sha256.txt"

# ==============================
# 3. Automated input-association tests
# ==============================
echo "========== Automated Tests =========="
python -m unittest discover -s tests -p "test_*.py" -v 2>&1 | tee "$RUN_DIR/automated_tests.log"

# ==============================
# 4. Verify runtime inputs against the frozen reference
# ==============================
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

# ==============================
# 5. Generate risk labels with the shared free-running B5 policy
# ==============================
echo "========== Step 2: Risk Label Generation =========="
python 2-risk_label.py \
    --manifest_path "$REFERENCE_MANIFEST" \
    --ycb_dir "$GT_ROOT" \
    --data_dir "$DATASET_ROOT" \
    --res_dir "$RESULT_ROOT" \
    --mesh_path_root "$CAD_MODEL_ROOT" \
    --target_seqs mustard0 bleach_hard_00_03_chaitanya bleach0 \
    --corruption_lists _occ40 _black10 _clean _drop60 _occ60 \
    --ci_object bleach_hard_00_03_chaitanya \
    --ci_episode _black10_2 _black10_3 _black10_4 _black10_5 \
    --cad_models_seq 006_mustard_bottle 021_bleach_cleanser 021_bleach_cleanser \
    --risk_threshold "$RISK_THRESHOLD" \
    --p_risk_threshold "$P_RISK_THRESHOLD"

cp "per_frame_label_threshold${RISK_THRESHOLD}.csv" "$RUN_DIR/"
cp "class_balance_summary_threshold${RISK_THRESHOLD}.csv" "$RUN_DIR/"

# ==============================
# 6. Clean evaluation
# ==============================
CLEAN_DIR="$RESULT_ROOT/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_clean"
echo "========== Step 3: Clean Evaluation =========="
python 3-train_evaluation.py \
    --csv_path "./per_frame_label_threshold${RISK_THRESHOLD}.csv" \
    --manifest_path "$REFERENCE_MANIFEST" \
    --result_dir "$CLEAN_DIR" \
    --gt_dir "$GT_ROOT/bleach_hard_00_03_chaitanya/annotated_poses" \
    --point_path "$CAD_MODEL_ROOT/021_bleach_cleanser/points.xyz" \
    --train_seqs bleach0 mustard0 \
    --test_base_seq bleach_hard_00_03_chaitanya \
    --data_dir "$DATASET_ROOT" \
    --p_risk_threshold "$P_RISK_THRESHOLD" \
    --risk_threshold "$RISK_THRESHOLD" \
    --seed "$SEED"

# ==============================
# 7. Five blackout evaluations
# ==============================
BLACK1="$RESULT_ROOT/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10"
BLACK2="$RESULT_ROOT/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_2"
BLACK3="$RESULT_ROOT/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_3"
BLACK4="$RESULT_ROOT/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_4"
BLACK5="$RESULT_ROOT/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_5"

echo "========== Step 4: Five Blackout Evaluations =========="
python 3-train_evaluation.py \
    --csv_path "./per_frame_label_threshold${RISK_THRESHOLD}.csv" \
    --manifest_path "$REFERENCE_MANIFEST" \
    --result_dir "$BLACK1" "$BLACK2" "$BLACK3" "$BLACK4" "$BLACK5" \
    --gt_dir "$GT_ROOT/bleach_hard_00_03_chaitanya/annotated_poses" \
    --point_path "$CAD_MODEL_ROOT/021_bleach_cleanser/points.xyz" \
    --train_seqs bleach0 mustard0 \
    --test_base_seq bleach_hard_00_03_chaitanya \
    --data_dir "$DATASET_ROOT" \
    --p_risk_threshold "$P_RISK_THRESHOLD" \
    --risk_threshold "$RISK_THRESHOLD" \
    --seed "$SEED"

# ==============================
# 8. Collect and validate outputs
# ==============================
find . -maxdepth 1 \
    \( -name "checkpoint2_*.csv" \
    -o -name "*recovery*.csv" \
    -o -name "*paired*.csv" \
    -o -name "*log*.csv" \) \
    -exec cp {} "$RUN_DIR/" \;

BLACKOUT_INTERVAL_CSV="$RUN_DIR/checkpoint2_blackout_frame_intervals_threshold${RISK_THRESHOLD}.csv"
require_file "$BLACKOUT_INTERVAL_CSV" "Exact blackout interval table"
python - "$BLACKOUT_INTERVAL_CSV" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path)
required = {
    "episode", "blackout_start_index", "blackout_end_index", "recovery_index",
    "blackout_start_frame", "blackout_end_frame", "recovery_frame",
}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"blackout interval table missing columns: {sorted(missing)}")
if len(df) != 5 or df["episode"].nunique() != 5:
    raise SystemExit(f"expected five exact blackout intervals, found {len(df)} rows")
if (df["blackout_start_index"] > df["blackout_end_index"]).any():
    raise SystemExit("blackout interval has start_index > end_index")
if (df["recovery_index"] != df["blackout_end_index"] + 1).any():
    raise SystemExit("recovery_index must immediately follow blackout_end_index")
print(f"Validated exact blackout intervals: {path}")
PY

echo "VERIFY_COMMAND=sha256sum -c sha256.txt" > "$RUN_DIR/VERIFY_COMMAND.txt"

# Log every final message before closing the tee process.
echo "=========================================="
echo "Reproduction completed successfully."
echo "Git commit: $GIT_COMMIT"
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
