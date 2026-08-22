#!/bin/bash
source /c/anaconda/etc/profile.d/conda.sh
conda activate yolov5

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

set -e

# ==============================
# 0. 基本设置
# ==============================
SEED=42
RISK_THRESHOLD=0.5
P_RISK_THRESHOLD=0.8

RUN_TIME=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="./reproduction_runs/run_${RUN_TIME}"

mkdir -p "$RUN_DIR"

LOG_FILE="$RUN_DIR/full_run.log"

# 所有终端输出同时写进 log
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "Reproduction run started"
echo "Time: $(date)"
echo "Run directory: $RUN_DIR"
echo "=========================================="

# ==============================
# 1. 记录代码版本
# ==============================
echo ""
echo "========== Code Version =========="

echo "Experiment timestamp: $(date)" \
> "$RUN_DIR/code_version.txt"

sha256sum \
1-build_dataset_manifest_all.py \
2-risk_label.py \
3-train_evaluation.py \
> "$RUN_DIR/code_sha256.txt"


# ==============================
# 2. 记录运行环境
# ==============================
echo ""
echo "========== Environment =========="

python --version | tee "$RUN_DIR/python_version.txt"

pip freeze > "$RUN_DIR/pip_freeze.txt"

uname -a > "$RUN_DIR/system_info.txt"

# ==============================
# 3. 记录随机种子
# ==============================
echo ""
echo "========== Random Seed =========="

echo "SEED=$SEED" | tee "$RUN_DIR/seed.txt"

export PYTHONHASHSEED=$SEED

# ==============================
# 4. 生成 dataset manifest
# ==============================
echo ""
echo "========== Step 1: Build Manifest =========="

echo "COMMAND: python 1-build_dataset_manifest_all.py"

python 1-build_dataset_manifest_all.py \
    --dataset_root ./datasets/YCBInEOAT_Corrupted \
    --gt_root ./datasets/YCBInEOAT \
    --result_root ./results_collection \
    --config ./manifest_config.json \
    --output ./dataset_manifest_all.csv

cp dataset_manifest_all.csv \
   "$RUN_DIR/dataset_manifest_all.csv"

# ==============================
# 5. 生成 risk labels
# ==============================
echo ""
echo "========== Step 2: Risk Label Generation =========="

echo "COMMAND: python 2-risk_label.py --manifest_path ./dataset_manifest_all.csv --risk_threshold $RISK_THRESHOLD --p_risk_threshold $P_RISK_THRESHOLD"

python 2-risk_label.py \
    --manifest_path ./dataset_manifest_all.csv \
    --ycb_dir ./datasets/YCBInEOAT \
    --data_dir ./datasets/YCBInEOAT_Corrupted \
    --res_dir ./results_collection --mesh_path_root ./datasets/YCB_Video_Models/CADmodels \
    --mesh_path_root ./datasets/YCB_Video_Models/CADmodels \
    --target_seqs mustard0 bleach_hard_00_03_chaitanya bleach0 \
    --corruption_lists _occ40 _black10 _clean _drop60 _occ60 \
    --ci_object "bleach_hard_00_03_chaitanya" \
    --ci_episode _black10_2 _black10_3 _black10_4 _black10_5 \
    --cad_models_seq 006_mustard_bottle 021_bleach_cleanser 021_bleach_cleanser 021_bleach_cleanser \
    --risk_threshold $RISK_THRESHOLD \
    --p_risk_threshold $P_RISK_THRESHOLD

cp "per_frame_label_threshold${RISK_THRESHOLD}.csv" \
   "$RUN_DIR/"

cp "class_balance_summary_threshold${RISK_THRESHOLD}.csv" \
   "$RUN_DIR/"

# ==============================
# 6. Clean run
# ==============================
echo ""
echo "========== Step 3: Clean Evaluation =========="

CLEAN_DIR="./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_clean"

echo "COMMAND: python 3-train_evaluation.py --result_dir $CLEAN_DIR"

python 3-train_evaluation.py \
    --csv_path "./per_frame_label_threshold${RISK_THRESHOLD}.csv" \
    --manifest_path "./dataset_manifest_all.csv" \
    --result_dir "$CLEAN_DIR" \
    --gt_dir ./datasets/YCBInEOAT/bleach_hard_00_03_chaitanya/annotated_poses \
    --point_path ./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/points.xyz \
    --train_seqs bleach0 mustard0 \
    --test_base_seq bleach_hard_00_03_chaitanya \
    --data_dir ./datasets/YCBInEOAT_Corrupted \
    --p_risk_threshold "$P_RISK_THRESHOLD" \
    --risk_threshold "$RISK_THRESHOLD" \
    --seed "$SEED"

# ==============================
# 7. Five blackout runs
# ==============================
echo ""
echo "========== Step 4: Five Blackout Evaluations =========="

BLACK1="./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10"
BLACK2="./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_2"
BLACK3="./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_3"
BLACK4="./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_4"
BLACK5="./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_5"

echo "COMMAND: python 3-train_evaluation.py --result_dir $BLACK1 $BLACK2 $BLACK3 $BLACK4 $BLACK5"

python 3-train_evaluation.py \
    --csv_path "./per_frame_label_threshold${RISK_THRESHOLD}.csv" \
    --manifest_path "./dataset_manifest_all.csv" \
    --result_dir \
        "$BLACK1" \
        "$BLACK2" \
        "$BLACK3" \
        "$BLACK4" \
        "$BLACK5" \
    --gt_dir ./datasets/YCBInEOAT/bleach_hard_00_03_chaitanya/annotated_poses \
    --point_path ./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/points.xyz \
    --train_seqs bleach0 mustard0 \
    --test_base_seq bleach_hard_00_03_chaitanya \
    --data_dir ./datasets/YCBInEOAT_Corrupted \
    --p_risk_threshold "$P_RISK_THRESHOLD" \
    --risk_threshold "$RISK_THRESHOLD" \
    --seed "$SEED"

# ==============================
# 8. 保存主要输出文件
# ==============================
echo ""
echo "========== Collect Outputs =========="

find . -maxdepth 1 \
    \( -name "checkpoint2_*.csv" \
    -o -name "*recovery*.csv" \
    -o -name "*paired*.csv" \
    -o -name "*log*.csv" \) \
    -exec cp {} "$RUN_DIR/" \;

# ==============================
# 9. 记录 blackout intervals
# ==============================
echo ""
echo "========== Blackout Episodes =========="

cat > "$RUN_DIR/blackout_episodes.txt" << EOF
bleach_hard_00_03_chaitanya_black10
bleach_hard_00_03_chaitanya_black10_2
bleach_hard_00_03_chaitanya_black10_3
bleach_hard_00_03_chaitanya_black10_4
bleach_hard_00_03_chaitanya_black10_5
EOF

# ==============================
# 10. SHA-256
# ==============================
echo ""
echo "========== SHA-256 =========="

find "$RUN_DIR" -type f \
    ! -name "sha256.txt" \
    -exec sha256sum {} \; \
    > "$RUN_DIR/sha256.txt"

cat "$RUN_DIR/sha256.txt"

echo ""
echo "=========================================="
echo "Reproduction completed successfully."
echo "Outputs saved to:"
echo "$RUN_DIR"
echo "=========================================="