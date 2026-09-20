Step1 Generating corruption datasets:
'''
python 0-corruption.py --dataset_base ./datasets/YCBInEOAT --out_dir ./datasets/YCBInEOAT_Corrupted --sequences mustard0 bleach_hard_00_03_chaitanya bleach0 --occlusion_rate 0.4 0.6 --dropout_rate 0.6
'''

Step2: Run SE(3)TrackNet predition.py to generates predictions in ./results/bleach0/, using the datasets you want to predict:
'''
python predict.py ^
--mode ycbineoat ^
--YCBInEOAT_dir datasets\YCBInEOAT\mustard0 ^
--train_data_path datasets\YCBInEOAT_data\mustard_bottle\train_data_blender_DR ^
--ckpt_dir YCBInEOAT_weights\mustard_bottle\model_best_val.pth.tar ^
--mean_std_path YCBInEOAT_weights\mustard_bottle ^
--class_id 5 ^
--model_path datasets\YCB_Video_Models\CADmodels\006_mustard_bottle\textured.obj ^
--outdir results_collection/mustard0/mustard0_clean
'''


Step 3：Check and Train with optional sequences:
'''
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export B5_NUM_THREADS=4
export B5_IO_WORKERS=4
export SAM2_DIR=/root/autodl-tmp/sam2
export SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_s.yaml
export SAM2_CHECKPOINT="$SAM2_DIR/checkpoints/sam2.1_hiera_small.pt"

bash run_train.sh
'''

Step 4: Test with ''full'' mode or ''simple'' mode or  together
'''
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export B5_NUM_THREADS=4
export B5_IO_WORKERS=4
export SAM2_DIR=/root/autodl-tmp/sam2
export SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_s.yaml
export SAM2_CHECKPOINT="$SAM2_DIR/checkpoints/sam2.1_hiera_small.pt"

bash run_test.sh --release /root/autodl-tmp/se3TrackNet-B5-reproduction/final_training_releases/train_20260914T125211Z_5a329d44  --variants full simple --check-only

bash run_test.sh --release /root/autodl-tmp/se3TrackNet-B5-reproduction/final_training_releases/train_20260914T125211Z_5a329d44  --variants full simple 
'''


OR run all the experiments at once:
'''
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export B5_NUM_THREADS=4
export B5_IO_WORKERS=4
export SAM2_DIR=/root/autodl-tmp/sam2
export SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_s.yaml
export SAM2_CHECKPOINT="$SAM2_DIR/checkpoints/sam2.1_hiera_small.pt"

bash run_all.sh --skip-standalone-checks
'''


| 目录 | 做什么 | 保存什么 |
|---|---|---|
| **`01_full_release`** | 训练完整方法：先训练 q0，再进行 rollout refitting 得到 q1 | 最终冻结模型、校准器、阈值配置、源码快照等 |
| **`03_full_simple`** | 用 `01` 的冻结版本运行 **full 和 simple 测试** | 两种策略的 AUC、recovery、逐帧记录及汇总 CSV |
| **`04_q0_training`** | 准备并冻结**不进行 rollout refitting** 的 q0 消融版本 | no-rollout 模型与配置，主要产物在 `no_rollout_release/` |
| **`05_ablations`** | 运行消融测试，并结合已有 full/simple 结果进行汇总 | 消融的测试结果、对比 CSV |