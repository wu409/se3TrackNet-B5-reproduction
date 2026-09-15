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
export OMP_NUM_THREADS=1

bash run_train.sh
'''

Step 4: Test with ''full'' mode or ''simple'' mode or  together
'''
export OMP_NUM_THREADS=4

bash run_test.sh --release /root/autodl-tmp/se3TrackNet-B5-reproduction/final_training_releases/train_20260914T125211Z_5a329d44  --variants full simple --check-only

bash run_test.sh --release /root/autodl-tmp/se3TrackNet-B5-reproduction/final_training_releases/train_20260914T125211Z_5a329d44  --variants full simple 
'''
