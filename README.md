Step1 Generating corruption datasets:
'''
python 0-corruption.py --dataset_base ./datasets/YCBInEOAT --out_dir ./datasets/YCBInEOAT_Corrupted --sequences mustard0 bleach_hard_00_03_chaitanya bleach0 --occlusion_rate 0.4 0.6 --dropout_rate 0.6
'''

Step2: Run SE(3)TrackNet predition.py to generates predictions in ./results/bleach0/, using the datasets you want to predict:
'''
python predict.py ^ --mode ycbineoat --YCBInEOAT_dir datasets\YCBInEOAT\bleach_hard_00_03_chaitanya --train_data_path datasets\YCBInEOAT_data\bleach_cleanser\train_data_blender_DR --ckpt_dir YCBInEOAT_weights\bleach_cleanser\model_best_val.pth.tar --mean_std_path YCBInEOAT_weights\bleach_cleanser --class_id 12 --model_path datasets\YCB_Video_Models\CADmodels\021_bleach_cleanser\textured.obj --outdir results/bleach_hard_00_03_chaitanya_black10
'''

Step 3: Evaluating ADD / ADD-S AUC metrics on prediction 
'''
python eval_ycbineoat.py --YCBInEOAT_dir ./datasets/YCBInEOAT --class_id 12 --ycb_dir ./datasets/YCB_Video_Models/ --res_dir ./results/
'''

Step 4: Modifying the config file: ''manifest_config.json''

Step 5: Generate reference_manifest.csv as data mapping:
'''
python 1-build_dataset_manifest_all.py ^
--mode build-reference ^
--dataset_root ./datasets/YCBInEOAT_Corrupted ^
--gt_root ./datasets/YCBInEOAT ^
--result_root ./results_collection ^
--config ./manifest_config.json ^
--output ./reference_manifest.csv
'''

Step6:  One-Step running
'''
bash run.sh
'''

