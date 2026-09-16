import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
# 使用固定随机种子，保证可复现性
np.random.seed(42)

def apply_patch_depth_dropout(depth_img, gt_mask, dropout_rate, patch_size=20):
    
    """
    只在物体 gt_mask 区域内，成块挖掉 60% 面积的深度 (Simulating Specular Reflections)
    """
    depth_corrupted = depth_img.copy()
    
    # 提取物体 Mask 内的像素坐标
    y_indices, x_indices = np.where(gt_mask > 0)
    total_obj_pixels = len(y_indices)
    
    if total_obj_pixels == 0:
        return depth_corrupted
        
    target_drop_pixels = total_obj_pixels * dropout_rate # 目标挖掉的像素总数
    dropped_pixels = 0
    
    # 随机在物体表面贴 $20 \times 20$ 的黑色方块，直到挖掉 60% 面积
    while dropped_pixels < target_drop_pixels:
        # 随机挑选物体表面上的一个中心点
        idx = np.random.randint(0, total_obj_pixels)
        cy, cx = y_indices[idx], x_indices[idx]
        
        # 计算 20x20 方块的边界
        y1, y2 = max(0, cy - patch_size // 2), min(depth_img.shape[0], cy + patch_size // 2)
        x1, x2 = max(0, cx - patch_size // 2), min(depth_img.shape[1], cx + patch_size // 2)
        
        # 只在物体的 gt_mask 范围内清零
        patch_mask = gt_mask[y1:y2, x1:x2] > 0
        depth_corrupted[y1:y2, x1:x2][patch_mask] = 0
        
        dropped_pixels += np.sum(patch_mask)
        
    return depth_corrupted

def generate_dropout_corruptions(seq_path, out_base_dir,dropout_rate=0.6):
    seq_name = os.path.basename(seq_path.rstrip('/\\'))
    rgb_files = sorted(glob.glob(os.path.join(seq_path, "rgb", "*.png")))
    depth_files = sorted(glob.glob(os.path.join(seq_path, "depth", "*.png")))
    mask_files = sorted(glob.glob(os.path.join(seq_path, "gt_mask", "*.png")))


    mask_path = mask_files[0]
    gt_mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = gt_mask_img > 0
    doc_name = int(dropout_rate*100)
    
    print(f"正在为序列 {seq_name} 生成块状深度缺失数据集...")
    dir_drop = os.path.join(out_base_dir, f"{seq_name}_drop{doc_name}")
    os.makedirs(os.path.join(dir_drop, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(dir_drop, "depth"), exist_ok=True)

    for i in range(len(rgb_files)):
        rgb = cv2.imread(rgb_files[i])
        depth = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
        
        if i < len(mask_files):
            gt_mask_img = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
            gt_mask = (gt_mask_img > 0)
        else:
            gt_mask = (depth > 400) & (depth < 1100) # 40cm~1.1m

        # 应用块状深度缺失 (Patch Dropout)
        depth_corrupted = apply_patch_depth_dropout(depth, gt_mask, dropout_rate=dropout_rate, patch_size=20)
        
        cv2.imwrite(os.path.join(dir_drop, "rgb", os.path.basename(rgb_files[i])), rgb) # RGB 保持原样
        cv2.imwrite(os.path.join(dir_drop, "depth", os.path.basename(depth_files[i])), depth_corrupted)

    print(f"序列 {seq_name}_{doc_name}块状深度受损数据集成功生成！")



def generate_corruptions(seq_path, out_base_dir,occlusion_rate):
    seq_name = os.path.basename(seq_path.rstrip('/\\'))
    rgb_files = sorted(glob.glob(os.path.join(seq_path, "rgb_clean", "*.png")))
    depth_files = sorted(glob.glob(os.path.join(seq_path, "depth_clean", "*.png")))
    mask_files =   sorted(glob.glob(os.path.join(seq_path, "gt_mask", "*.png")))
    print(f"正在为序列 {seq_name} 生成 3 套受损数据集...")
    
    # 建立 3 个独立的受损输出文件夹
    dirs_occ = []
    for j in occlusion_rate:
        occ = int(j*100)
        dir_occ = os.path.join(out_base_dir, f"{seq_name}_occ{occ}")
        os.makedirs(os.path.join(dir_occ, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(dir_occ, "depth"), exist_ok=True)
        dirs_occ.append(dir_occ)
    dir_black = os.path.join(out_base_dir, f"{seq_name}_black10")
    os.makedirs(os.path.join(dir_black, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(dir_black, "depth"), exist_ok=True)

    total_frames = len(rgb_files)
    objs_h=[]
    for i in range(total_frames):
        rgb = cv2.imread(rgb_files[i])
        depth = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
        
        # ---------------- 1. Condition 1: 精准遮挡目标物体 60% 的面积 ----------------
        for j in range(len(occlusion_rate)):
            rgb_occ = rgb.copy()
            depth_occ = depth.copy()

            if  50 <= i < 300:
                gt_mask_img = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
                gt_mask = (gt_mask_img > 0)
                
                if np.sum(gt_mask) > 0:
                    # 用 gt_mask 精准找到瓶子在 2D 图像上的外接矩形框 [x, y, w, h]
                    y_indices, x_indices = np.where(gt_mask)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    
                    obj_h = y_max - y_min
                    obj_w = x_max - x_min                   
                    occ_h = int(obj_h * occlusion_rate[j])  
                    rgb_occ[y_min : y_min + occ_h, x_min : x_max] = 0
                    depth_occ[y_min : y_min + occ_h, x_min : x_max] = 0
                        
            print(dirs_occ[j])
            cv2.imwrite(os.path.join(dirs_occ[j], "rgb", os.path.basename(rgb_files[i])), rgb_occ)
            cv2.imwrite(os.path.join(dirs_occ[j], "depth", os.path.basename(depth_files[i])), depth_occ)


        # ---------------- 3. Condition 3: 10 帧完全黑屏 (第 150~160 帧完全断连) ----------------
        rgb_black = rgb.copy()
        depth_black = depth.copy()
        if 65 <= i < 75: # 10 帧全黑
            rgb_black[:] = 0    # 视觉完全缺失
            depth_black[:] = 0  # 深度完全缺失
            
        cv2.imwrite(os.path.join(dir_black, "rgb", os.path.basename(rgb_files[i])), rgb_black)
        cv2.imwrite(os.path.join(dir_black, "depth", os.path.basename(depth_files[i])), depth_black)

    print(f"序列 {seq_name} 的受损数据集生成完毕！")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="生成 YCBInEOAT 受损数据集")
    parser.add_argument('--dataset_base', type=str, default="./datasets/YCBInEOAT", help="原始数据集基础路径")
    parser.add_argument('--out_dir', type=str, default="./datasets/YCBInEOAT_Corrupted5", help="受损数据集输出路径")
    parser.add_argument('--sequences', nargs='+', default=[ "mustard0", "bleach0", "bleach_hard_00_03_chaitanya"], help="要处理的序列名称列表")  #
    parser.add_argument('--occlusion_rate', nargs='+',type=float, default=[0.4,0.6], help="遮挡率")
    parser.add_argument('--dropout_rate', type=float, default=0.6, help="dropout率")
    args = parser.parse_args()
    dataset_base = args.dataset_base
    out_dir = args.out_dir
    sequences = args.sequences
    occlusion_rate = args.occlusion_rate
    dropout_rate = args.dropout_rate

    for seq in sequences:
        seq_p = os.path.join(dataset_base, seq)
        if os.path.exists(seq_p):
            generate_corruptions(seq_p, out_dir,occlusion_rate)
            generate_dropout_corruptions(seq_p, out_dir,dropout_rate)


#=========================================以下为验证代码==================================
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # 1. 载入同一个帧的【原始深度图】与【受损深度图】
# clean_path = "./datasets/YCBInEOAT/bleach_hard_00_03_chaitanya/depth_drop60/1581269947277570589.png"
# corrupted_path = "./datasets/YCBInEOAT_Corrupted/bleach_hard_00_03_chaitanya_drop60/depth/1581269947277570589.png"

# # 读取并换算为米 (m)
# depth_clean = cv2.imread(clean_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
# depth_corrupted = cv2.imread(corrupted_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

# # 2. 计算两者之间的绝对像素差值
# depth_diff = np.abs(depth_clean - depth_corrupted)

# # 3. 绘制 1x3 对比大图
# plt.figure(figsize=(16, 5))

# # 子图 1: 原始干净深度图
# plt.subplot(1, 3, 1)
# plt.imshow(depth_clean, cmap='viridis', vmin=0.3, vmax=1.5)
# plt.colorbar(label='Depth (meters)')
# plt.title('1. Clean Original Depth', fontsize=12)
# plt.axis('off')

# # 子图 2: Dropout / 噪声后的受损深度图
# plt.subplot(1, 3, 2)
# plt.imshow(depth_corrupted, cmap='viridis', vmin=0.3, vmax=1.5)
# plt.colorbar(label='Depth (meters)')
# plt.title('2. Corrupted Depth (Dropout + Noise)', fontsize=12)
# plt.axis('off')

# # 子图 3: 深度损失残差图 (高亮显示被抹黑/破坏的区域)
# plt.subplot(1, 3, 3)
# plt.imshow(depth_diff, cmap='magma')
# plt.colorbar(label='Difference (meters)')
# plt.title('3. Absolute Depth Difference Map', fontsize=12)
# plt.axis('off')

# plt.tight_layout()
# plt.savefig('depth_corruption_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()

# print("深度对比图已成功生成并保存为: depth_corruption_comparison.png！")