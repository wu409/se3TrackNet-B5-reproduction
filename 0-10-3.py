import os
import shutil


# ============================================================
# 修改这里
# ============================================================

rgb_dir = "/home/wyg/se3TrackNet-B5-reproduction/datasets/YCBInEOAT_Corrupted/mustard0_clean/rgb"
depth_dir = "/home/wyg/FoundationPose/demo_data/mustard0/depth"

# 匹配出来的 depth 保存到这里
output_depth_dir = "./datasets/YCBInEOAT_Corrupted/mustard0_clean/matched_depth"


# ============================================================
# 创建输出目录
# ============================================================

os.makedirs(output_depth_dir, exist_ok=True)


# ============================================================
# 获取 RGB 文件名
# ============================================================

rgb_files = {
    f for f in os.listdir(rgb_dir)
    if os.path.isfile(os.path.join(rgb_dir, f))
}

print(f"RGB 文件数量: {len(rgb_files)}")


# ============================================================
# 获取 Depth 文件名
# ============================================================

depth_files = {
    f for f in os.listdir(depth_dir)
    if os.path.isfile(os.path.join(depth_dir, f))
}

print(f"Depth 文件数量: {len(depth_files)}")


# ============================================================
# 找到文件名完全一致的文件
# ============================================================

matched_files = sorted(rgb_files & depth_files)

print(f"匹配成功数量: {len(matched_files)}")


# ============================================================
# 复制匹配的 Depth
# ============================================================

for filename in matched_files:

    src = os.path.join(
        depth_dir,
        filename
    )

    dst = os.path.join(
        output_depth_dir,
        filename
    )

    shutil.copy2(src, dst)


# ============================================================
# 检查 RGB 中哪些没有对应 Depth
# ============================================================

missing_depth = sorted(rgb_files - depth_files)

print("\n========== 完成 ==========")

print(f"RGB 总数量       : {len(rgb_files)}")
print(f"Depth 总数量     : {len(depth_files)}")
print(f"匹配成功         : {len(matched_files)}")
print(f"RGB 缺少 Depth   : {len(missing_depth)}")


if missing_depth:

    print("\n以下 RGB 没有对应的 Depth:")

    for filename in missing_depth[:20]:
        print(filename)

    if len(missing_depth) > 20:
        print(f"... 另外还有 {len(missing_depth) - 20} 个")


print(
    "\n匹配后的 Depth 已保存到:",
    os.path.abspath(output_depth_dir)
)
