import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 修改成你的 depth 路径
# =========================

"""
depth: datasets/YCBInEOAT_Corrupted/mustard0_clean/depth/1581120439070828983.png
rgb: datasets/YCBInEOAT_Corrupted/mustard0_black10/rgb/1581120439070828983.png


"""
depth_path = "datasets/YCBInEOAT_Corrupted/mustard0_clean/depth/1581120440294045357.png"


# =========================
# 2. 读取原始 depth
# =========================
depth_raw = cv2.imread(
    depth_path,
    cv2.IMREAD_UNCHANGED
)

if depth_raw is None:
    raise FileNotFoundError(
        f"Cannot read depth image: {depth_path}"
    )


print("========== RAW DEPTH ==========")
print("shape :", depth_raw.shape)
print("dtype :", depth_raw.dtype)
print("min   :", np.min(depth_raw))
print("max   :", np.max(depth_raw))


# =========================
# 3. 转成米
# 如果你的 depth 原始单位就是 mm
# =========================
depth = depth_raw.astype(np.float32) / 1000.0


print("\n========== DEPTH IN METERS ==========")
print("min :", np.nanmin(depth))
print("max :", np.nanmax(depth))


# =========================
# 4. 有效深度区域
# 与你 B5 中的定义保持一致
# =========================
valid = (
    np.isfinite(depth)
    & (depth > 0.05)
    & (depth < 5.0)
)

valid_count = np.count_nonzero(valid)
total_count = depth.size

print("\n========== VALID DEPTH ==========")
print("valid pixels :", valid_count)
print("total pixels :", total_count)
print(
    "valid ratio  :",
    valid_count / total_count
)


# =========================
# 5. 有效深度统计
# =========================
if valid_count > 0:

    values = depth[valid]

    print("\n========== DEPTH STATISTICS ==========")
    print("mean   :", np.mean(values), "m")
    print("median :", np.median(values), "m")
    print("std    :", np.std(values), "m")
    print("min    :", np.min(values), "m")
    print("max    :", np.max(values), "m")

    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(
            f"P{p:02d}    :",
            np.percentile(values, p),
            "m"
        )


# =========================
# 6. 可视化
# =========================
depth_show = depth.copy()

# 无效 depth 设置成 NaN
depth_show[~valid] = np.nan


fig, ax = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)


# ---- 原始 depth ----
im0 = ax[0].imshow(depth)
ax[0].set_title("Raw Depth (m)")
ax[0].axis("off")

plt.colorbar(
    im0,
    ax=ax[0],
    fraction=0.046,
    pad=0.04
)


# ---- 只显示有效 depth ----
im1 = ax[1].imshow(depth_show)
ax[1].set_title("Valid Depth")
ax[1].axis("off")

plt.colorbar(
    im1,
    ax=ax[1],
    fraction=0.046,
    pad=0.04
)


# ---- 有效 / 无效 mask ----
ax[2].imshow(
    valid,
    cmap="gray"
)

ax[2].set_title(
    f"Valid Depth Mask\n"
    f"ratio={valid_count / total_count:.3f}"
)

ax[2].axis("off")


plt.tight_layout()
plt.show()
