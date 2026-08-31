import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


VALID_MIN_M = 0.05
VALID_MAX_M = 5.0


def main(args):
    # ============================================================
    # 1. 读取 depth
    # ============================================================
    depth_raw = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)

    if depth_raw is None:
        raise FileNotFoundError(f"Cannot read depth: {args.depth}")

    print("Depth raw dtype :", depth_raw.dtype)
    print("Depth raw shape :", depth_raw.shape)
    print("Depth raw min   :", np.min(depth_raw))
    print("Depth raw max   :", np.max(depth_raw))

    # 和你 evaluation 中保持完全一致：mm -> m
    depth = depth_raw.astype(np.float32) / 1000.0


    # ============================================================
    # 2. 读取 recovery mask
    # ============================================================
    mask_raw = cv2.imread(args.mask, cv2.IMREAD_UNCHANGED)

    if mask_raw is None:
        raise FileNotFoundError(f"Cannot read mask: {args.mask}")

    if mask_raw.ndim == 3:
        mask = np.any(mask_raw > 0, axis=2)
    else:
        mask = mask_raw > 0

    print("\nMask shape      :", mask.shape)
    print("Mask pixels     :", np.count_nonzero(mask))


    # ============================================================
    # 3. 检查尺寸
    # ============================================================
    if depth.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Depth/mask shape mismatch: "
            f"depth={depth.shape}, mask={mask.shape}"
        )


    # ============================================================
    # 4. 有效 depth 定义
    # ============================================================
    valid_depth = (
        np.isfinite(depth)
        & (depth > VALID_MIN_M)
        & (depth < VALID_MAX_M)
    )

    # mask ∩ valid_depth
    mask_valid = mask & valid_depth

    # mask 内无效 depth
    mask_invalid = mask & (~valid_depth)


    # ============================================================
    # 5. 基本统计
    # ============================================================
    mask_pixels = int(np.count_nonzero(mask))
    valid_pixels = int(np.count_nonzero(mask_valid))
    invalid_pixels = int(np.count_nonzero(mask_invalid))

    valid_ratio = (
        valid_pixels / mask_pixels
        if mask_pixels > 0
        else 0.0
    )

    print("\n========== MASK x DEPTH ==========")
    print(f"Mask pixels            : {mask_pixels}")
    print(f"Valid depth pixels     : {valid_pixels}")
    print(f"Invalid depth pixels   : {invalid_pixels}")
    print(f"Valid depth ratio      : {valid_ratio:.6f}")


    # ============================================================
    # 6. mask 内 depth 分布
    # ============================================================
    if valid_pixels > 0:
        d = depth[mask_valid]

        print("\n========== DEPTH IN MASK ==========")
        print(f"Mean       : {np.mean(d):.6f} m")
        print(f"Median     : {np.median(d):.6f} m")
        print(f"Std        : {np.std(d):.6f} m")
        print(f"Min        : {np.min(d):.6f} m")
        print(f"Max        : {np.max(d):.6f} m")
        print(f"P01        : {np.percentile(d, 1):.6f} m")
        print(f"P05        : {np.percentile(d, 5):.6f} m")
        print(f"P10        : {np.percentile(d, 10):.6f} m")
        print(f"P25        : {np.percentile(d, 25):.6f} m")
        print(f"P50        : {np.percentile(d, 50):.6f} m")
        print(f"P75        : {np.percentile(d, 75):.6f} m")
        print(f"P90        : {np.percentile(d, 90):.6f} m")
        print(f"P95        : {np.percentile(d, 95):.6f} m")
        print(f"P99        : {np.percentile(d, 99):.6f} m")

        # 一个很直观的深度跨度
        p10 = np.percentile(d, 10)
        p90 = np.percentile(d, 90)

        print(
            f"P90-P10 span: "
            f"{(p90 - p10) * 100:.3f} cm"
        )


    # ============================================================
    # 7. 可视化 mask ∩ depth
    # ============================================================

    # --- 图1：原始 mask
    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap="gray")
    plt.title("Recovery Mask")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        "check_recovery_mask.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()


    # --- 图2：整张 depth
    depth_vis = depth.copy()
    depth_vis[~valid_depth] = np.nan

    plt.figure(figsize=(8, 6))
    im = plt.imshow(depth_vis)
    plt.colorbar(im, label="Depth (m)")
    plt.title("Depth Image")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        "check_depth.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()


    # --- 图3：mask 内 depth
    masked_depth = np.full_like(
        depth,
        np.nan,
        dtype=np.float32
    )
    masked_depth[mask_valid] = depth[mask_valid]

    plt.figure(figsize=(8, 6))
    im = plt.imshow(masked_depth)
    plt.colorbar(im, label="Depth (m)")
    plt.title("Valid Depth Inside Recovery Mask")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        "check_mask_depth.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()


    # --- 图4：mask 中无效 depth 的位置
    plt.figure(figsize=(8, 6))
    plt.imshow(mask_invalid, cmap="gray")
    plt.title(
        f"Invalid Depth Inside Mask "
        f"({invalid_pixels}/{mask_pixels})"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        "check_mask_invalid_depth.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()


    # --- 图5：depth histogram
    if valid_pixels > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(
            depth[mask_valid],
            bins=80
        )
        plt.xlabel("Depth (m)")
        plt.ylabel("Number of Pixels")
        plt.title("Depth Distribution Inside Recovery Mask")
        plt.tight_layout()
        plt.savefig(
            "check_mask_depth_histogram.png",
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()


    print("\nSaved:")
    print("  check_recovery_mask.png")
    print("  check_depth.png")
    print("  check_mask_depth.png")
    print("  check_mask_invalid_depth.png")
    print("  check_mask_depth_histogram.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--depth",
        required=True,
        help="YCBInEOAT depth png"
    )

    parser.add_argument(
        "--mask",
        required=True,
        help="recovery mask png"
    )

    args = parser.parse_args()

    main(args)