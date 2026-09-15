import os
import json
import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from b5_revision import CONFIG as B5_POLICY_CONFIG
import cv2
from scipy.spatial.transform import Rotation as R_sci


DEFAULT_FOUNDATIONPOSE_PYTHON = os.environ.get(
    "FOUNDATIONPOSE_PYTHON",
    "/home/wyg/anaconda3/envs/foundationpose/bin/python",
)
DEFAULT_FOUNDATIONPOSE_DIR = os.environ.get(
    "FOUNDATIONPOSE_DIR",
    "/home/wyg/FoundationPose",
)


def se3_log_map(T):
    R_mat, t_vec = T[:3, :3], T[:3, 3]
    w_vec = R_sci.from_matrix(R_mat).as_rotvec()
    return np.concatenate([t_vec, w_vec])


def se3_exp_map(delta):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_sci.from_rotvec(delta[3:]).as_matrix()
    T[:3, 3] = delta[:3]
    return T


def compute_se3_prior(T_prev1, T_prev2):
    delta = se3_log_map(np.linalg.inv(T_prev2) @ T_prev1)
    return T_prev1 @ se3_exp_map(delta)


# ---------------------------------------------------------------------
# Blackout detection is based ONLY on the current full depth frame.
# x4/support is NOT used to decide whether a frame is a blackout.
# ---------------------------------------------------------------------
DEPTH_BLACKOUT_VALID_MIN_M = 0.05
DEPTH_BLACKOUT_VALID_MAX_M = 5.0

# A complete blackout should leave almost no valid depth in the whole image.
# 1% is intentionally conservative: losing the target object's depth alone
# will not trigger blackout as long as the rest of the scene still has depth.
DEPTH_BLACKOUT_VALID_RATIO_THRESHOLD = 0.01

# Bound repeated reliance on the recursive temporal prior.
MAX_PRIOR_STREAK = 5


def detect_depth_blackout(
    depth_real,
    valid_min_m=DEPTH_BLACKOUT_VALID_MIN_M,
    valid_max_m=DEPTH_BLACKOUT_VALID_MAX_M,
    valid_ratio_threshold=DEPTH_BLACKOUT_VALID_RATIO_THRESHOLD,
):
    """
    Determine whether the CURRENT FRAME is a true depth blackout.

    This uses the WHOLE depth image, not CAD projection support (x4).

    A frame is considered blackout only when the fraction of physically valid
    depth pixels in the full image is extremely small.

    Returns
    -------
    is_blackout : bool
    diagnostics : dict
    """
    if depth_real is None:
        return True, {
            "depth_valid_pixels": 0,
            "depth_total_pixels": 0,
            "depth_valid_ratio": 0.0,
            "depth_blackout_threshold": float(valid_ratio_threshold),
        }

    depth = np.asarray(depth_real, dtype=np.float32)

    if depth.ndim > 2:
        depth = np.squeeze(depth)

    if depth.ndim != 2 or depth.size == 0:
        return True, {
            "depth_valid_pixels": 0,
            "depth_total_pixels": int(depth.size),
            "depth_valid_ratio": 0.0,
            "depth_blackout_threshold": float(valid_ratio_threshold),
        }

    valid = (
        np.isfinite(depth)
        & (depth > float(valid_min_m))
        & (depth < float(valid_max_m))
    )

    valid_pixels = int(np.count_nonzero(valid))
    total_pixels = int(depth.size)
    valid_ratio = (
        valid_pixels / float(total_pixels)
        if total_pixels > 0
        else 0.0
    )

    is_blackout = bool(
        valid_ratio <= float(valid_ratio_threshold)
    )

    return is_blackout, {
        "depth_valid_pixels": valid_pixels,
        "depth_total_pixels": total_pixels,
        "depth_valid_ratio": float(valid_ratio),
        "depth_blackout_threshold": float(valid_ratio_threshold),
    }


def b5_recovery_needed(
    depth_real,
    state,
    blackout_min_frames=10,
    max_prior_streak=MAX_PRIOR_STREAK,
    depth_blackout_valid_ratio_threshold=(
        DEPTH_BLACKOUT_VALID_RATIO_THRESHOLD
    ),
):
    """
    Helper for the evaluator/label generator to decide whether RGB should be
    loaded BEFORE calling b5_transition().

    IMPORTANT:
      - no x4/support
      - current full-depth frame decides blackout/non-blackout
      - recovery is needed on:
          A) first non-blackout frame after a long blackout
          B) after MAX_PRIOR_STREAK consecutive prior-reliance frames
    """
    is_blackout, depth_diag = detect_depth_blackout(
        depth_real,
        valid_ratio_threshold=(
            depth_blackout_valid_ratio_threshold
        ),
    )

    if is_blackout:
        return False, None, depth_diag

    consecutive_blackout = int(
        state.get("consecutive_blackout", 0)
    )
    prior_streak = int(
        state.get("prior_streak", 0)
    )

    if consecutive_blackout >= int(blackout_min_frames):
        return True, "blackout_exit", depth_diag

    if prior_streak >= int(max_prior_streak):
        return True, "prior_streak", depth_diag

    return False, None, depth_diag


def init_b5_state():
    return {
        # Exact manifest RGB paths, collected only until the first recovery.
        "sam2_rgb_paths": [],
        "sam2_segmentation_finished": False,
        "sam2_path_error": None,
        "consecutive_blackout": 0,
        "exited_blackout": False,

        "blackout_start_idx": 0,
        "blackout_end_idx": int(1e10),
        "blackout_start_frame": None,
        "blackout_end_frame": None,
        "last_blackout_idx": None,
        "last_blackout_frame": None,
        # Recovery-frame bookkeeping is intentionally split by trigger.
        # "recovery_frame" is retained as a deprecated alias for the most
        # recent recovery attempt; blackout reporting must use
        # "blackout_recovery_frame" / blackout_intervals instead.
        "recovery_frame": None,
        "last_recovery_frame": None,
        "blackout_recovery_frame": None,
        "prior_streak_recovery_frame": None,
        "blackout_intervals": [],

        # Explicitly initialized drift-control state.
        "prior_streak": 0,
        "prior_drift_score": 0.0,

        # Current full-frame depth blackout diagnostics.
        "is_depth_blackout": False,
        "depth_valid_ratio": np.nan,
        "depth_valid_pixels": 0,
        "depth_total_pixels": 0,

        # Last non-blackout B5 output. During a true blackout these are frozen,
        # therefore they represent the pre-blackout reference frame used to
        # construct rgb_template2.
        "last_reference_rgb_real": None,
        "last_reference_T_final": None,
        "last_reference_frame_id": None,

        # Previous-frame operational B5 output. Unlike last_reference_T_final,
        # this cache is updated on EVERY frame (including blackout frames).
        # It is used only for recovery motion-plausibility diagnostics/gating.
        "last_operational_T_final": None,
        "last_operational_frame_id": None,

        # First-frame template assets.
        "initial_rgb_real": None,
        "initial_mask": None,
        "init_mask_path": None,
        "rgb_template1": None,
        "template1_mask": None,
        "template1_bbox_xyxy": None,

        # Frozen pre-blackout Template2.
        "blackout_rgb_template2": None,
        "blackout_template2_mask": None,
        "blackout_template2_diagnostics": None,
        "blackout_reference_frame_id": None,
    }



# ---------------------------------------------------------------------
# Template-guided, mask-based FoundationPose recovery
#
# Recovery pipeline:
#   init_mask.png + first RGB
#       -> rgb_template1
#
#   cached pre-blackout/reference RGB + cached T_final + CAD
#       -> pose-guided search ROI
#       -> Template1 matching inside ROI
#       -> rgb_template2 + template2_mask
#
#   recovery-frame RGB
#       -> Template2 whole-frame matching
#       -> padded binary recovery mask
#
#   current RGB + current depth + recovery mask
#       -> FoundationPose.register(...)
#       -> T_recovery
# ---------------------------------------------------------------------

TEMPLATE1_ANGLE_SET_DEG = (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0)
TEMPLATE1_SCALE_MULTIPLIERS = (0.72, 0.85, 1.00, 1.15, 1.30)
TEMPLATE1_MIN_MATCH_SCORE = 0.18

TEMPLATE2_SCALES = (0.70, 0.82, 0.92, 1.00, 1.08, 1.18, 1.32, 1.45)
TEMPLATE2_ANGLES_DEG = (-35.0, -25.0, -15.0, 0.0, 15.0, 25.0, 35.0)
TEMPLATE2_MIN_MATCH_SCORE = 0.18

TEMPLATE2_CROP_MARGIN_PX = 8
REFERENCE_ROI_PADDING_FACTOR = 0.75
REFERENCE_ROI_MIN_PADDING_PX = 32
FOUNDATIONPOSE_MASK_PADDING_PX = 8

# Final CAD-mask recovery validation.
# The independent FoundationPose candidate is checked by projecting the CAD
# model with T_recovery and comparing the projected silhouette with the
# template-derived recovery mask.
RECOVERY_CAD_MASK_IOU_WEIGHT = 0.30
RECOVERY_CAD_MASK_COVERAGE_WEIGHT = 0.70
RECOVERY_CAD_MASK_LOW_THRESHOLD = 0.40
RECOVERY_CAD_MASK_HIGH_THRESHOLD = 0.70
RECOVERY_CAD_MIN_PROJECTED_PIXELS = 20

# FoundationPose official demos commonly use 5 registration refinement
# iterations. The B5 recovery default is increased to 10 to give the
# independent re-registration a little more convergence budget.
DEFAULT_FOUNDATIONPOSE_REFINE_ITER = 10


# ---------------------------------------------------------------------
# Recovery localization / pose-validity defaults.
# These are development-time safety gates, not learned scores.  They should
# be frozen before any genuinely untouched final evaluation.
# ---------------------------------------------------------------------
TEMPLATE2_TOP_K = 5
TEMPLATE2_PEAKS_PER_HYPOTHESIS = 3
TEMPLATE2_CANDIDATE_NMS_IOU = 0.45
RECOVERY_CANDIDATE_MIN_VALID_DEPTH_RATIO = 0.40
RECOVERY_CANDIDATE_MAX_DEPTH_DELTA_M = 0.35

# Harry-requested target-object pose-validity evidence.
#
# LEGACY thresholds below are retained for compatibility/diagnostics only.
# Active admission is configured in recovery_gate.py (unvalidated development).
# Former gate structure:
#   1) HARD catastrophe gates:
#        - finite pose / valid CAD projection
#        - previous-frame operational motion plausibility
#        - rendered-depth support and residual
#   2) SOFT consistency evidence:
#        - visible-mask IoU
#        - CAD coverage
#        - reprojection-center consistency
#        - Template2 ambiguity margin
#      At least RECOVERY_MIN_SOFT_EVIDENCE_PASSES soft cues must agree.
#
# These values are development thresholds and must be frozen before the
# untouched final evaluation.
RECOVERY_MIN_VISIBLE_MASK_IOU = 0.05
RECOVERY_MIN_CAD_COVERAGE = 0.05
RECOVERY_MAX_REPROJECTION_CENTER_ERROR_NORM = 0.45
RECOVERY_MAX_RENDERED_DEPTH_MEDIAN_RESIDUAL_M = 0.12
RECOVERY_MIN_RENDERED_DEPTH_SUPPORT = 20
RECOVERY_MIN_TEMPLATE_SCORE_MARGIN = 0.001
RECOVERY_MIN_SOFT_EVIDENCE_PASSES = 2

# Motion hard gates are evaluated against the immediately previous
# operational B5 pose (t-1), NOT the frozen pre-blackout Template2 reference.
RECOVERY_MAX_TRANSLATION_JUMP_M = 0.3
RECOVERY_MAX_ROTATION_JUMP_DEG = 170.0


def build_rgb_template1(
    initial_rgb_real,
    initial_mask,
    bbox_margin_px=8,
):
    """
    Crop rgb_template1 from the first RGB using YCBInEOAT init_mask.png.
    Pixels outside the template mask are zeroed.
    """
    if initial_rgb_real is None or initial_mask is None:
        return None, None, None

    rgb = np.asarray(initial_rgb_real)
    mask = np.asarray(initial_mask)

    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=2)
    else:
        mask = mask > 0

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"[Recovery][Template1] invalid initial RGB shape: {rgb.shape}"
        )

    if mask.shape[:2] != rgb.shape[:2]:
        raise ValueError(
            "[Recovery][Template1] initial RGB/mask shape mismatch: "
            f"rgb={rgb.shape}, mask={mask.shape}"
        )

    ys, xs = np.where(mask)
    if len(xs) < 20:
        raise ValueError(
            "[Recovery][Template1] initial_mask foreground too small: "
            f"{len(xs)} pixels"
        )

    h, w = mask.shape
    x1 = max(0, int(xs.min()) - int(bbox_margin_px))
    y1 = max(0, int(ys.min()) - int(bbox_margin_px))
    x2 = min(w, int(xs.max()) + 1 + int(bbox_margin_px))
    y2 = min(h, int(ys.max()) + 1 + int(bbox_margin_px))

    rgb_template1 = rgb[y1:y2, x1:x2].copy()
    template1_mask = mask[y1:y2, x1:x2].astype(np.uint8)
    rgb_template1[template1_mask == 0] = 0

    return (
        rgb_template1,
        template1_mask,
        (x1, y1, x2, y2),
    )


def _load_init_mask_file(init_mask_path):
    """Load official YCBInEOAT init_mask.png as a boolean object mask."""
    if init_mask_path is None:
        return None

    path = Path(init_mask_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"[Recovery][Template1] init_mask.png not found: {path}"
        )

    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(
            f"[Recovery][Template1] failed to read init mask: {path}"
        )

    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=2)
    else:
        mask = mask > 0

    if np.count_nonzero(mask) < 20:
        raise ValueError(
            f"[Recovery][Template1] init mask foreground too small: "
            f"{np.count_nonzero(mask)} pixels, file={path}"
        )

    return mask.astype(bool)


def _resolve_init_mask_path(
    init_mask_path=None,
    base_sequence=None,
    ycbineoat_root="./datasets/YCBInEOAT",
):
    """
    Preferred:
        init_mask_path="./datasets/YCBInEOAT/mustard0/init_mask.png"

    Convenience:
        base_sequence="mustard0"
        -> ./datasets/YCBInEOAT/mustard0/init_mask.png
    """
    if init_mask_path is not None:
        return os.path.abspath(str(init_mask_path))

    if base_sequence is None:
        return None

    return os.path.abspath(
        os.path.join(
            str(ycbineoat_root),
            str(base_sequence),
            "init_mask.png",
        )
    )


def _ensure_template1_cached(
    state,
    current_rgb_real=None,
    initial_rgb_real=None,
    initial_mask=None,
    init_mask_path=None,
):
    """
    Build rgb_template1 exactly once:
        first RGB + official init_mask.png -> rgb_template1.
    """
    if state.get("rgb_template1") is not None:
        return state

    if state.get("initial_rgb_real") is None:
        src_rgb = (
            initial_rgb_real
            if initial_rgb_real is not None
            else current_rgb_real
        )
        if src_rgb is not None:
            state["initial_rgb_real"] = np.asarray(
                src_rgb,
                dtype=np.uint8,
            ).copy()

    if state.get("initial_mask") is None:
        if initial_mask is not None:
            mask = np.asarray(initial_mask)
            if mask.ndim == 3:
                mask = np.any(mask > 0, axis=2)
            else:
                mask = mask > 0
            state["initial_mask"] = mask.astype(bool)
        elif init_mask_path is not None:
            state["initial_mask"] = _load_init_mask_file(
                init_mask_path
            )
            state["init_mask_path"] = os.path.abspath(
                str(init_mask_path)
            )

    if (
        state.get("initial_rgb_real") is None
        or state.get("initial_mask") is None
    ):
        return state

    (
        rgb_template1,
        template1_mask,
        template1_bbox,
    ) = build_rgb_template1(
        initial_rgb_real=state["initial_rgb_real"],
        initial_mask=state["initial_mask"],
    )

    state["rgb_template1"] = np.asarray(
        rgb_template1,
        dtype=np.uint8,
    ).copy()
    state["template1_mask"] = np.asarray(
        template1_mask,
        dtype=np.uint8,
    ).copy()
    state["template1_bbox_xyxy"] = np.asarray(
        template1_bbox,
        dtype=np.int64,
    )

    return state


def _rotate_template(
    rgb_template,
    mask_template,
    angle_deg,
):
    h, w = mask_template.shape
    center = (
        (w - 1) * 0.5,
        (h - 1) * 0.5,
    )

    M = cv2.getRotationMatrix2D(
        center,
        float(angle_deg),
        1.0,
    )

    rgb_rot = cv2.warpAffine(
        rgb_template,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    mask_rot = cv2.warpAffine(
        mask_template.astype(np.uint8),
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return (
        rgb_rot,
        (mask_rot > 0).astype(np.uint8),
    )


def _clip_bbox_xyxy(
    bbox_xyxy,
    image_shape,
):
    h, w = image_shape[:2]

    x1, y1, x2, y2 = [
        int(round(v))
        for v in bbox_xyxy
    ]

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    return x1, y1, x2, y2


def _bbox_from_binary_mask(
    mask,
    margin_px=0,
):
    mask_bool = np.asarray(mask) > 0
    ys, xs = np.where(mask_bool)

    if len(xs) < 1:
        return None

    h, w = mask_bool.shape[:2]

    x1 = max(
        0,
        int(xs.min()) - int(margin_px),
    )
    y1 = max(
        0,
        int(ys.min()) - int(margin_px),
    )
    x2 = min(
        w,
        int(xs.max()) + 1 + int(margin_px),
    )
    y2 = min(
        h,
        int(ys.max()) + 1 + int(margin_px),
    )

    return x1, y1, x2, y2


def _project_cad_bbox(
    T_pose,
    model_pts_3d,
    K,
    image_shape,
):
    """
    Project CAD points with T_pose and return the visible projected bbox.

    This does not use depth or GT. It is only a spatial guide for finding
    Template1 on the pre-blackout/reference RGB.
    """
    T_pose = np.asarray(
        T_pose,
        dtype=np.float64,
    ).reshape(4, 4)

    pts = np.asarray(
        model_pts_3d,
        dtype=np.float64,
    ).reshape(-1, 3)

    K_arr = np.asarray(
        K,
        dtype=np.float64,
    ).reshape(3, 3)

    pts_cam = (
        T_pose[:3, :3]
        @ pts.T
    ).T + T_pose[:3, 3]

    z = pts_cam[:, 2]
    valid = (
        np.isfinite(z)
        & (z > 1e-6)
    )

    if np.count_nonzero(valid) < 20:
        return None

    pts_cam = pts_cam[valid]
    z = pts_cam[:, 2]

    u = (
        K_arr[0, 0]
        * pts_cam[:, 0]
        / z
        + K_arr[0, 2]
    )

    v = (
        K_arr[1, 1]
        * pts_cam[:, 1]
        / z
        + K_arr[1, 2]
    )

    h, w = image_shape[:2]

    inside = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (u < w)
        & (v >= 0)
        & (v < h)
    )

    if np.count_nonzero(inside) < 20:
        return None

    u = u[inside]
    v = v[inside]

    return _clip_bbox_xyxy(
        (
            np.floor(u.min()),
            np.floor(v.min()),
            np.ceil(u.max()) + 1,
            np.ceil(v.max()) + 1,
        ),
        image_shape,
    )


def _make_padded_search_bbox(
    object_bbox,
    image_shape,
    padding_factor=REFERENCE_ROI_PADDING_FACTOR,
    min_padding_px=REFERENCE_ROI_MIN_PADDING_PX,
):
    x1, y1, x2, y2 = object_bbox

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    pad = max(
        int(min_padding_px),
        int(
            round(
                float(padding_factor)
                * max(bw, bh)
            )
        ),
    )

    return _clip_bbox_xyxy(
        (
            x1 - pad,
            y1 - pad,
            x2 + pad,
            y2 + pad,
        ),
        image_shape,
    )


def _masked_multiscale_template_match(
    current_rgb_real,
    rgb_template,
    template_mask,
    scales,
    angles_deg,
    min_match_score,
    search_bbox=None,
):
    """
    Masked multiscale/rotation template matching.

    Returns a full-frame matched binary mask plus diagnostics.
    """
    if (
        current_rgb_real is None
        or rgb_template is None
        or template_mask is None
    ):
        return None, {
            "match_success": False,
            "match_score": np.nan,
            "match_reason": "missing_input",
        }

    current_rgb = np.asarray(
        current_rgb_real,
        dtype=np.uint8,
    )

    template_rgb = np.asarray(
        rgb_template,
        dtype=np.uint8,
    )

    template_mask = (
        np.asarray(template_mask) > 0
    ).astype(np.uint8)

    if np.count_nonzero(template_mask) < 20:
        return None, {
            "match_success": False,
            "match_score": np.nan,
            "match_reason": "template_mask_too_small",
        }

    h_img, w_img = current_rgb.shape[:2]

    if search_bbox is None:
        sx1, sy1, sx2, sy2 = (
            0,
            0,
            w_img,
            h_img,
        )
    else:
        sx1, sy1, sx2, sy2 = _clip_bbox_xyxy(
            search_bbox,
            current_rgb.shape,
        )

    search_rgb = current_rgb[
        sy1:sy2,
        sx1:sx2,
    ]

    if search_rgb.size == 0:
        return None, {
            "match_success": False,
            "match_score": np.nan,
            "match_reason": "empty_search_roi",
        }

    # a/b channels are less sensitive than RGB intensity to illumination.
    search_lab = cv2.cvtColor(
        search_rgb,
        cv2.COLOR_RGB2LAB,
    )
    search_feature = (
        search_lab[:, :, 1:3]
        .astype(np.float32)
    )

    best = None

    for scale in scales:
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0:
            continue

        new_w = max(
            8,
            int(
                round(
                    template_rgb.shape[1]
                    * scale
                )
            ),
        )
        new_h = max(
            8,
            int(
                round(
                    template_rgb.shape[0]
                    * scale
                )
            ),
        )

        if (
            new_w >= search_rgb.shape[1]
            or new_h >= search_rgb.shape[0]
        ):
            continue

        scaled_rgb = cv2.resize(
            template_rgb,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

        scaled_mask = cv2.resize(
            template_mask,
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST,
        )

        scaled_mask = (
            scaled_mask > 0
        ).astype(np.uint8)

        if np.count_nonzero(
            scaled_mask
        ) < 20:
            continue

        for angle_deg in angles_deg:
            (
                rot_rgb,
                rot_mask,
            ) = _rotate_template(
                scaled_rgb,
                scaled_mask,
                angle_deg,
            )

            if np.count_nonzero(
                rot_mask
            ) < 20:
                continue

            rot_lab = cv2.cvtColor(
                rot_rgb.astype(np.uint8),
                cv2.COLOR_RGB2LAB,
            )

            rot_feature = (
                rot_lab[:, :, 1:3]
                .astype(np.float32)
            )

            mask_2ch = np.repeat(
                rot_mask[:, :, None]
                .astype(np.float32),
                2,
                axis=2,
            )

            try:
                response = cv2.matchTemplate(
                    search_feature,
                    rot_feature,
                    cv2.TM_CCORR_NORMED,
                    mask=mask_2ch,
                )
            except cv2.error:
                response = cv2.matchTemplate(
                    search_feature,
                    rot_feature,
                    cv2.TM_CCORR_NORMED,
                )

            response = np.nan_to_num(
                response,
                nan=-1.0,
                posinf=-1.0,
                neginf=-1.0,
            )

            _, max_val, _, max_loc = (
                cv2.minMaxLoc(response)
            )

            candidate = {
                "score": float(max_val),
                "loc": (
                    int(max_loc[0]),
                    int(max_loc[1]),
                ),
                "scale": scale,
                "angle_deg":
                    float(angle_deg),
                "mask":
                    rot_mask.copy(),
            }

            if (
                best is None
                or candidate["score"]
                > best["score"]
            ):
                best = candidate

    if best is None:
        return None, {
            "match_success": False,
            "match_score": np.nan,
            "match_reason":
                "no_valid_hypothesis",
        }

    th, tw = best["mask"].shape

    x1 = (
        sx1
        + best["loc"][0]
    )
    y1 = (
        sy1
        + best["loc"][1]
    )
    x2 = x1 + tw
    y2 = y1 + th

    if (
        x1 < 0
        or y1 < 0
        or x2 > w_img
        or y2 > h_img
    ):
        return None, {
            "match_success": False,
            "match_score":
                float(best["score"]),
            "match_reason":
                "matched_bbox_outside_image",
        }

    full_mask = np.zeros(
        (h_img, w_img),
        dtype=np.uint8,
    )

    full_mask[
        y1:y2,
        x1:x2,
    ] = (
        best["mask"] > 0
    ).astype(np.uint8)

    full_mask = (
        full_mask > 0
    ).astype(np.uint8)

    success = bool(
        best["score"]
        >= float(min_match_score)
        and np.count_nonzero(
            full_mask
        ) >= 20
    )

    diagnostics = {
        "match_success":
            success,
        "match_score":
            float(best["score"]),
        "match_scale":
            float(best["scale"]),
        "match_angle_deg":
            float(best["angle_deg"]),
        "match_bbox_xyxy":
            np.asarray(
                [x1, y1, x2, y2],
                dtype=np.int64,
            ),
        "search_bbox_xyxy":
            np.asarray(
                [sx1, sy1, sx2, sy2],
                dtype=np.int64,
            ),
        "match_reason":
            (
                "ok"
                if success
                else "score_below_threshold"
            ),
    }

    return (
        full_mask.astype(bool)
        if success
        else None,
        diagnostics,
    )


def build_rgb_template2(
    reference_rgb_real,
    reference_T_final,
    model_pts_3d,
    K,
    rgb_template1,
    template1_mask,
):
    """
    Build rgb_template2 on the pre-blackout/reference frame.

    reference_T_final + CAD determines a pose-guided search ROI.
    rgb_template1 is matched only inside that ROI. The matched object crop
    from the reference RGB becomes rgb_template2.
    """
    if (
        reference_rgb_real is None
        or reference_T_final is None
    ):
        return None, None, {
            "template1_guided_match_success":
                False,
            "template1_guided_match_reason":
                "missing_reference_rgb_or_pose",
        }

    reference_rgb = np.asarray(
        reference_rgb_real,
        dtype=np.uint8,
    )

    cad_bbox = _project_cad_bbox(
        T_pose=reference_T_final,
        model_pts_3d=model_pts_3d,
        K=K,
        image_shape=reference_rgb.shape,
    )

    if cad_bbox is None:
        return None, None, {
            "template1_guided_match_success":
                False,
            "template1_guided_match_reason":
                "cad_projection_failed",
        }

    search_bbox = _make_padded_search_bbox(
        cad_bbox,
        reference_rgb.shape,
    )

    cad_w = max(
        1,
        cad_bbox[2] - cad_bbox[0],
    )
    cad_h = max(
        1,
        cad_bbox[3] - cad_bbox[1],
    )

    template_h, template_w = (
        template1_mask.shape
    )

    base_scale = np.sqrt(
        (
            float(cad_w)
            * float(cad_h)
        )
        / max(
            float(template_w)
            * float(template_h),
            1.0,
        )
    )

    scale_candidates = [
        float(
            np.clip(
                base_scale
                * multiplier,
                0.25,
                4.0,
            )
        )
        for multiplier
        in TEMPLATE1_SCALE_MULTIPLIERS
    ]

    (
        matched_mask,
        match_diag,
    ) = _masked_multiscale_template_match(
        current_rgb_real=reference_rgb,
        rgb_template=rgb_template1,
        template_mask=template1_mask,
        scales=scale_candidates,
        angles_deg=(
            TEMPLATE1_ANGLE_SET_DEG
        ),
        min_match_score=(
            TEMPLATE1_MIN_MATCH_SCORE
        ),
        search_bbox=search_bbox,
    )

    diagnostics = {
        "template1_guided_match_success":
            bool(
                match_diag.get(
                    "match_success",
                    False,
                )
            ),
        "template1_guided_match_score":
            float(
                match_diag.get(
                    "match_score",
                    np.nan,
                )
            ),
        "template1_guided_match_scale":
            match_diag.get(
                "match_scale"
            ),
        "template1_guided_match_angle_deg":
            match_diag.get(
                "match_angle_deg"
            ),
        "template1_guided_match_bbox_xyxy":
            match_diag.get(
                "match_bbox_xyxy"
            ),
        "template1_guided_search_bbox_xyxy":
            np.asarray(
                search_bbox,
                dtype=np.int64,
            ),
        "reference_cad_bbox_xyxy":
            np.asarray(
                cad_bbox,
                dtype=np.int64,
            ),
        "template1_guided_match_reason":
            match_diag.get(
                "match_reason"
            ),
    }

    if matched_mask is None:
        return (
            None,
            None,
            diagnostics,
        )

    template2_bbox = _bbox_from_binary_mask(
        matched_mask,
        margin_px=(
            TEMPLATE2_CROP_MARGIN_PX
        ),
    )

    if template2_bbox is None:
        diagnostics[
            "template1_guided_match_success"
        ] = False
        diagnostics[
            "template1_guided_match_reason"
        ] = "matched_mask_empty"
        return (
            None,
            None,
            diagnostics,
        )

    x1, y1, x2, y2 = (
        template2_bbox
    )

    rgb_template2 = (
        reference_rgb[
            y1:y2,
            x1:x2,
        ]
        .copy()
    )

    template2_mask = (
        np.asarray(
            matched_mask[
                y1:y2,
                x1:x2,
            ]
        ) > 0
    ).astype(np.uint8)

    if (
        rgb_template2.size == 0
        or np.count_nonzero(
            template2_mask
        ) < 20
    ):
        diagnostics[
            "template1_guided_match_success"
        ] = False
        diagnostics[
            "template1_guided_match_reason"
        ] = "template2_crop_invalid"
        return (
            None,
            None,
            diagnostics,
        )

    rgb_template2[
        template2_mask == 0
    ] = 0

    diagnostics[
        "template2_bbox_xyxy"
    ] = np.asarray(
        template2_bbox,
        dtype=np.int64,
    )

    return (
        rgb_template2,
        template2_mask,
        diagnostics,
    )



def _bbox_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _masked_multiscale_template_candidates(
    current_rgb_real,
    rgb_template,
    template_mask,
    scales,
    angles_deg,
    search_bbox=None,
    top_k=TEMPLATE2_TOP_K,
    peaks_per_hypothesis=TEMPLATE2_PEAKS_PER_HYPOTHESIS,
    nms_iou=TEMPLATE2_CANDIDATE_NMS_IOU,
):
    """Return diverse high-scoring template hypotheses instead of only Top-1."""
    if current_rgb_real is None or rgb_template is None or template_mask is None:
        return []

    current_rgb = np.asarray(current_rgb_real, dtype=np.uint8)
    template_rgb = np.asarray(rgb_template, dtype=np.uint8)
    template_mask = (np.asarray(template_mask) > 0).astype(np.uint8)
    if np.count_nonzero(template_mask) < 20:
        return []

    h_img, w_img = current_rgb.shape[:2]
    if search_bbox is None:
        sx1, sy1, sx2, sy2 = 0, 0, w_img, h_img
    else:
        sx1, sy1, sx2, sy2 = _clip_bbox_xyxy(search_bbox, current_rgb.shape)
    search_rgb = current_rgb[sy1:sy2, sx1:sx2]
    if search_rgb.size == 0:
        return []

    search_lab = cv2.cvtColor(search_rgb, cv2.COLOR_RGB2LAB)
    search_feature = search_lab[:, :, 1:3].astype(np.float32)
    candidates = []

    for scale in scales:
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0:
            continue
        new_w = max(8, int(round(template_rgb.shape[1] * scale)))
        new_h = max(8, int(round(template_rgb.shape[0] * scale)))
        if new_w >= search_rgb.shape[1] or new_h >= search_rgb.shape[0]:
            continue
        scaled_rgb = cv2.resize(template_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        scaled_mask = cv2.resize(template_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        scaled_mask = (scaled_mask > 0).astype(np.uint8)
        if np.count_nonzero(scaled_mask) < 20:
            continue

        for angle_deg in angles_deg:
            rot_rgb, rot_mask = _rotate_template(scaled_rgb, scaled_mask, angle_deg)
            if np.count_nonzero(rot_mask) < 20:
                continue
            rot_lab = cv2.cvtColor(rot_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)
            rot_feature = rot_lab[:, :, 1:3].astype(np.float32)
            mask_2ch = np.repeat(rot_mask[:, :, None].astype(np.float32), 2, axis=2)
            try:
                response = cv2.matchTemplate(
                    search_feature, rot_feature, cv2.TM_CCORR_NORMED, mask=mask_2ch
                )
            except cv2.error:
                response = cv2.matchTemplate(search_feature, rot_feature, cv2.TM_CCORR_NORMED)
            response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
            work = response.copy()
            th, tw = rot_mask.shape
            suppress_x = max(2, tw // 3)
            suppress_y = max(2, th // 3)
            for _ in range(int(peaks_per_hypothesis)):
                _, max_val, _, max_loc = cv2.minMaxLoc(work)
                if not np.isfinite(max_val) or max_val < -0.5:
                    break
                lx, ly = int(max_loc[0]), int(max_loc[1])
                x1, y1 = sx1 + lx, sy1 + ly
                x2, y2 = x1 + tw, y1 + th
                if x1 >= 0 and y1 >= 0 and x2 <= w_img and y2 <= h_img:
                    candidates.append({
                        "score": float(max_val),
                        "loc": (lx, ly),
                        "bbox_xyxy": (x1, y1, x2, y2),
                        "scale": scale,
                        "angle_deg": float(angle_deg),
                        "mask": rot_mask.copy(),
                        "search_bbox_xyxy": (sx1, sy1, sx2, sy2),
                    })
                rx1 = max(0, lx - suppress_x)
                ry1 = max(0, ly - suppress_y)
                rx2 = min(work.shape[1], lx + suppress_x + 1)
                ry2 = min(work.shape[0], ly + suppress_y + 1)
                work[ry1:ry2, rx1:rx2] = -1.0

    candidates.sort(key=lambda c: c["score"], reverse=True)
    kept = []
    for cand in candidates:
        if all(_bbox_iou_xyxy(cand["bbox_xyxy"], k["bbox_xyxy"]) < float(nms_iou) for k in kept):
            kept.append(cand)
        if len(kept) >= int(top_k):
            break
    return kept


def _candidate_full_mask(candidate, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in candidate["bbox_xyxy"]]
    mask = np.zeros((h, w), dtype=np.uint8)
    local = (np.asarray(candidate["mask"]) > 0).astype(np.uint8)
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h or local.shape != (y2-y1, x2-x1):
        return None
    mask[y1:y2, x1:x2] = local
    return mask.astype(bool)


def _depth_candidate_diagnostics(mask, depth_real, expected_depth_m=None):
    out = {
        "valid_depth_ratio": 0.0,
        "median_depth_m": np.nan,
        "depth_iqr_m": np.nan,
        "depth_delta_from_reference_m": np.nan,
        "depth_gate_pass": False,
    }
    if mask is None or depth_real is None:
        return out
    mask = np.asarray(mask) > 0
    depth = np.asarray(depth_real, dtype=np.float32)
    if depth.shape[:2] != mask.shape[:2] or np.count_nonzero(mask) == 0:
        return out
    valid = (
        mask & np.isfinite(depth)
        & (depth > DEPTH_BLACKOUT_VALID_MIN_M)
        & (depth < DEPTH_BLACKOUT_VALID_MAX_M)
    )
    n_mask = int(np.count_nonzero(mask))
    n_valid = int(np.count_nonzero(valid))
    out["valid_depth_ratio"] = n_valid / float(max(n_mask, 1))
    if n_valid == 0:
        return out
    vals = depth[valid]
    med = float(np.median(vals))
    q25, q75 = np.percentile(vals, [25.0, 75.0])
    out["median_depth_m"] = med
    out["depth_iqr_m"] = float(q75 - q25)
    depth_ok = out["valid_depth_ratio"] >= RECOVERY_CANDIDATE_MIN_VALID_DEPTH_RATIO
    if expected_depth_m is not None and np.isfinite(expected_depth_m):
        delta = abs(med - float(expected_depth_m))
        out["depth_delta_from_reference_m"] = float(delta)
        depth_ok = depth_ok and delta <= RECOVERY_CANDIDATE_MAX_DEPTH_DELTA_M
    out["depth_gate_pass"] = bool(depth_ok)
    return out

def match_rgb_template2_and_pad(
    current_rgb_real,
    rgb_template2,
    template2_mask,
    current_depth_real=None,
    reference_T_final=None,
    padding_px=FOUNDATIONPOSE_MASK_PADDING_PX,
):
    """
    Recovery-frame localization with Top-K template hypotheses and a depth gate.

    The visual matcher proposes diverse candidates. Depth is used only as a
    hard plausibility filter; no hand-tuned weighted score is formed.
    """
    candidates = _masked_multiscale_template_candidates(
        current_rgb_real=current_rgb_real,
        rgb_template=rgb_template2,
        template_mask=template2_mask,
        scales=TEMPLATE2_SCALES,
        angles_deg=TEMPLATE2_ANGLES_DEG,
        search_bbox=None,
        top_k=TEMPLATE2_TOP_K,
    )

    diagnostics = {
        "template2_match_success": False,
        "template2_match_score": np.nan,
        "template2_second_score": np.nan,
        "template2_score_margin": np.nan,
        "template2_match_scale": None,
        "template2_match_angle_deg": None,
        "template2_match_bbox_xyxy": None,
        "template2_match_reason": "no_valid_hypothesis",
        "template2_candidate_count": int(len(candidates)),
        "template2_candidates": [],
        "foundationpose_mask_padding_px": int(padding_px),
        "recovery_visible_mask": None,
    }
    if len(candidates) == 0:
        return None, diagnostics

    expected_depth_m = None
    if reference_T_final is not None:
        try:
            expected_depth_m = float(np.asarray(reference_T_final, dtype=np.float64).reshape(4, 4)[2, 3])
        except Exception:
            expected_depth_m = None

    surviving = []
    for rank, cand in enumerate(candidates, start=1):
        full_mask = _candidate_full_mask(cand, np.asarray(current_rgb_real).shape)
        depth_diag = _depth_candidate_diagnostics(
            full_mask,
            current_depth_real,
            expected_depth_m=expected_depth_m,
        )
        row = {
            "rank": int(rank),
            "score": float(cand["score"]),
            "bbox_xyxy": [int(v) for v in cand["bbox_xyxy"]],
            "scale": float(cand["scale"]),
            "angle_deg": float(cand["angle_deg"]),
            **depth_diag,
        }
        diagnostics["template2_candidates"].append(row)
        if full_mask is None:
            continue
        if cand["score"] < float(TEMPLATE2_MIN_MATCH_SCORE):
            continue
        if current_depth_real is not None and not depth_diag["depth_gate_pass"]:
            continue
        surviving.append((cand, full_mask, depth_diag))

    if len(surviving) == 0:
        diagnostics["template2_match_reason"] = "all_topk_candidates_rejected_by_score_or_depth"
        return None, diagnostics

    surviving.sort(key=lambda item: item[0]["score"], reverse=True)
    best, visible_mask, best_depth_diag = surviving[0]
    second_score = float(surviving[1][0]["score"]) if len(surviving) > 1 else np.nan
    score_margin = (
        float(best["score"] - second_score)
        if np.isfinite(second_score)
        else np.inf
    )

    diagnostics.update({
        "template2_match_success": True,
        "template2_match_score": float(best["score"]),
        "template2_second_score": second_score,
        "template2_score_margin": float(score_margin),
        "template2_match_scale": float(best["scale"]),
        "template2_match_angle_deg": float(best["angle_deg"]),
        "template2_match_bbox_xyxy": np.asarray(best["bbox_xyxy"], dtype=np.int64),
        "template2_match_reason": "ok",
        "template2_selected_valid_depth_ratio": float(best_depth_diag["valid_depth_ratio"]),
        "template2_selected_median_depth_m": float(best_depth_diag["median_depth_m"]),
        "template2_selected_depth_delta_from_reference_m": float(best_depth_diag["depth_delta_from_reference_m"]),
        "recovery_visible_mask": np.asarray(visible_mask, dtype=np.uint8),
    })

    mask_u8 = np.asarray(visible_mask, dtype=np.uint8)
    if int(padding_px) > 0:
        kernel_size = 2 * int(padding_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
    mask_u8 = (mask_u8 > 0).astype(np.uint8)
    if np.count_nonzero(mask_u8) < 20:
        diagnostics["template2_match_success"] = False
        diagnostics["template2_match_reason"] = "padded_mask_too_small"
        return None, diagnostics
    return mask_u8.astype(bool), diagnostics


def _project_cad_silhouette_mask(
    T_pose,
    model_pts_3d,
    K,
    image_shape,
):
    """
    Project CAD points with T_pose and rasterize a compact 2-D silhouette.

    A convex hull is used instead of sparse projected points so the overlap
    metric measures object-region consistency rather than point density.
    """
    h, w = image_shape[:2]

    T = np.asarray(
        T_pose,
        dtype=np.float64,
    ).reshape(4, 4)
    pts = np.asarray(
        model_pts_3d,
        dtype=np.float64,
    ).reshape(-1, 3)
    K_arr = np.asarray(
        K,
        dtype=np.float64,
    ).reshape(3, 3)

    if pts.shape[0] < 3:
        return None

    pts_cam = (
        T[:3, :3] @ pts.T
    ).T + T[:3, 3]

    z = pts_cam[:, 2]
    valid = (
        np.isfinite(pts_cam).all(axis=1)
        & np.isfinite(z)
        & (z > 1e-6)
    )

    if np.count_nonzero(valid) < 3:
        return None

    pts_cam = pts_cam[valid]
    z = pts_cam[:, 2]

    u = (
        K_arr[0, 0] * pts_cam[:, 0] / z
        + K_arr[0, 2]
    )
    v = (
        K_arr[1, 1] * pts_cam[:, 1] / z
        + K_arr[1, 2]
    )

    inside = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (u < w)
        & (v >= 0)
        & (v < h)
    )

    if np.count_nonzero(inside) < 3:
        return None

    uv = np.stack(
        [u[inside], v[inside]],
        axis=1,
    )
    uv = np.round(uv).astype(np.int32)

    hull = cv2.convexHull(uv)
    if hull is None or len(hull) < 3:
        return None

    cad_mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )
    cv2.fillConvexPoly(
        cad_mask,
        hull,
        1,
    )

    if (
        np.count_nonzero(cad_mask)
        < RECOVERY_CAD_MIN_PROJECTED_PIXELS
    ):
        return None

    return cad_mask.astype(bool)


def evaluate_recovery_cad_mask_consistency(
    T_recovery,
    model_pts_3d,
    K,
    recovery_mask,
):
    """
    Score the independent recovery candidate using only CAD-mask reprojection
    consistency. No GT, T_obs, T_prior, or recursive history is used here.

    score = 0.30 * IoU + 0.70 * CAD coverage

    CAD coverage is weighted more heavily because recovery_mask is deliberately
    padded before FoundationPose registration, which can depress plain IoU even
    when the CAD projection is correctly contained by the mask.
    """
    result = {
        "recovery_cad_mask_iou": 0.0,
        "recovery_cad_coverage": 0.0,
        "recovery_quality_score": 0.0,
        "recovery_cad_projected_pixels": 0,
        "recovery_mask_pixels": 0,
        "recovery_cad_mask_valid": False,
    }

    if T_recovery is None or recovery_mask is None:
        return result

    rec_mask = (
        np.asarray(recovery_mask) > 0
    )
    result["recovery_mask_pixels"] = int(
        np.count_nonzero(rec_mask)
    )

    if result["recovery_mask_pixels"] < 20:
        return result

    cad_mask = _project_cad_silhouette_mask(
        T_pose=T_recovery,
        model_pts_3d=model_pts_3d,
        K=K,
        image_shape=rec_mask.shape,
    )

    if cad_mask is None:
        return result

    cad_pixels = int(
        np.count_nonzero(cad_mask)
    )
    result["recovery_cad_projected_pixels"] = cad_pixels

    intersection = int(
        np.count_nonzero(
            cad_mask & rec_mask
        )
    )
    union = int(
        np.count_nonzero(
            cad_mask | rec_mask
        )
    )

    iou = (
        intersection / float(union)
        if union > 0
        else 0.0
    )
    coverage = (
        intersection / float(cad_pixels)
        if cad_pixels > 0
        else 0.0
    )

    score = (
        RECOVERY_CAD_MASK_IOU_WEIGHT * iou
        + RECOVERY_CAD_MASK_COVERAGE_WEIGHT
        * coverage
    )
    score = float(
        np.clip(score, 0.0, 1.0)
    )

    result.update({
        "recovery_cad_mask_iou": float(iou),
        "recovery_cad_coverage": float(coverage),
        "recovery_quality_score": score,
        "recovery_cad_mask_valid": True,
    })
    return result



def _rotation_angle_deg(R_rel):
    value = (np.trace(np.asarray(R_rel, dtype=np.float64)) - 1.0) * 0.5
    value = float(np.clip(value, -1.0, 1.0))
    return float(np.degrees(np.arccos(value)))


def _mask_centroid_and_diag(mask):
    mask = np.asarray(mask) > 0
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None
    centroid = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)
    diag = float(np.hypot(float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)))
    return centroid, max(diag, 1.0)


def _project_cad_depth_residual(
    T_pose,
    model_pts_3d,
    K,
    depth_real,
    visible_mask=None,
):
    """Approximate rendered-object depth residual with a CAD-point z-buffer."""
    out = {
        "recovery_rendered_depth_support": 0,
        "recovery_rendered_depth_median_residual_m": np.inf,
        "recovery_rendered_depth_mean_residual_m": np.inf,
    }
    if T_pose is None or depth_real is None:
        return out
    depth = np.asarray(depth_real, dtype=np.float32)
    if depth.ndim != 2:
        return out
    h, w = depth.shape
    T = np.asarray(T_pose, dtype=np.float64).reshape(4, 4)
    pts = np.asarray(model_pts_3d, dtype=np.float64).reshape(-1, 3)
    K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
    pts_cam = (T[:3, :3] @ pts.T).T + T[:3, 3]
    z = pts_cam[:, 2]
    valid = np.isfinite(pts_cam).all(axis=1) & (z > 1e-6)
    if np.count_nonzero(valid) < 3:
        return out
    pts_cam = pts_cam[valid]
    z = pts_cam[:, 2]
    u = np.round(K_arr[0, 0] * pts_cam[:, 0] / z + K_arr[0, 2]).astype(np.int64)
    v = np.round(K_arr[1, 1] * pts_cam[:, 1] / z + K_arr[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if np.count_nonzero(inside) < 3:
        return out
    u, v, z = u[inside], v[inside], z[inside]
    flat = v * w + u
    zbuf = np.full(h * w, np.inf, dtype=np.float64)
    np.minimum.at(zbuf, flat, z)
    projected = np.isfinite(zbuf).reshape(h, w)
    rendered_depth = zbuf.reshape(h, w)
    observed_valid = (
        np.isfinite(depth)
        & (depth > DEPTH_BLACKOUT_VALID_MIN_M)
        & (depth < DEPTH_BLACKOUT_VALID_MAX_M)
    )
    support = projected & observed_valid
    if visible_mask is not None:
        support &= (np.asarray(visible_mask) > 0)
    n = int(np.count_nonzero(support))
    out["recovery_rendered_depth_support"] = n
    if n == 0:
        return out
    residual = np.abs(rendered_depth[support] - depth[support].astype(np.float64))
    out["recovery_rendered_depth_median_residual_m"] = float(np.median(residual))
    out["recovery_rendered_depth_mean_residual_m"] = float(np.mean(residual))
    return out


def evaluate_recovery_pose_validity(
    T_recovery, model_pts_3d, K, visible_mask, depth_real,
    reference_T_final=None, template_score_margin=None,
    previous_T_final=None, T_prior_current=None,
):
    """Current-frame admission; history jumps and old soft cues are diagnostics."""
    from recovery_gate import evaluate_current_frame
    # Validate shape before using the projection helpers.
    silhouette = None
    T = np.asarray(T_recovery) if T_recovery is not None else None
    if (T is not None and T.shape == (4, 4) and np.isfinite(T).all()
            and visible_mask is not None and depth_real is not None
            and np.asarray(visible_mask).ndim == 2):
        try:
            silhouette = _project_cad_silhouette_mask(
                T_pose=T, model_pts_3d=model_pts_3d, K=K,
                image_shape=np.asarray(visible_mask).shape)
        except (ValueError, TypeError, IndexError):
            silhouette = None
    result = evaluate_current_frame(T_recovery, model_pts_3d, K, depth_real,
                                    visible_mask, silhouette)
    # Historical output fields are retained, but no duplicate-evidence voting.
    result.update(recovery_soft_evidence_pass_count=None,
                  recovery_soft_evidence_required=None, recovery_soft_evidence={"gate": dict(result)},
                  recovery_template_score_margin=template_score_margin,
                  recovery_template_score_margin_available=bool(
                      template_score_margin is not None and np.isfinite(template_score_margin)))
    if T is None or T.shape != (4, 4) or not np.isfinite(T).all():
        return result
    for ref, prefix in ((previous_T_final if previous_T_final is not None else reference_T_final,
                         "prev"), (T_prior_current, "prior")):
        if ref is None:
            continue
        ref = np.asarray(ref)
        if ref.shape != (4, 4) or not np.isfinite(ref).all():
            continue
        translation = float(np.linalg.norm(T[:3, 3]-ref[:3, 3]))
        rotation = _rotation_angle_deg(ref[:3, :3].T @ T[:3, :3])
        if prefix == "prev":
            result.update(recovery_prev_translation_jump_m=translation,
                          recovery_prev_rotation_jump_deg=rotation,
                          recovery_translation_jump_m=translation, recovery_rotation_jump_deg=rotation,
                          recovery_motion_reference_source="diagnostic_only_previous_or_legacy")
        else:
            result.update(recovery_prior_translation_innovation_m=translation,
                          recovery_prior_rotation_innovation_deg=rotation)
    if silhouette is not None and visible_mask is not None:
        result.update(evaluate_recovery_cad_mask_consistency(
            T, model_pts_3d, K, visible_mask))
        result.update(_project_cad_depth_residual(T, model_pts_3d, K, depth_real, visible_mask))
        c_cad, _ = _mask_centroid_and_diag(silhouette)
        c_vis, diag = _mask_centroid_and_diag(visible_mask)
        if c_cad is not None and c_vis is not None:
            result["recovery_reprojection_center_error_norm"] = float(
                np.linalg.norm(c_cad-c_vis)/max(diag, 1.0))
    return result

def _save_recovery_debug_artifacts(
    current_rgb_real,
    current_depth_real,
    recovery_mask,
    frame_id,
    recovery_trigger,
    base_sequence=None,
    output_root="./recovery_debug",
    template2_candidates=None,
):
    """
    Save recovery diagnostics:
      1) recovery_mask.png
      2) recovery_mask_overlay.png
      3) recovery_depth_masked.png
      4) recovery_topk_candidates.png (when candidate diagnostics exist)

    A short content hash prevents different blackout episodes with the same
    frame_id from overwriting each other while keeping reruns deterministic.
    """
    if (
        current_rgb_real is None
        or current_depth_real is None
        or recovery_mask is None
    ):
        return None

    rgb = np.asarray(
        current_rgb_real,
        dtype=np.uint8,
    )
    depth = np.asarray(
        current_depth_real,
        dtype=np.float32,
    )
    mask = (
        np.asarray(recovery_mask) > 0
    )

    if (
        rgb.ndim != 3
        or rgb.shape[2] != 3
        or depth.shape[:2] != rgb.shape[:2]
        or mask.shape[:2] != rgb.shape[:2]
    ):
        return None

    digest = hashlib.sha256()
    digest.update(
        np.ascontiguousarray(rgb).tobytes()
    )
    digest.update(
        np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
    )
    short_hash = digest.hexdigest()[:10]

    safe_base = (
        str(base_sequence)
        if base_sequence is not None
        else "unknown_sequence"
    )
    safe_trigger = str(recovery_trigger).replace(
        os.sep,
        "_",
    )

    event_dir = os.path.abspath(
        os.path.join(
            output_root,
            safe_base,
            (
                f"frame_{int(frame_id):06d}_"
                f"{safe_trigger}_{short_hash}"
            ),
        )
    )
    os.makedirs(
        event_dir,
        exist_ok=True,
    )

    mask_u8 = mask.astype(np.uint8) * 255
    cv2.imwrite(
        os.path.join(
            event_dir,
            "recovery_mask.png",
        ),
        mask_u8,
    )

    rgb_bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )
    overlay = rgb_bgr.copy()
    overlay_mask = overlay[mask]
    if overlay_mask.size > 0:
        red = np.zeros_like(overlay_mask)
        red[:, 2] = 255
        overlay[mask] = (
            0.55 * overlay_mask.astype(np.float32)
            + 0.45 * red.astype(np.float32)
        ).astype(np.uint8)

    cv2.imwrite(
        os.path.join(
            event_dir,
            "recovery_mask_overlay.png",
        ),
        overlay,
    )

    depth_vis = np.zeros(
        depth.shape[:2],
        dtype=np.uint8,
    )
    valid_depth = (
        mask
        & np.isfinite(depth)
        & (depth > DEPTH_BLACKOUT_VALID_MIN_M)
        & (depth < DEPTH_BLACKOUT_VALID_MAX_M)
    )

    if np.count_nonzero(valid_depth) > 0:
        values = depth[valid_depth]
        z_lo = float(np.percentile(values, 2.0))
        z_hi = float(np.percentile(values, 98.0))
        if z_hi <= z_lo + 1e-6:
            depth_vis[valid_depth] = 255
        else:
            normalized = (
                (depth[valid_depth] - z_lo)
                / (z_hi - z_lo)
            )
            normalized = np.clip(
                normalized,
                0.0,
                1.0,
            )
            depth_vis[valid_depth] = (
                1.0 + 254.0 * normalized
            ).astype(np.uint8)

    cv2.imwrite(
        os.path.join(
            event_dir,
            "recovery_depth_masked.png",
        ),
        depth_vis,
    )

    if template2_candidates:
        candidate_vis = rgb_bgr.copy()
        for row in template2_candidates:
            bbox = row.get("bbox_xyxy")
            if bbox is None or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            rank = int(row.get("rank", 0))
            score = float(row.get("score", np.nan))
            depth_ok = bool(row.get("depth_gate_pass", False))
            thickness = 2 if depth_ok else 1
            cv2.rectangle(candidate_vis, (x1, y1), (x2, y2), (255, 255, 255), thickness)
            cv2.putText(
                candidate_vis,
                f"#{rank} s={score:.3f} depth={'Y' if depth_ok else 'N'}",
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(
            os.path.join(event_dir, "recovery_topk_candidates.png"),
            candidate_vis,
        )

    return event_dir


_FOUNDATIONPOSE_REGISTER_RUNNER = r"""
import argparse
import gc
import json
import os
import sys

import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

parser = argparse.ArgumentParser()
parser.add_argument("--foundationpose_dir", required=True)
parser.add_argument("--rgb_npy", required=True)
parser.add_argument("--depth_npy", required=True)
parser.add_argument("--mask_npy", required=True)
parser.add_argument("--K_txt", required=True)
parser.add_argument("--mesh_file", required=True)
parser.add_argument("--output_pose", required=True)
parser.add_argument("--output_diagnostics", required=True)
parser.add_argument("--refiner_weight", required=True)
parser.add_argument("--iteration", type=int, default=10)
args = parser.parse_args()

fp_dir = os.path.abspath(args.foundationpose_dir)
os.chdir(fp_dir)
sys.path.insert(0, fp_dir)

from estimater import FoundationPose
from learning.training.predict_pose_refine import PoseRefinePredictor
from learning.training.predict_score import ScorePredictor

rgb = np.load(args.rgb_npy).astype(np.uint8)
depth = np.load(args.depth_npy).astype(np.float32)
ob_mask = np.load(args.mask_npy).astype(bool)
K = np.loadtxt(args.K_txt).reshape(3, 3).astype(np.float64)

mesh = trimesh.load(
    args.mesh_file,
    process=False,
)

if hasattr(mesh, "geometry"):
    geometries = [
        g for g in mesh.geometry.values()
    ]
    if len(geometries) == 0:
        raise RuntimeError("Mesh scene contains no geometry.")
    mesh = trimesh.util.concatenate(geometries)

if len(mesh.vertices) == 0:
    raise RuntimeError("Mesh has no vertices.")

try:
    mesh.fix_normals()
except Exception:
    pass

scorer = ScorePredictor()
refiner = PoseRefinePredictor()

refiner_weight = os.path.abspath(
    args.refiner_weight
)
official_refiner = os.path.abspath(
    os.path.join(
        fp_dir,
        "weights",
        "2023-10-28-18-33-37",
        "model_best.pth",
    )
)

if refiner_weight != official_refiner:
    ckpt = torch.load(
        refiner_weight,
        map_location="cuda",
    )
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    refiner.model.load_state_dict(ckpt)
    refiner.model.cuda().eval()

glctx = dr.RasterizeCudaContext()

debug_dir = os.path.join(os.path.dirname(args.output_pose),'debug')

est = FoundationPose(
    model_pts=np.asarray(
        mesh.vertices,
        dtype=np.float64,
    ),
    model_normals=np.asarray(
        mesh.vertex_normals,
        dtype=np.float64,
    ),
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    glctx=glctx,
    debug=0,
    debug_dir=debug_dir
)

pose = est.register(
    K=K,
    rgb=rgb,
    depth=depth,
    ob_mask=ob_mask,
    iteration=int(args.iteration),
)

pose = np.asarray(
    pose,
    dtype=np.float64,
).reshape(4, 4)

np.savetxt(
    args.output_pose,
    pose,
)

scores = est.scores
if torch.is_tensor(scores):
    scores = scores.detach().float().cpu().numpy()
scores = np.asarray(scores, dtype=np.float64).reshape(-1)
scores = scores[np.isfinite(scores)]
scores = np.sort(scores)[::-1]
score_top1 = float(scores[0]) if len(scores) >= 1 else None
score_top2 = float(scores[1]) if len(scores) >= 2 else None
score_margin = float(scores[0] - scores[1]) if len(scores) >= 2 else None
with open(args.output_diagnostics, "w", encoding="utf-8") as stream:
    json.dump({
        "foundationpose_score_top1": score_top1,
        "foundationpose_score_top2": score_top2,
        "foundationpose_score_margin": score_margin,
        "foundationpose_score_count": int(len(scores)),
    }, stream, indent=2, allow_nan=False)

del est, scorer, refiner
gc.collect()
torch.cuda.empty_cache()
"""


def foundationpose_register_from_mask(
    current_rgb_real,
    current_depth_real,
    recovery_mask,
    K,
    mesh_file,
    foundationpose_python=None,
    foundationpose_dir=None,
    foundationpose_refiner_weight=None,
    refine_iter=DEFAULT_FOUNDATIONPOSE_REFINE_ITER):
    """
    Run full FoundationPose register() from the current RGB-D and the
    template-generated recovery mask. No recursive pose initializes FoundationPose.
    """
    foundationpose_python = (
        foundationpose_python
        or DEFAULT_FOUNDATIONPOSE_PYTHON
    )

    foundationpose_dir = (
        foundationpose_dir
        or DEFAULT_FOUNDATIONPOSE_DIR
    )

    fp_python = Path(
        foundationpose_python
    )
    fp_dir = Path(
        foundationpose_dir
    )
    mesh_path = Path(
        mesh_file
    )

    if foundationpose_refiner_weight is None:
        foundationpose_refiner_weight = (
            fp_dir
            / "weights"
            / "2023-10-28-18-33-37"
            / "model_best.pth"
        )

    weight_path = Path(
        foundationpose_refiner_weight
    )

    scorer_weight = (
        fp_dir
        / "weights"
        / "2024-01-11-20-02-45"
        / "model_best.pth"
    )

    checks = [
        (
            fp_python.is_file(),
            f"Python not found: {fp_python}",
        ),
        (
            fp_dir.is_dir(),
            f"repo not found: {fp_dir}",
        ),
        (
            mesh_path.is_file(),
            f"mesh not found: {mesh_path}",
        ),
        (
            weight_path.is_file(),
            f"refiner weight not found: {weight_path}",
        ),
        (
            scorer_weight.is_file(),
            f"scorer weight not found: {scorer_weight}",
        ),
    ]

    for ok, message in checks:
        if not ok:
            print(
                "[Recovery][FoundationPose] "
                + message
            )
            return None, False, {
                "foundationpose_error":
                    message,
            }

    if (
        current_rgb_real is None
        or current_depth_real is None
        or recovery_mask is None
    ):
        return None, False, {
            "foundationpose_error":
                "missing_rgb_depth_or_mask",
        }

    rgb = np.asarray(
        current_rgb_real,
        dtype=np.uint8,
    )

    depth = np.asarray(
        current_depth_real,
        dtype=np.float32,
    )

    mask = (
        np.asarray(
            recovery_mask
        ) > 0
    )

    if (
        rgb.shape[:2]
        != depth.shape[:2]
        or rgb.shape[:2]
        != mask.shape[:2]
    ):
        message = (
            "RGB/depth/mask shape mismatch: "
            f"rgb={rgb.shape}, "
            f"depth={depth.shape}, "
            f"mask={mask.shape}"
        )
        print(
            "[Recovery][FoundationPose] "
            + message
        )
        return None, False, {
            "foundationpose_error":
                message,
        }

    if np.count_nonzero(mask) < 20:
        return None, False, {
            "foundationpose_error":
                "recovery_mask_too_small",
        }

    K_arr = np.asarray(
        K,
        dtype=np.float64,
    ).reshape(3, 3)

    with tempfile.TemporaryDirectory(
        prefix="b5_template_foundationpose_"
    ) as tmpdir:
        tmp = Path(tmpdir)

        rgb_path = tmp / "rgb.npy"
        depth_path = tmp / "depth.npy"
        mask_path = tmp / "mask.npy"
        K_path = tmp / "K.txt"
        output_pose_path = (
            tmp / "T_recovery.txt"
        )
        output_diagnostics_path = tmp / "foundationpose_diagnostics.json"
        runner_path = (
            tmp
            / "foundationpose_register_runner.py"
        )

        np.save(
            rgb_path,
            rgb,
        )
        np.save(
            depth_path,
            depth,
        )
        np.save(
            mask_path,
            mask.astype(np.uint8),
        )
        np.savetxt(
            K_path,
            K_arr,
        )

        runner_path.write_text(
            _FOUNDATIONPOSE_REGISTER_RUNNER,
            encoding="utf-8",
        )

        cmd = [
            str(fp_python),
            str(runner_path),
            "--foundationpose_dir",
            str(fp_dir),
            "--rgb_npy",
            str(rgb_path),
            "--depth_npy",
            str(depth_path),
            "--mask_npy",
            str(mask_path),
            "--K_txt",
            str(K_path),
            "--mesh_file",
            str(mesh_path),
            "--output_pose",
            str(output_pose_path),
            "--refiner_weight",
            str(weight_path),
            "--output_diagnostics",
            str(output_diagnostics_path),
            "--iteration",
            str(int(refine_iter)),
        ]

        env = os.environ.copy()
        env.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True",
        )

        print(
            "[Recovery] Template2 mask -> "
            "FoundationPose.register() "
            f"(iteration={int(refine_iter)})"
        )

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            message = str(exc)
            print(
                "[Recovery][FoundationPose] "
                + message
            )
            return None, False, {
                "foundationpose_error":
                    message,
            }
        except Exception as exc:
            message = str(exc)
            print(
                "[Recovery][FoundationPose] "
                f"subprocess failed: {message}"
            )
            return None, False, {
                "foundationpose_error":
                    message,
            }

        if proc.stdout:
            print(
                proc.stdout.rstrip()
            )

        if proc.returncode != 0:
            return None, False, {
                "foundationpose_error":
                    (
                        "subprocess_exit_"
                        f"{proc.returncode}"
                    ),
                "foundationpose_stdout":
                    proc.stdout,
            }

        if not output_pose_path.is_file():
            return None, False, {
                "foundationpose_error":
                    "output_pose_missing",
                "foundationpose_stdout":
                    proc.stdout,
            }

        try:
            T_recovery = np.loadtxt(
                output_pose_path
            ).reshape(4, 4)
        except Exception as exc:
            return None, False, {
                "foundationpose_error":
                    (
                        "output_pose_read_failed: "
                        f"{exc}"
                    ),
                "foundationpose_stdout":
                    proc.stdout,
            }

        # Read before TemporaryDirectory cleanup.
        fp_diagnostics = {"foundationpose_error": None}
        if output_diagnostics_path.is_file():
            try:
                fp_diagnostics.update(json.loads(
                    output_diagnostics_path.read_text(encoding="utf-8")
                ))
            except Exception as exc:
                fp_diagnostics["foundationpose_score_diagnostics_error"] = str(exc)

    if not np.all(
        np.isfinite(T_recovery)
    ):
        return None, False, {
            "foundationpose_error":
                "non_finite_pose",
        }

    return (
        np.asarray(T_recovery, dtype=np.float64),
        True,
        fp_diagnostics,
    )


def actual_recovery_action(
    current_rgb_real,
    current_depth_real,
    model_pts_3d,
    K,
    mesh_file,
    rgb_template1,
    template1_mask,
    reference_rgb_real=None,
    reference_T_final=None,
    reference_frame_id=None,
    prebuilt_rgb_template2=None,
    prebuilt_template2_mask=None,
    prebuilt_template2_diagnostics=None,
    foundationpose_python=None,
    foundationpose_dir=None,
    foundationpose_refiner_weight=None,
    refine_iter=DEFAULT_FOUNDATIONPOSE_REFINE_ITER,
):
    """
    Recovery chain:
      first RGB + init_mask.png -> rgb_template1

      pre-blackout RGB + pre-blackout T_final + CAD + rgb_template1
          -> rgb_template2

      recovery RGB + rgb_template2
          -> template match -> padded recovery mask

      current RGB-D + recovery mask + CAD
          -> FoundationPose.register()
          -> T_recovery

    No pose-registration pre-initializer is used.
    """
    diagnostics = {
        "recovery_mask": None,
        "rgb_template1": None,
        "template1_mask": None,
        "rgb_template2": None,
        "template2_mask": None,
        "reference_frame_id": reference_frame_id,
    }

    try:
        if rgb_template1 is None or template1_mask is None:
            diagnostics["recovery_failure_reason"] = "template1_missing"
            return None, False, diagnostics

        diagnostics["rgb_template1"] = np.asarray(
            rgb_template1,
            dtype=np.uint8,
        ).copy()
        diagnostics["template1_mask"] = np.asarray(
            template1_mask,
            dtype=np.uint8,
        ).copy()

        if (
            prebuilt_rgb_template2 is not None
            and prebuilt_template2_mask is not None
        ):
            rgb_template2 = np.asarray(
                prebuilt_rgb_template2,
                dtype=np.uint8,
            ).copy()
            template2_mask = np.asarray(
                prebuilt_template2_mask,
                dtype=np.uint8,
            ).copy()
            template2_diag = dict(
                prebuilt_template2_diagnostics or {}
            )
            template2_diag["template2_source"] = (
                "frozen_pre_blackout"
            )
        else:
            (
                rgb_template2,
                template2_mask,
                template2_diag,
            ) = build_rgb_template2(
                reference_rgb_real=reference_rgb_real,
                reference_T_final=reference_T_final,
                model_pts_3d=model_pts_3d,
                K=K,
                rgb_template1=rgb_template1,
                template1_mask=template1_mask,
            )
            template2_diag["template2_source"] = (
                "cached_previous_reference"
            )

        diagnostics.update(template2_diag)

        if rgb_template2 is None or template2_mask is None:
            diagnostics["recovery_failure_reason"] = (
                "template2_build_failed"
            )
            return None, False, diagnostics

        diagnostics["rgb_template2"] = np.asarray(
            rgb_template2,
            dtype=np.uint8,
        ).copy()
        diagnostics["template2_mask"] = np.asarray(
            template2_mask,
            dtype=np.uint8,
        ).copy()

        (
            recovery_mask,
            current_match_diag,
        ) = match_rgb_template2_and_pad(
            current_rgb_real=current_rgb_real,
            rgb_template2=rgb_template2,
            template2_mask=template2_mask,
            current_depth_real=current_depth_real,
            reference_T_final=reference_T_final,
        )
        diagnostics.update(current_match_diag)

        if recovery_mask is None:
            diagnostics["recovery_failure_reason"] = (
                "template2_recovery_match_failed"
            )
            return None, False, diagnostics

        diagnostics["recovery_mask"] = np.asarray(
            recovery_mask,
            dtype=np.uint8,
        ).copy()

        (
            T_recovery,
            fp_ok,
            fp_diag,
        ) = foundationpose_register_from_mask(
            current_rgb_real=current_rgb_real,
            current_depth_real=current_depth_real,
            recovery_mask=recovery_mask,
            K=K,
            mesh_file=mesh_file,
            foundationpose_python=foundationpose_python,
            foundationpose_dir=foundationpose_dir,
            foundationpose_refiner_weight=(
                foundationpose_refiner_weight
            ),
            refine_iter=refine_iter,
        )

        if fp_diag is not None:
            diagnostics.update(fp_diag)

        if not fp_ok or T_recovery is None:
            diagnostics["recovery_failure_reason"] = (
                "foundationpose_register_failed"
            )
            return None, False, diagnostics

        diagnostics["recovery_failure_reason"] = None

        return (
            np.asarray(
                T_recovery,
                dtype=np.float64,
            ).reshape(4, 4),
            True,
            diagnostics,
        )

    except Exception as exc:
        print(
            "[Recovery] template-guided "
            f"recovery failed: {exc}"
        )
        diagnostics["recovery_failure_reason"] = str(exc)
        return None, False, diagnostics


def _copy_diag_value(
    diagnostics,
    key,
):
    if (
        diagnostics is None
        or diagnostics.get(key) is None
    ):
        return None

    value = diagnostics[key]

    if hasattr(
        value,
        "copy",
    ):
        return value.copy()

    return value


def _execute_independent_recovery(
    recovery_trigger,
    frame_index,
    frame_id,
    state,
    rgb_real,
    depth_real,
    model_pts,
    K,
    mesh_file,
    foundationpose_python,
    foundationpose_dir,
    foundationpose_refiner_weight,
    foundationpose_refine_iter,
    T_prior_current=None,
    base_sequence=None,
    recovery_debug_root="./recovery_debug",
):
    """
    blackout_exit:
        use Template2 frozen at blackout onset from the immediately
        preceding non-blackout RGB/T_final.

    prior_streak:
        build Template2 from the most recent non-blackout reference frame.
    """
    reference_rgb_real = state.get(
        "last_reference_rgb_real"
    )
    reference_T_final = state.get(
        "last_reference_T_final"
    )
    reference_frame_id = state.get(
        "last_reference_frame_id"
    )

    prebuilt_rgb_template2 = None
    prebuilt_template2_mask = None
    prebuilt_template2_diag = None

    if recovery_trigger == "blackout_exit":
        prebuilt_rgb_template2 = state.get(
            "blackout_rgb_template2"
        )
        prebuilt_template2_mask = state.get(
            "blackout_template2_mask"
        )
        prebuilt_template2_diag = state.get(
            "blackout_template2_diagnostics"
        )
        if state.get("blackout_reference_frame_id") is not None:
            reference_frame_id = state.get(
                "blackout_reference_frame_id"
            )

    (
        T_recovery,
        recovery_ok,
        diagnostics,
    ) = actual_recovery_action(
        current_rgb_real=rgb_real,
        current_depth_real=depth_real,
        model_pts_3d=model_pts,
        K=K,
        mesh_file=mesh_file,
        rgb_template1=state.get("rgb_template1"),
        template1_mask=state.get("template1_mask"),
        reference_rgb_real=reference_rgb_real,
        reference_T_final=reference_T_final,
        reference_frame_id=reference_frame_id,
        prebuilt_rgb_template2=prebuilt_rgb_template2,
        prebuilt_template2_mask=prebuilt_template2_mask,
        prebuilt_template2_diagnostics=prebuilt_template2_diag,
        foundationpose_python=foundationpose_python,
        foundationpose_dir=foundationpose_dir,
        foundationpose_refiner_weight=(
            foundationpose_refiner_weight
        ),
        refine_iter=foundationpose_refine_iter,
    )

    # FoundationPose completion means only that a raw candidate was generated.
    # Acceptance is a separate target-object pose-validity decision.
    raw_recovery_generated = bool(recovery_ok and T_recovery is not None)
    if raw_recovery_generated:
        visible_mask = diagnostics.get("recovery_visible_mask")
        if visible_mask is None:
            visible_mask = diagnostics.get("recovery_mask")
        validity_diag = evaluate_recovery_pose_validity(
            T_recovery=T_recovery,
            model_pts_3d=model_pts,
            K=K,
            visible_mask=visible_mask,
            depth_real=depth_real,
            # Keep the frozen pre-blackout reference only as a legacy fallback.
            reference_T_final=reference_T_final,
            template_score_margin=diagnostics.get("template2_score_margin"),
            # Motion plausibility is measured against the immediately previous
            # operational B5 output; current prior innovation is diagnostic.
            previous_T_final=state.get("last_operational_T_final"),
            T_prior_current=T_prior_current,
        )
        diagnostics.update(validity_diag)
        accepted_recovery = bool(validity_diag.get("accepted_recovery", False))
    else:
        accepted_recovery = False
        diagnostics.update({
            "raw_recovery_generated": False,
            "recovery_pose_valid": False,
            "accepted_recovery": False,
            "recovery_rejection_reasons": ["raw_recovery_not_generated"],
            "recovery_prev_translation_jump_m": np.inf,
            "recovery_prev_rotation_jump_deg": np.inf,
            "recovery_prior_translation_innovation_m": np.inf,
            "recovery_prior_rotation_innovation_deg": np.inf,
            "recovery_motion_reference_source": None,
            "recovery_translation_jump_m": np.inf,
            "recovery_rotation_jump_deg": np.inf,
            "recovery_reprojection_center_error_norm": np.inf,
            "recovery_template_score_margin": diagnostics.get("template2_score_margin", np.nan),
            "recovery_template_score_margin_available": False,
            "recovery_hard_gate_pass": False,
            "recovery_soft_evidence_pass_count": 0,
            "recovery_soft_evidence_required": int(RECOVERY_MIN_SOFT_EVIDENCE_PASSES),
            "recovery_soft_evidence": {},
            "recovery_rendered_depth_support": 0,
            "recovery_rendered_depth_median_residual_m": np.inf,
            "recovery_rendered_depth_mean_residual_m": np.inf,
            "recovery_cad_mask_iou": 0.0,
            "recovery_cad_coverage": 0.0,
            "recovery_quality_score": 0.0,
            "recovery_cad_projected_pixels": 0,
            "recovery_mask_pixels": int(np.count_nonzero(diagnostics.get("recovery_mask"))) if diagnostics.get("recovery_mask") is not None else 0,
            "recovery_cad_mask_valid": False,
        })

    if diagnostics.get("recovery_mask") is not None:
        debug_dir = _save_recovery_debug_artifacts(
            current_rgb_real=rgb_real,
            current_depth_real=depth_real,
            recovery_mask=diagnostics.get("recovery_mask"),
            frame_id=frame_id,
            recovery_trigger=recovery_trigger,
            base_sequence=base_sequence,
            output_root=recovery_debug_root,
            template2_candidates=diagnostics.get("template2_candidates"),
        )
        diagnostics["recovery_debug_dir"] = debug_dir

    blackout_interval = None
    if (
        recovery_trigger == "blackout_exit"
        and len(state.get("blackout_intervals", [])) > 0
    ):
        blackout_interval = dict(
            state["blackout_intervals"][-1]
        )

    recovery_info = {
        "recovery_method": (
            "InitMaskTemplate1"
            "+PreBlackoutPoseGuidedTemplate2"
            "+RecoveryFrameTemplateMatch"
            "+PaddedMask"
            "+FoundationPoseRegister"
        ),
        "recovery_trigger": str(recovery_trigger),
        "recovery_frame": frame_id,
        "recovery_frame_index": frame_index,
        "reference_frame_id": reference_frame_id,
        "init_mask_path": state.get("init_mask_path"),
        # Backward-compatible recovery_success now means POSE ACCEPTED, not
        # merely that FoundationPose returned a matrix.
        "recovery_success": bool(accepted_recovery),
        "raw_recovery_generated": bool(raw_recovery_generated),
        "recovery_pose_valid": bool(accepted_recovery),
        "accepted_recovery": bool(accepted_recovery),
        "T_raw_recovery": (
            np.asarray(T_recovery, dtype=np.float64).copy()
            if raw_recovery_generated else None
        ),
        "T_accepted_recovery": (
            np.asarray(T_recovery, dtype=np.float64).copy()
            if accepted_recovery else None
        ),
        # Deprecated alias retained for old log consumers: this is RAW pose.
        "T_recovery": (
            np.asarray(T_recovery, dtype=np.float64).copy()
            if raw_recovery_generated else None
        ),
        "blackout_interval": blackout_interval,
    }

    diagnostic_keys = [
        "recovery_mask",
        "rgb_template1",
        "template1_mask",
        "template1_bbox_xyxy",
        "rgb_template2",
        "template2_mask",
        "template2_bbox_xyxy",
        "template2_source",
        "reference_cad_bbox_xyxy",
        "template1_guided_search_bbox_xyxy",
        "template1_guided_match_success",
        "template1_guided_match_score",
        "template1_guided_match_scale",
        "template1_guided_match_angle_deg",
        "template1_guided_match_bbox_xyxy",
        "template1_guided_match_reason",
        "template2_match_success",
        "template2_match_score",
        "template2_second_score",
        "template2_score_margin",
        "template2_candidate_count",
        "template2_candidates",
        "template2_selected_valid_depth_ratio",
        "template2_selected_median_depth_m",
        "template2_selected_depth_delta_from_reference_m",
        "recovery_visible_mask",
        "template2_match_scale",
        "template2_match_angle_deg",
        "template2_match_bbox_xyxy",
        "template2_match_reason",
        "foundationpose_mask_padding_px",
        "foundationpose_error",
        "foundationpose_stdout",
        "recovery_cad_mask_iou",
        "recovery_cad_coverage",
        "recovery_quality_score",
        "recovery_cad_projected_pixels",
        "recovery_mask_pixels",
        "recovery_cad_mask_valid",
        "recovery_reprojection_center_error_norm",
        "recovery_rendered_depth_support",
        "recovery_rendered_depth_median_residual_m",
        "recovery_rendered_depth_mean_residual_m",
        "recovery_template_score_margin",
        "recovery_template_score_margin_available",
        "recovery_prev_translation_jump_m",
        "recovery_prev_rotation_jump_deg",
        "recovery_prior_translation_innovation_m",
        "recovery_prior_rotation_innovation_deg",
        "recovery_motion_reference_source",
        "recovery_translation_jump_m",
        "recovery_rotation_jump_deg",
        "recovery_hard_gate_pass",
        "recovery_soft_evidence_pass_count",
        "recovery_soft_evidence_required",
        "recovery_soft_evidence",
        "recovery_pose_valid",
        "accepted_recovery",
        "recovery_rejection_reasons",
        "recovery_debug_dir",
        "recovery_failure_reason",
    ]

    for key in diagnostic_keys:
        recovery_info[key] = _copy_diag_value(
            diagnostics,
            key,
        )
    # Preserve new diagnostics even for callers retaining legacy field lists.
    recovery_info.update({k: v for k, v in diagnostics.items()
                          if k.startswith("recovery_") and k not in recovery_info})

    return (
        T_recovery,
        bool(accepted_recovery),
        recovery_info,
    )


DEFAULT_SAM2_PYTHON = os.environ.get(
    "SAM2_PYTHON", "/home/wyg/anaconda3/envs/sam2/bin/python"
)
DEFAULT_SAM2_DIR = os.environ.get("SAM2_DIR", "/home/wyg/sam2")
DEFAULT_SAM2_CONFIG = os.environ.get(
    "SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml"
)
DEFAULT_SAM2_CHECKPOINT = os.environ.get(
    "SAM2_CHECKPOINT", "/home/wyg/sam2/checkpoints/sam2.1_hiera_large.pt"
)
DEFAULT_SAM2_CACHE_ROOT = os.environ.get(
    "SAM2_CACHE_ROOT", "./sam2_recovery_cache"
)
SAM2_MIN_MASK_PIXELS = 20


_SAM2_VIDEO_RUNNER = r'''import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--sam2_dir", required=True)
parser.add_argument("--model_cfg", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--rgb_paths", required=True)
parser.add_argument("--initial_mask", required=True)
parser.add_argument("--target_index", type=int, required=True)
parser.add_argument("--output_mask", required=True)
parser.add_argument("--output_diagnostics", required=True)
args = parser.parse_args()

sam2_dir = os.path.abspath(args.sam2_dir)
os.chdir(sam2_dir)
sys.path.insert(0, sam2_dir)
from sam2.build_sam import build_sam2_video_predictor

with open(args.rgb_paths, "r", encoding="utf-8") as stream:
    rgb_paths = json.load(stream)
if not isinstance(rgb_paths, list) or not rgb_paths:
    raise RuntimeError("rgb_paths must be a non-empty JSON list")
if args.target_index != len(rgb_paths) - 1:
    raise RuntimeError(
        f"target_index must be last supplied frame: target={args.target_index}, "
        f"n_frames={len(rgb_paths)}"
    )
for path in rgb_paths:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

initial_mask = np.load(args.initial_mask).astype(bool)
if initial_mask.ndim != 2 or int(initial_mask.sum()) < 20:
    raise RuntimeError(f"invalid official initial mask: shape={initial_mask.shape}, pixels={int(initial_mask.sum())}")

diagnostics = {
    "sam2_frames_supplied": len(rgb_paths),
    "sam2_target_index": int(args.target_index),
    "sam2_mask_pixels": 0,
    "sam2_propagated_frames": 0,
}

# SAM2 discovers only numerically named .jpg/.jpeg files. Link the exact
# manifest artifacts into that layout. PIL detects their real byte format, so
# PNG sources are not re-encoded even though the temporary link ends in .jpg.
with tempfile.TemporaryDirectory(prefix="b5_sam2_video_") as video_dir:
    for index, source in enumerate(rgb_paths):
        destination = os.path.join(video_dir, f"{index:08d}.jpg")
        try:
            os.symlink(os.path.abspath(source), destination)
        except OSError:
            try:
                os.link(os.path.abspath(source), destination)
            except OSError:
                shutil.copy2(source, destination)

    predictor = build_sam2_video_predictor(
        args.model_cfg,
        os.path.abspath(args.checkpoint),
        device="cuda",
        mode="eval",
        apply_postprocessing=True,
    )
    inference_state = predictor.init_state(
        video_path=video_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=False,
    )
    predictor.reset_state(inference_state)
    _, object_ids, initial_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        mask=initial_mask,
    )
    target_mask = None
    target_score = None
    for frame_index, object_ids, mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=len(rgb_paths),
        reverse=False,
    ):
        diagnostics["sam2_propagated_frames"] += 1
        if int(frame_index) != int(args.target_index):
            continue
        ids = [int(x) for x in object_ids]
        if 1 not in ids:
            raise RuntimeError(f"object id 1 missing at target frame; ids={ids}")
        object_index = ids.index(1)
        logits = mask_logits[object_index]
        if logits.ndim == 3:
            logits = logits[0]
        target_score = logits.detach().float().cpu().numpy()
        target_mask = target_score > 0.0
        break
    if target_mask is None:
        raise RuntimeError("SAM2 did not emit the requested recovery frame")
    if target_mask.shape != initial_mask.shape:
        raise RuntimeError(
            f"SAM2 mask shape mismatch: initial={initial_mask.shape}, target={target_mask.shape}"
        )
    diagnostics["sam2_mask_pixels"] = int(target_mask.sum())
    diagnostics["sam2_logit_median_foreground"] = (
        float(np.median(target_score[target_mask])) if target_mask.any() else None
    )
    np.save(args.output_mask, target_mask.astype(np.uint8))
    Path(args.output_diagnostics).write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8"
    )
    predictor.reset_state(inference_state)
    del inference_state, predictor, initial_logits, mask_logits, target_score

gc.collect()
torch.cuda.empty_cache()
'''


def _sam2_cache_key(
    rgb_paths, initial_mask_path, sam2_python, sam2_dir,
    sam2_config, sam2_checkpoint,
):
    digest = hashlib.sha256(b"b5_sam2_recovery_v1\0")
    for path in rgb_paths:
        resolved = os.path.abspath(str(path))
        stat = os.stat(resolved)
        digest.update(resolved.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    mask_path = os.path.abspath(str(initial_mask_path))
    stat = os.stat(mask_path)
    digest.update(mask_path.encode("utf-8"))
    digest.update(str(stat.st_size).encode("ascii"))
    digest.update(str(stat.st_mtime_ns).encode("ascii"))
    digest.update(str(sam2_config).encode("utf-8"))
    for path in (sam2_python, sam2_checkpoint):
        resolved = os.path.abspath(str(path))
        stat = os.stat(resolved)
        digest.update(resolved.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    digest.update(os.path.abspath(str(sam2_dir)).encode("utf-8"))
    try:
        commit = subprocess.run(
            ["git", "-C", str(sam2_dir), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, check=False,
        ).stdout.strip()
        digest.update(commit.encode("ascii"))
    except Exception:
        pass
    return digest.hexdigest()


def sam2_mask_at_recovery(
    rgb_paths,
    initial_mask_path,
    sam2_python=None,
    sam2_dir=None,
    sam2_config=None,
    sam2_checkpoint=None,
    cache_root=None,
):
    """Propagate the official first-frame mask through the recovery frame.

    SAM2 runs in its own process. The process exits immediately after saving the
    recovery-frame mask, releasing model/state GPU memory before FoundationPose
    starts. RGB paths are exact resolved manifest paths, in manifest order.
    """
    diagnostics = {
        "sam2_called": False,
        "sam2_subprocess_started": False,
        "sam2_cache_hit": False,
        "sam2_error": None,
        "sam2_mask_pixels": 0,
        "sam2_frames_supplied": len(rgb_paths or []),
    }
    sam2_python = sam2_python or DEFAULT_SAM2_PYTHON
    sam2_dir = sam2_dir or DEFAULT_SAM2_DIR
    sam2_config = sam2_config or DEFAULT_SAM2_CONFIG
    sam2_checkpoint = sam2_checkpoint or DEFAULT_SAM2_CHECKPOINT
    cache_root = cache_root or DEFAULT_SAM2_CACHE_ROOT
    if not rgb_paths:
        diagnostics["sam2_error"] = "no_rgb_paths_collected"
        return None, False, diagnostics
    checks = [
        (Path(sam2_python).is_file(), f"SAM2 Python not found: {sam2_python}"),
        (Path(sam2_dir).is_dir(), f"SAM2 repository not found: {sam2_dir}"),
        (Path(sam2_checkpoint).is_file(), f"SAM2 checkpoint not found: {sam2_checkpoint}"),
        (initial_mask_path is not None and Path(initial_mask_path).is_file(),
         f"official initial mask not found: {initial_mask_path}"),
    ]
    for path in rgb_paths:
        checks.append((Path(path).is_file(), f"manifest RGB not found: {path}"))
    for ok, message in checks:
        if not ok:
            diagnostics["sam2_error"] = message
            return None, False, diagnostics
    initial_mask = _load_init_mask_file(initial_mask_path)
    diagnostics["sam2_called"] = True
    try:
        cache_key = _sam2_cache_key(
            rgb_paths, initial_mask_path, sam2_python, sam2_dir,
            sam2_config, sam2_checkpoint,
        )
        cache_dir = Path(cache_root) / cache_key
        cache_mask = cache_dir / "recovery_mask.npy"
        cache_diag = cache_dir / "diagnostics.json"
        if cache_mask.is_file() and cache_diag.is_file():
            mask = np.load(cache_mask).astype(bool)
            if mask.shape == initial_mask.shape and int(mask.sum()) >= SAM2_MIN_MASK_PIXELS:
                diagnostics.update(json.loads(cache_diag.read_text(encoding="utf-8")))
                diagnostics["sam2_cache_hit"] = True
                diagnostics["sam2_mask_pixels"] = int(mask.sum())
                diagnostics["sam2_cache_dir"] = str(cache_dir.resolve())
                return mask, True, diagnostics
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="b5_sam2_runner_") as tmpdir:
            tmp = Path(tmpdir)
            runner_path = tmp / "sam2_recovery_runner.py"
            rgb_json = tmp / "rgb_paths.json"
            initial_mask_npy = tmp / "initial_mask.npy"
            output_mask = tmp / "recovery_mask.npy"
            output_diag = tmp / "diagnostics.json"
            runner_path.write_text(_SAM2_VIDEO_RUNNER, encoding="utf-8")
            rgb_json.write_text(
                json.dumps([os.path.abspath(str(x)) for x in rgb_paths]),
                encoding="utf-8",
            )
            np.save(initial_mask_npy, initial_mask.astype(np.uint8))
            command = [
                str(sam2_python), str(runner_path),
                "--sam2_dir", str(sam2_dir),
                "--model_cfg", str(sam2_config),
                "--checkpoint", str(sam2_checkpoint),
                "--rgb_paths", str(rgb_json),
                "--initial_mask", str(initial_mask_npy),
                "--target_index", str(len(rgb_paths) - 1),
                "--output_mask", str(output_mask),
                "--output_diagnostics", str(output_diag),
            ]
            env = os.environ.copy()
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            diagnostics["sam2_subprocess_started"] = True
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                check=False,
            )
            diagnostics["sam2_stdout"] = process.stdout
            if process.returncode != 0:
                diagnostics["sam2_error"] = f"sam2_subprocess_exit_{process.returncode}"
                return None, False, diagnostics
            if not output_mask.is_file():
                diagnostics["sam2_error"] = "sam2_output_mask_missing"
                return None, False, diagnostics
            mask = np.load(output_mask).astype(bool)
            if output_diag.is_file():
                diagnostics.update(json.loads(output_diag.read_text(encoding="utf-8")))
            if mask.shape != initial_mask.shape:
                diagnostics["sam2_error"] = (
                    f"sam2_mask_shape_mismatch: initial={initial_mask.shape}, current={mask.shape}"
                )
                return None, False, diagnostics
            if int(mask.sum()) < SAM2_MIN_MASK_PIXELS:
                diagnostics["sam2_error"] = "sam2_recovery_mask_too_small"
                return None, False, diagnostics
            np.save(cache_mask, mask.astype(np.uint8))
            cache_diag.write_text(
                json.dumps(
                    {key: value for key, value in diagnostics.items()
                     if key not in ("sam2_stdout", "sam2_cache_hit")},
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
        diagnostics["sam2_mask_pixels"] = int(mask.sum())
        diagnostics["sam2_cache_dir"] = str(cache_dir.resolve())
        return mask, True, diagnostics
    except Exception as exc:
        diagnostics["sam2_error"] = f"sam2_exception: {exc}"
        return None, False, diagnostics


def _execute_sam2_recovery(
    recovery_trigger,
    frame_index,
    frame_id,
    state,
    rgb_real,
    depth_real,
    model_pts,
    K,
    mesh_file,
    foundationpose_python,
    foundationpose_dir,
    foundationpose_refiner_weight,
    foundationpose_refine_iter,
    T_prior_current=None,
    base_sequence=None,
    recovery_debug_root="./recovery_debug",
    sam2_python=None,
    sam2_dir=None,
    sam2_config=None,
    sam2_checkpoint=None,
    sam2_cache_root=None,
):
    diagnostics = {
        "recovery_mask_source": "sam2.1_video_from_official_initial_mask",
        "recovery_failure_reason": None,
        "raw_recovery_generated": False,
        "accepted_recovery": False,
        "recovery_rejection_reasons": [],
    }
    expected_path_count = int(frame_index) + 1
    if (
        state.get("sam2_path_error") is not None
        or len(state.get("sam2_rgb_paths", [])) != expected_path_count
    ):
        diagnostics.update({
            "sam2_error": state.get("sam2_path_error") or (
                f"incomplete_rgb_path_sequence: expected={expected_path_count}, "
                f"collected={len(state.get('sam2_rgb_paths', []))}"
            ),
            "sam2_called": False,
            "recovery_failure_reason": "sam2_rgb_sequence_invalid",
            "recovery_rejection_reasons": ["raw_recovery_not_generated"],
        })
        info = dict(diagnostics)
        info.update({
            "recovery_method": "SAM2.1VideoFromOfficialInitMask+FoundationPoseRegister",
            "recovery_trigger": str(recovery_trigger),
            "recovery_frame": frame_id,
            "recovery_frame_index": frame_index,
            "init_mask_path": state.get("init_mask_path"),
            "recovery_success": False,
            "T_raw_recovery": None,
            "T_accepted_recovery": None,
            "T_recovery": None,
            "blackout_interval": (
                dict(state["blackout_intervals"][-1])
                if recovery_trigger == "blackout_exit" and state.get("blackout_intervals")
                else None
            ),
        })
        return None, False, info
    recovery_mask, sam_ok, sam_diag = sam2_mask_at_recovery(
        rgb_paths=state.get("sam2_rgb_paths", []),
        initial_mask_path=state.get("init_mask_path"),
        sam2_python=sam2_python,
        sam2_dir=sam2_dir,
        sam2_config=sam2_config,
        sam2_checkpoint=sam2_checkpoint,
        cache_root=sam2_cache_root,
    )
    diagnostics.update(sam_diag)
    if not sam_ok or recovery_mask is None:
        diagnostics["recovery_failure_reason"] = "sam2_recovery_mask_failed"
        diagnostics["recovery_rejection_reasons"] = ["raw_recovery_not_generated"]
        T_recovery = None
        accepted = False
    else:
        diagnostics["recovery_mask"] = recovery_mask.astype(np.uint8)
        diagnostics["recovery_visible_mask"] = recovery_mask.astype(np.uint8)
        T_recovery, fp_ok, fp_diag = foundationpose_register_from_mask(
            current_rgb_real=rgb_real,
            current_depth_real=depth_real,
            recovery_mask=recovery_mask,
            K=K,
            mesh_file=mesh_file,
            foundationpose_python=foundationpose_python,
            foundationpose_dir=foundationpose_dir,
            foundationpose_refiner_weight=foundationpose_refiner_weight,
            refine_iter=foundationpose_refine_iter,
        )
        diagnostics.update(fp_diag or {})
        raw_generated = bool(fp_ok and T_recovery is not None)
        diagnostics["raw_recovery_generated"] = raw_generated
        if raw_generated:
            validity = evaluate_recovery_pose_validity(
                T_recovery=T_recovery,
                model_pts_3d=model_pts,
                K=K,
                visible_mask=recovery_mask,
                depth_real=depth_real,
                reference_T_final=state.get("last_reference_T_final"),
                template_score_margin=diagnostics.get("foundationpose_score_margin"),
                previous_T_final=state.get("last_operational_T_final"),
                T_prior_current=T_prior_current,
            )
            diagnostics.update(validity)
            accepted = bool(validity.get("accepted_recovery", False))
            if not accepted:
                diagnostics["recovery_failure_reason"] = "raw_recovery_rejected_by_pose_validity"
        else:
            accepted = False
            diagnostics["recovery_failure_reason"] = "foundationpose_register_failed"
            diagnostics["recovery_rejection_reasons"] = ["raw_recovery_not_generated"]
    diagnostics["accepted_recovery"] = bool(accepted)
    diagnostics["recovery_pose_valid"] = bool(accepted)
    if diagnostics.get("recovery_mask") is not None:
        diagnostics["recovery_debug_dir"] = _save_recovery_debug_artifacts(
            current_rgb_real=rgb_real,
            current_depth_real=depth_real,
            recovery_mask=diagnostics["recovery_mask"],
            frame_id=frame_id,
            recovery_trigger=recovery_trigger,
            base_sequence=base_sequence,
            output_root=recovery_debug_root,
            template2_candidates=None,
        )
    info = dict(diagnostics)
    info.update({
        "recovery_method": "SAM2.1VideoFromOfficialInitMask+FoundationPoseRegister",
        "recovery_trigger": str(recovery_trigger),
        "recovery_frame": frame_id,
        "recovery_frame_index": frame_index,
        "reference_frame_id": 0,
        "init_mask_path": state.get("init_mask_path"),
        "recovery_success": bool(accepted),
        "T_raw_recovery": (
            np.asarray(T_recovery, dtype=np.float64).copy()
            if diagnostics.get("raw_recovery_generated", False) else None
        ),
        "T_accepted_recovery": (
            np.asarray(T_recovery, dtype=np.float64).copy() if accepted else None
        ),
        "T_recovery": (
            np.asarray(T_recovery, dtype=np.float64).copy()
            if diagnostics.get("raw_recovery_generated", False) else None
        ),
        "blackout_interval": (
            dict(state["blackout_intervals"][-1])
            if recovery_trigger == "blackout_exit" and state.get("blackout_intervals")
            else None
        ),
    })
    return T_recovery, bool(accepted), info


def b5_transition(
    T_obs,
    T_prior,
    support,
    depth_real,
    model_pts,
    K,
    frame_index,
    frame_id,
    state,
    E_obs_hat_cm,
    E_prior_hat_cm,
    p_obs_risk,
    p_prior_risk,
    p_risk_threshold,
    prior_advantage_margin_cm=0.1,
    blackout_min_frames=10,
    rgb_real=None,
    initial_rgb_real=None,
    initial_mask=None,
    init_mask_path=None,
    base_sequence=None,
    ycbineoat_root="./datasets/YCBInEOAT",
    mesh_file=None,
    foundationpose_python=None,
    foundationpose_dir=None,
    foundationpose_refiner_weight=None,
    foundationpose_refine_iter=DEFAULT_FOUNDATIONPOSE_REFINE_ITER,
    recovery_debug_root="./recovery_debug",
    depth_blackout_valid_ratio_threshold=DEPTH_BLACKOUT_VALID_RATIO_THRESHOLD,
    max_prior_streak=MAX_PRIOR_STREAK,
    rgb_path=None,
    sam2_python=None,
    sam2_dir=None,
    sam2_config=None,
    sam2_checkpoint=None,
    sam2_cache_root=None,
):
    """
    Final shared B5 transition used by BOTH label rollout and deployment.

    There is only one learned pose-quality semantics:

        E_hat_cm  : predicted ADD-S error of a pose hypothesis
        p_risk    : P(E > absolute_risk_threshold) from the SAME calibrator

    Observation and prior are evaluated by the SAME estimator/calibrator, so
    larger values always mean worse / riskier for both hypotheses.

    Mode decision:
      MODE 1: observation absolute risk is low.
      MODE 2: observation is risky AND
              E_prior_hat + margin < E_obs_hat.
      MODE 3: otherwise use bounded relative-quality fusion. At the streak
              limit only, preserve one LEGACY weak-fusion step and reset.

    Ground truth is never an input to this function.
    """
    _ = support  # support is a learned feature, never a blackout detector.
    from b5_revision import relative_alpha

    required_values = np.asarray([
        E_obs_hat_cm,
        E_prior_hat_cm,
        p_obs_risk,
        p_prior_risk,
        p_risk_threshold,
        prior_advantage_margin_cm,
    ], dtype=np.float64)
    if not np.all(np.isfinite(required_values)):
        raise ValueError(f"Non-finite shared B5 inputs: {required_values.tolist()}")
    if depth_real is None or model_pts is None or K is None:
        raise ValueError("depth_real, model_pts and K are required")
    if state is None:
        state = init_b5_state()

    E_obs_hat_cm = max(float(E_obs_hat_cm), 0.0)
    E_prior_hat_cm = max(float(E_prior_hat_cm), 0.0)
    p_obs_risk = float(np.clip(p_obs_risk, 0.0, 1.0))
    p_prior_risk = float(np.clip(p_prior_risk, 0.0, 1.0))
    p_risk_threshold = float(p_risk_threshold)
    prior_advantage_margin_cm = float(prior_advantage_margin_cm)
    if not 0.0 < p_risk_threshold < 1.0:
        raise ValueError(f"p_risk_threshold must be in (0,1), got {p_risk_threshold}")
    if prior_advantage_margin_cm < 0.0:
        raise ValueError("prior_advantage_margin_cm must be non-negative")

    state = dict(state)
    state["reset_motion_history"] = False
    state["last_fusion_alpha"] = None
    state["last_forced_streak_reset"] = False
    state["policy_version"] = B5_POLICY_CONFIG["version"]
    state["output_uncertain"] = bool(p_obs_risk > p_risk_threshold and p_prior_risk > p_risk_threshold)
    state["blackout_intervals"] = [
        dict(item) for item in state.get("blackout_intervals", [])
    ]
    state.setdefault("prior_streak", 0)
    state.setdefault("prior_drift_score", 0.0)
    state.setdefault("last_recovery_frame", None)
    state.setdefault("blackout_recovery_frame", None)
    state.setdefault("prior_streak_recovery_frame", None)
    # Backward-compatible initialization for states created by older code.
    state.setdefault("last_operational_T_final", None)
    state.setdefault("last_operational_frame_id", None)
    state.setdefault("sam2_rgb_paths", [])
    state.setdefault("sam2_segmentation_finished", False)
    state.setdefault("sam2_path_error", None)
    if not state["sam2_segmentation_finished"]:
        if rgb_path is None:
            state["sam2_path_error"] = "rgb_path_not_supplied"
        else:
            resolved_rgb_path = os.path.abspath(str(rgb_path))
            paths = list(state["sam2_rgb_paths"])
            if int(frame_index) != len(paths):
                state["sam2_path_error"] = (
                    f"non_contiguous_rgb_path_sequence: frame_index={frame_index}, "
                    f"collected={len(paths)}"
                )
            elif not os.path.isfile(resolved_rgb_path):
                state["sam2_path_error"] = f"rgb_path_not_found: {resolved_rgb_path}"
            else:
                paths.append(resolved_rgb_path)
                state["sam2_rgb_paths"] = paths
    state["last_E_obs_hat_cm"] = E_obs_hat_cm
    state["last_E_prior_hat_cm"] = E_prior_hat_cm
    state["last_p_obs_risk"] = p_obs_risk
    state["last_p_prior_risk"] = p_prior_risk
    state["last_predicted_error_gap_cm"] = E_prior_hat_cm - E_obs_hat_cm

    resolved_init_mask_path = _resolve_init_mask_path(
        init_mask_path=init_mask_path,
        base_sequence=base_sequence,
        ycbineoat_root=ycbineoat_root,
    )
    state = _ensure_template1_cached(
        state=state,
        current_rgb_real=rgb_real,
        initial_rgb_real=initial_rgb_real,
        initial_mask=initial_mask,
        init_mask_path=resolved_init_mask_path,
    )

    recovery_info = None

    # A. True blackout detection uses the complete current depth frame only.
    is_blackout, depth_diag = detect_depth_blackout(
        depth_real,
        valid_ratio_threshold=depth_blackout_valid_ratio_threshold,
    )
    state["is_depth_blackout"] = bool(is_blackout)
    state["depth_valid_ratio"] = float(depth_diag["depth_valid_ratio"])
    state["depth_valid_pixels"] = int(depth_diag["depth_valid_pixels"])
    state["depth_total_pixels"] = int(depth_diag["depth_total_pixels"])

    if is_blackout:
        state["consecutive_blackout"] = int(state.get("consecutive_blackout", 0)) + 1
        if state["consecutive_blackout"] == 1:
            state["blackout_start_idx"] = frame_index
            state["blackout_start_frame"] = frame_id
            state["blackout_rgb_template2"] = None
            state["blackout_template2_mask"] = None
            state["blackout_template2_diagnostics"] = None
            state["blackout_reference_frame_id"] = state.get("last_reference_frame_id")
            # SAM2 will propagate from frame 0 through the recovery frame.
            # No pose-guided template or historical-depth gate is constructed.
        state["last_blackout_idx"] = frame_index
        state["last_blackout_frame"] = frame_id
        state["prior_streak"] = 0
        state["prior_drift_score"] = 0.0
    else:
        if int(state.get("consecutive_blackout", 0)) >= int(blackout_min_frames):
            state["exited_blackout"] = True
            state["blackout_end_idx"] = frame_index
            state["blackout_end_frame"] = state["last_blackout_frame"]
            state["blackout_recovery_frame"] = frame_id
            state["blackout_intervals"].append({
                "blackout_start_index": state["blackout_start_idx"],
                "blackout_end_index": state["last_blackout_idx"],
                "recovery_index": frame_index,
                "blackout_start_frame": state["blackout_start_frame"],
                "blackout_end_frame": state["last_blackout_frame"],
                "recovery_frame": frame_id,
            })
        state["consecutive_blackout"] = 0

    # B. Only blackout exit triggers recovery; a prior-streak limit uses weak fusion.
    recovery_trigger = None
    if not is_blackout and state.get("exited_blackout", False):
        recovery_trigger = "blackout_exit"

    # C. State machine.
    if is_blackout:
        current_mode = "MODE_3_BLACKOUT_WAITING"
        T_final = T_prior
        state["last_fusion_alpha"] = 1.0
        state["output_uncertain"] = True

    elif recovery_trigger is not None:
        T_recovery, recovery_ok, recovery_info = _execute_sam2_recovery(
            recovery_trigger=recovery_trigger,
            frame_index=frame_index,
            frame_id=frame_id,
            state=state,
            rgb_real=rgb_real,
            depth_real=depth_real,
            model_pts=model_pts,
            K=K,
            mesh_file=mesh_file,
            foundationpose_python=foundationpose_python,
            foundationpose_dir=foundationpose_dir,
            foundationpose_refiner_weight=foundationpose_refiner_weight,
            foundationpose_refine_iter=foundationpose_refine_iter,
            T_prior_current=T_prior,
            base_sequence=base_sequence,
            recovery_debug_root=recovery_debug_root,
            sam2_python=sam2_python,
            sam2_dir=sam2_dir,
            sam2_config=sam2_config,
            sam2_checkpoint=sam2_checkpoint,
            sam2_cache_root=sam2_cache_root,
        )
        if recovery_ok and T_recovery is not None:
            T_final = np.asarray(T_recovery, dtype=np.float64).reshape(4, 4)
            recovery_decision = "accept_valid_recovery"
            recovery_used = True
            state["reset_motion_history"] = True
            state["motion_history_restarted"] = True
            state["output_uncertain"] = False
        else:
            T_final = T_prior
            recovery_used = False
            state["output_uncertain"] = True
            if recovery_info is not None and recovery_info.get("raw_recovery_generated", False):
                recovery_decision = "reject_invalid_raw_recovery_use_prior"
            else:
                recovery_decision = "raw_recovery_generation_failed_use_prior"
        if recovery_info is not None:
            recovery_info["recovery_fusion_alpha"] = 1.0 if recovery_used else 0.0
            recovery_info["recovery_decision"] = recovery_decision
            recovery_info["recovery_used"] = bool(recovery_used)
            recovery_info["recovery_direct_accept"] = bool(recovery_used)
            recovery_info["T_operational_b5"] = np.asarray(T_final, dtype=np.float64).copy()
            recovery_info["operational_mode"] = "MODE_3_RECOVERY_EXECUTE"
        current_mode = "MODE_3_RECOVERY_EXECUTE"
        state["last_fusion_alpha"] = None  # recovery is not obs/prior interpolation
        state["exited_blackout"] = False
        state["prior_streak"] = 0
        state["prior_drift_score"] = 0.0
        state["last_recovery_frame"] = frame_id
        state["sam2_segmentation_finished"] = True
        state["sam2_rgb_paths"] = []
        state["recovery_frame"] = frame_id  # deprecated generic alias
        if recovery_trigger == "blackout_exit":
            state["blackout_recovery_frame"] = frame_id
        else:
            state["prior_streak_recovery_frame"] = frame_id
        state["blackout_rgb_template2"] = None
        state["blackout_template2_mask"] = None
        state["blackout_template2_diagnostics"] = None
        state["blackout_reference_frame_id"] = None

    else:
        PRIOR_DRIFT_DECAY = B5_POLICY_CONFIG["prior_drift_decay"]
        PRIOR_DRIFT_INCREMENT = B5_POLICY_CONFIG["prior_drift_increment"]
        prior_streak = int(state.get("prior_streak", 0))
        prior_drift_score = float(state.get("prior_drift_score", 0.0))
        drift_penalty = max(0.0, 1.0 - prior_drift_score)

        # After five consecutive strong fusions, force one weak-fusion frame.
        # Its existing branch resets prior_streak without calling recovery.
        if prior_streak < int(max_prior_streak) and p_obs_risk <= p_risk_threshold:
            current_mode = "MODE_1_ACCEPT"
            T_final = T_obs
            state["last_fusion_alpha"] = 0.0
            state["prior_streak"] = 0
            state["prior_drift_score"] = PRIOR_DRIFT_DECAY * prior_drift_score

        elif (
            E_prior_hat_cm + prior_advantage_margin_cm < E_obs_hat_cm
            and prior_streak < int(max_prior_streak)
        ):
            current_mode = "MODE_2_UNCERTAINTY_FUSION"
            T_delta = np.linalg.inv(T_obs) @ T_prior
            alpha = relative_alpha(E_obs_hat_cm, E_prior_hat_cm, mode=2)
            state["last_fusion_alpha"] = float(alpha)
            T_final = T_obs @ se3_exp_map(alpha * se3_log_map(T_delta))
            state["prior_streak"] = prior_streak + 1
            state["prior_drift_score"] = min(
                1.0, prior_drift_score + PRIOR_DRIFT_INCREMENT
            )

        else:
            current_mode = "MODE_3_UNCERTAIN_FUSION"
            T_delta = np.linalg.inv(T_obs) @ T_prior
            if prior_streak >= int(max_prior_streak):
                # User-requested exception: retain the original weak-fusion
                # brake after five consecutive MODE2 frames. No recovery call.
                advantage_ratio = max(0.0, E_obs_hat_cm-E_prior_hat_cm)/max(E_obs_hat_cm, 1e-6)
                # Preserve the legacy formula's scale; change only its clipping bounds.
                alpha = float(np.clip(B5_POLICY_CONFIG["legacy_weak_scale"] * min(1.0, advantage_ratio)
                    * (1.0-p_prior_risk) * drift_penalty,
                    B5_POLICY_CONFIG["legacy_weak_min_alpha"], B5_POLICY_CONFIG["legacy_weak_max_alpha"]))
                state["last_forced_streak_reset"] = True
            else:
                alpha = relative_alpha(E_obs_hat_cm, E_prior_hat_cm, mode=3)
            state["last_fusion_alpha"] = float(alpha)
            T_final = T_obs @ se3_exp_map(alpha * se3_log_map(T_delta))
            state["prior_streak"] = 0
            state["prior_drift_score"] = PRIOR_DRIFT_DECAY * prior_drift_score

    # D1. Cache last NON-BLACKOUT reference for Template2 construction.
    #     This remains frozen throughout blackout.
    if not is_blackout and rgb_real is not None and T_final is not None:
        state["last_reference_rgb_real"] = np.asarray(rgb_real, dtype=np.uint8).copy()
        state["last_reference_T_final"] = np.asarray(
            T_final, dtype=np.float64
        ).reshape(4, 4).copy()
        state["last_reference_frame_id"] = frame_id

    # D2. Cache the actual operational B5 output on EVERY frame, including
    #     blackout frames.  Because this update happens at the very end of the
    #     transition, a recovery at frame t sees frame t-1 here.
    if T_final is not None:
        state["last_operational_T_final"] = np.asarray(
            T_final, dtype=np.float64
        ).reshape(4, 4).copy()
        state["last_operational_frame_id"] = frame_id

    return T_final, current_mode, state, recovery_info
