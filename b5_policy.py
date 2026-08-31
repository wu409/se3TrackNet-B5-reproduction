import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
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
        "consecutive_blackout": 0,
        "exited_blackout": False,

        "blackout_start_idx": 0,
        "blackout_end_idx": int(1e10),
        "blackout_start_frame": None,
        "blackout_end_frame": None,
        "last_blackout_idx": None,
        "last_blackout_frame": None,
        "recovery_frame": None,
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


def match_rgb_template2_and_pad(
    current_rgb_real,
    rgb_template2,
    template2_mask,
    padding_px=(
        FOUNDATIONPOSE_MASK_PADDING_PX
    ),
):
    """
    Match rgb_template2 on the recovery frame, then pad/dilate the matched
    binary mask before passing it to FoundationPose.register().
    """
    (
        matched_mask,
        match_diag,
    ) = _masked_multiscale_template_match(
        current_rgb_real=(
            current_rgb_real
        ),
        rgb_template=rgb_template2,
        template_mask=template2_mask,
        scales=TEMPLATE2_SCALES,
        angles_deg=(
            TEMPLATE2_ANGLES_DEG
        ),
        min_match_score=(
            TEMPLATE2_MIN_MATCH_SCORE
        ),
        search_bbox=None,
    )

    diagnostics = {
        "template2_match_success":
            bool(
                match_diag.get(
                    "match_success",
                    False,
                )
            ),
        "template2_match_score":
            float(
                match_diag.get(
                    "match_score",
                    np.nan,
                )
            ),
        "template2_match_scale":
            match_diag.get(
                "match_scale"
            ),
        "template2_match_angle_deg":
            match_diag.get(
                "match_angle_deg"
            ),
        "template2_match_bbox_xyxy":
            match_diag.get(
                "match_bbox_xyxy"
            ),
        "template2_match_reason":
            match_diag.get(
                "match_reason"
            ),
        "foundationpose_mask_padding_px":
            int(padding_px),
    }

    if matched_mask is None:
        return None, diagnostics

    mask_u8 = (
        np.asarray(
            matched_mask
        ) > 0
    ).astype(np.uint8)

    if int(padding_px) > 0:
        kernel_size = (
            2 * int(padding_px) + 1
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                kernel_size,
                kernel_size,
            ),
        )
        mask_u8 = cv2.dilate(
            mask_u8,
            kernel,
            iterations=1,
        )

    mask_u8 = (
        mask_u8 > 0
    ).astype(np.uint8)

    if np.count_nonzero(
        mask_u8
    ) < 20:
        diagnostics[
            "template2_match_success"
        ] = False
        diagnostics[
            "template2_match_reason"
        ] = "padded_mask_too_small"
        return None, diagnostics

    return (
        mask_u8.astype(bool),
        diagnostics,
    )


_FOUNDATIONPOSE_REGISTER_RUNNER = r"""
import argparse
import gc
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
parser.add_argument("--refiner_weight", required=True)
parser.add_argument("--iteration", type=int, default=5)
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
    refine_iter=5,
    timeout_sec=600,
):
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
            "--iteration",
            str(int(refine_iter)),
        ]

        env = os.environ.copy()
        env.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True",
        )
        env.setdefault(
            "FP_REFINE_CHUNK_SIZE",
            "2",
        )
        env.setdefault(
            "FP_SCORE_CHUNK_SIZE",
            "2",
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
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            message = (
                "FoundationPose register timeout "
                f"after {timeout_sec}s"
            )
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

    if not np.all(
        np.isfinite(T_recovery)
    ):
        return None, False, {
            "foundationpose_error":
                "non_finite_pose",
        }

    return (
        np.asarray(
            T_recovery,
            dtype=np.float64,
        ),
        True,
        {
            "foundationpose_error":
                None,
        },
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
    refine_iter=5,
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
        "recovery_success": bool(recovery_ok),
        "T_recovery": (
            np.asarray(
                T_recovery,
                dtype=np.float64,
            ).copy()
            if recovery_ok and T_recovery is not None
            else None
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
        "template2_match_scale",
        "template2_match_angle_deg",
        "template2_match_bbox_xyxy",
        "template2_match_reason",
        "foundationpose_mask_padding_px",
        "foundationpose_error",
        "foundationpose_stdout",
        "recovery_failure_reason",
    ]

    for key in diagnostic_keys:
        recovery_info[key] = _copy_diag_value(
            diagnostics,
            key,
        )

    return (
        T_recovery,
        bool(recovery_ok),
        recovery_info,
    )


def b5_transition(
    T_obs,
    T_prior,
    p_obs_bad,
    p_prior_bad,
    support,  # backward compatibility ONLY; deliberately ignored
    depth_real,
    model_pts,
    K,
    p_obs_threshold,
    p_prior_threshold,
    frame_index,
    frame_id,
    state,
    blackout_min_frames=10,
    use_prior_predictor=True,
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
    foundationpose_refine_iter=5,
    depth_blackout_valid_ratio_threshold=(
        DEPTH_BLACKOUT_VALID_RATIO_THRESHOLD
    ),
    max_prior_streak=MAX_PRIOR_STREAK,
):
    """
    Shared B5 transition for label rollout and deployment evaluation.

    IMPORTANT CHANGE
    ----------------
    x4/support no longer decides blackout.

    Blackout is detected ONLY from the CURRENT FULL DEPTH IMAGE:
        valid_depth_ratio =
            # valid depth pixels in the full frame / # all pixels

        blackout if:
            valid_depth_ratio <= depth_blackout_valid_ratio_threshold

    Therefore, a target object whose depth blends into the background can make
    x4=1 without falsely creating a blackout, because the background still
    contributes valid depth over the full image.

    Policy:
      1) true full-depth blackout
            -> MODE_3_BLACKOUT_WAITING, propagate T_prior

      2) first valid-depth frame after >= blackout_min_frames blackout frames
            -> Template2 matching -> padded mask -> FoundationPose.register()

      3) low observation risk
            -> MODE_1_ACCEPT

      4) observation risky + prior reliable
            -> MODE_2_UNCERTAINTY_FUSION
            -> count prior_reliance streak

      5) after max_prior_streak consecutive MODE_2 frames
            -> force independent recovery on the next valid-depth frame

      6) both uncertain
            -> MODE_3_UNCERTAIN_FUSION

    The 'support' argument is kept only so existing callers do not crash.
    It is NEVER read by this function.
    """
    # Explicitly ignore x4/support.
    _ = support

    if p_obs_threshold is None:
        raise ValueError(
            "p_obs_threshold must be provided."
        )

    if (
        use_prior_predictor
        and p_prior_bad is not None
        and p_prior_threshold is None
    ):
        raise ValueError(
            "p_prior_threshold must be provided when "
            "use_prior_predictor=True."
        )

    state = dict(state)
    state["blackout_intervals"] = [
        dict(item)
        for item in state.get(
            "blackout_intervals",
            [],
        )
    ]
    state.setdefault(
        "prior_streak",
        0,
    )
    state.setdefault(
        "prior_drift_score",
        0.0,
    )

    # Build Template1 exactly once from the first RGB and init_mask.png.
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

    # ==============================================================
    # A. TRUE blackout detection from the full depth image only.
    # ==============================================================
    is_blackout, depth_diag = detect_depth_blackout(
        depth_real,
        valid_ratio_threshold=(
            depth_blackout_valid_ratio_threshold
        ),
    )

    state["is_depth_blackout"] = bool(
        is_blackout
    )
    state["depth_valid_ratio"] = float(
        depth_diag["depth_valid_ratio"]
    )
    state["depth_valid_pixels"] = int(
        depth_diag["depth_valid_pixels"]
    )
    state["depth_total_pixels"] = int(
        depth_diag["depth_total_pixels"]
    )

    if is_blackout:
        state["consecutive_blackout"] = int(
            state.get(
                "consecutive_blackout",
                0,
            )
        ) + 1

        if state["consecutive_blackout"] == 1:
            state["blackout_start_idx"] = (
                frame_index
            )
            state["blackout_start_frame"] = (
                frame_id
            )

            # Freeze Template2 at blackout onset from the immediately
            # preceding frame: previous RGB + previous T_final + CAD
            # + Template1.
            state["blackout_rgb_template2"] = None
            state["blackout_template2_mask"] = None
            state["blackout_template2_diagnostics"] = None
            state["blackout_reference_frame_id"] = (
                state.get("last_reference_frame_id")
            )

            if (
                state.get("last_reference_rgb_real") is not None
                and state.get("last_reference_T_final") is not None
                and state.get("rgb_template1") is not None
                and state.get("template1_mask") is not None
            ):
                (
                    frozen_template2,
                    frozen_template2_mask,
                    frozen_template2_diag,
                ) = build_rgb_template2(
                    reference_rgb_real=(
                        state["last_reference_rgb_real"]
                    ),
                    reference_T_final=(
                        state["last_reference_T_final"]
                    ),
                    model_pts_3d=model_pts,
                    K=K,
                    rgb_template1=(
                        state["rgb_template1"]
                    ),
                    template1_mask=(
                        state["template1_mask"]
                    ),
                )

                if (
                    frozen_template2 is not None
                    and frozen_template2_mask is not None
                ):
                    state["blackout_rgb_template2"] = (
                        np.asarray(
                            frozen_template2,
                            dtype=np.uint8,
                        ).copy()
                    )
                    state["blackout_template2_mask"] = (
                        np.asarray(
                            frozen_template2_mask,
                            dtype=np.uint8,
                        ).copy()
                    )

                state[
                    "blackout_template2_diagnostics"
                ] = dict(
                    frozen_template2_diag or {}
                )

        state["last_blackout_idx"] = (
            frame_index
        )
        state["last_blackout_frame"] = (
            frame_id
        )

        # Blackout has its own recovery path; do not carry an older
        # prior-reliance streak through it.
        state["prior_streak"] = 0
        state["prior_drift_score"] = 0.0

    else:
        if (
            int(
                state.get(
                    "consecutive_blackout",
                    0,
                )
            )
            >= int(blackout_min_frames)
        ):
            state["exited_blackout"] = True
            state["blackout_end_idx"] = (
                frame_index
            )
            state["blackout_end_frame"] = (
                state["last_blackout_frame"]
            )
            state["recovery_frame"] = frame_id

            state["blackout_intervals"].append({
                "blackout_start_index":
                    state["blackout_start_idx"],
                "blackout_end_index":
                    state["last_blackout_idx"],
                "recovery_index":
                    frame_index,
                "blackout_start_frame":
                    state["blackout_start_frame"],
                "blackout_end_frame":
                    state["last_blackout_frame"],
                "recovery_frame":
                    frame_id,
            })

        state["consecutive_blackout"] = 0

    # ==============================================================
    # B. Determine recovery trigger.
    # ==============================================================
    recovery_trigger = None

    if (
        not is_blackout
        and state.get(
            "exited_blackout",
            False,
        )
    ):
        recovery_trigger = (
            "blackout_exit"
        )

    elif (
        not is_blackout
        and int(
            state.get(
                "prior_streak",
                0,
            )
        )
        >= int(max_prior_streak)
    ):
        recovery_trigger = (
            "prior_streak"
        )

    # ==============================================================
    # C. State machine.
    # ==============================================================
    if is_blackout:
        current_mode = (
            "MODE_3_BLACKOUT_WAITING"
        )
        T_final = T_prior

    elif recovery_trigger is not None:
        (
            T_recovery,
            recovery_ok,
            recovery_info,
        ) = _execute_independent_recovery(
            recovery_trigger=(
                recovery_trigger
            ),
            frame_index=frame_index,
            frame_id=frame_id,
            state=state,
            rgb_real=rgb_real,
            depth_real=depth_real,
            model_pts=model_pts,
            K=K,
            mesh_file=mesh_file,
            foundationpose_python=(
                foundationpose_python
            ),
            foundationpose_dir=(
                foundationpose_dir
            ),
            foundationpose_refiner_weight=(
                foundationpose_refiner_weight
            ),
            foundationpose_refine_iter=(
                foundationpose_refine_iter
            ),
        )

        if (
            recovery_ok
            and T_recovery is not None
        ):
            T_final = T_recovery
        else:
            T_final = T_prior

        current_mode = (
            "MODE_3_RECOVERY_EXECUTE"
        )

        # Recovery closes both forms of accumulated uncertainty.
        state["exited_blackout"] = False
        state["prior_streak"] = 0
        state["prior_drift_score"] = 0.0
        state["recovery_frame"] = frame_id

        state["blackout_rgb_template2"] = None
        state["blackout_template2_mask"] = None
        state["blackout_template2_diagnostics"] = None
        state["blackout_reference_frame_id"] = None

    else:
        PRIOR_DRIFT_DECAY = 0.90
        PRIOR_DRIFT_INCREMENT = 0.08

        prior_streak = int(
            state.get(
                "prior_streak",
                0,
            )
        )

        prior_drift_score = float(
            state.get(
                "prior_drift_score",
                0.0,
            )
        )

        effective_prior_risk = (
            (
                float(p_prior_bad)
                if p_prior_bad is not None
                else 1.0
            )
            + prior_drift_score
        )

        # ----------------------------------------------------------
        # MODE 1: observation accepted.
        # ----------------------------------------------------------
        if p_obs_bad <= p_obs_threshold:
            current_mode = (
                "MODE_1_ACCEPT"
            )
            T_final = T_obs

            state["prior_streak"] = 0
            state["prior_drift_score"] = (
                PRIOR_DRIFT_DECAY
                * prior_drift_score
            )

        # ----------------------------------------------------------
        # MODE 2: bounded uncertainty-aware prior assistance.
        # ----------------------------------------------------------
        elif (
            use_prior_predictor
            and effective_prior_risk
            <= p_prior_threshold
            and prior_streak
            < int(max_prior_streak)
        ):
            current_mode = (
                "MODE_2_UNCERTAINTY_FUSION"
            )

            T_delta = (
                np.linalg.inv(T_obs)
                @ T_prior
            )

            prior_confidence = max(
                0.0,
                1.0 - effective_prior_risk,
            )

            drift_penalty = max(
                0.0,
                1.0 - prior_drift_score,
            )

            alpha = np.clip(
                prior_confidence
                * drift_penalty,
                0.05,
                0.35,
            )

            # Observation remains the anchor.
            T_final = (
                T_obs
                @ se3_exp_map(
                    alpha
                    * se3_log_map(
                        T_delta
                    )
                )
            )

            state["prior_streak"] = (
                prior_streak + 1
            )

            state["prior_drift_score"] = min(
                1.0,
                prior_drift_score
                + PRIOR_DRIFT_INCREMENT,
            )

        # ----------------------------------------------------------
        # MODE 3: both uncertain.
        # ----------------------------------------------------------
        elif use_prior_predictor:
            current_mode = (
                "MODE_3_UNCERTAIN_FUSION"
            )

            T_delta = (
                np.linalg.inv(T_obs)
                @ T_prior
            )

            uncertainty = min(
                1.0,
                effective_prior_risk,
            )

            alpha = np.clip(
                0.25
                * (1.0 - uncertainty),
                0.0,
                0.25,
            )

            T_final = (
                T_obs
                @ se3_exp_map(
                    alpha
                    * se3_log_map(
                        T_delta
                    )
                )
            )

            state["prior_streak"] = 0
            state["prior_drift_score"] = (
                PRIOR_DRIFT_DECAY
                * prior_drift_score
            )

        else:
            current_mode = (
                "MODE_BOOTSTRAP_PRIOR"
            )
            T_final = T_prior

    # ==============================================================
    # D. Cache the last non-blackout reference for future recovery.
    #
    # During blackout this cache is deliberately NOT updated, so on the
    # first valid frame after blackout it still refers to the frame
    # immediately before the blackout.
    # ==============================================================
    if (
        not is_blackout
        and rgb_real is not None
        and T_final is not None
    ):
        state["last_reference_rgb_real"] = (
            np.asarray(
                rgb_real,
                dtype=np.uint8,
            ).copy()
        )
        state["last_reference_T_final"] = (
            np.asarray(
                T_final,
                dtype=np.float64,
            ).reshape(4, 4).copy()
        )
        state["last_reference_frame_id"] = (
            frame_id
        )

    return (
        T_final,
        current_mode,
        state,
        recovery_info,
    )
