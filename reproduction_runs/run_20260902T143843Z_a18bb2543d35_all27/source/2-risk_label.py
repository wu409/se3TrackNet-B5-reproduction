import os
import json
import hashlib
import numpy as np
import pandas as pd
import cv2
import Utils as U
from scipy.spatial.transform import Rotation as R_sci
import trimesh
import pyrender
import argparse
import open3d as o3d
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from b5_policy import se3_log_map, compute_se3_prior, init_b5_state, b5_transition, b5_recovery_needed



def select_risk_threshold(y_true, probs):
    """
    Learn an operating threshold from non-test labels/probabilities.

    Criterion:
        maximize balanced accuracy = 0.5 * (TPR + TNR)

    The policy later treats:
        p_bad <= threshold  -> acceptable
        p_bad >  threshold  -> risky
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)

    if len(y_true) != len(probs):
        raise ValueError(
            f"Threshold selection length mismatch: "
            f"labels={len(y_true)}, probs={len(probs)}"
        )

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "Threshold selection requires both risk classes."
        )

    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.linspace(0.05, 0.95, 91):
        pred_bad = (probs > threshold).astype(np.int64)

        tp = np.sum((pred_bad == 1) & (y_true == 1))
        tn = np.sum((pred_bad == 0) & (y_true == 0))
        fp = np.sum((pred_bad == 1) & (y_true == 0))
        fn = np.sum((pred_bad == 0) & (y_true == 1))

        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        balanced_accuracy = 0.5 * (tpr + tnr)

        if balanced_accuracy > best_score:
            best_score = float(balanced_accuracy)
            best_threshold = float(threshold)

    return best_threshold, best_score


K = np.array([
    [3.195820007324218750e+02, 0.0, 3.202149847676955687e+02],
    [0.0, 4.171186828613281250e+02, 2.443486680871046701e+02],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

cv_to_gl = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

def reliability_depth_residual(depth_real, pred_pose, scene, renderer, mesh_node):
    pose_render = cv_to_gl @ pred_pose
    scene.set_pose(mesh_node, pose_render)
    depth_render = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    valid = depth_render > 0
    if np.sum(valid) > 20:
        residual = np.abs(depth_render[valid] - depth_real[valid])
        return np.mean(residual) * 100
    return 20.0

def reliability_inlier_ratio(valid_depth, Z_pred, Z_real):
    if valid_depth.sum() > 0:
        Z_pred_valid = Z_pred[valid_depth]
        Z_real_valid = Z_real[valid_depth]
        depth_diff = np.abs(Z_pred_valid - Z_real_valid) * 100
        return 1 - np.mean(depth_diff < 2.0)
    return 1.0

def resolve_path(path_value, root):
    path_value = str(path_value)
    if os.path.isabs(path_value):
        return path_value
    if os.path.exists(path_value):
        return path_value
    return os.path.normpath(os.path.join(root, path_value))

def compute_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_manifest_artifacts(manifest, args):
    """Second-line defense for the already frozen reference manifest."""
    roots = {
        "rgb": args.data_dir,
        "depth": args.data_dir,
        "gt": args.ycb_dir,
        "pred": args.res_dir,
    }
    verified = {}
    for row in manifest.itertuples(index=False):
        for artifact_type, root in roots.items():
            path = resolve_path(getattr(row, f"{artifact_type}_path"), root)
            expected = str(getattr(row, f"{artifact_type}_sha256")).lower()
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"[{row.sequence} frame {row.frame_id}] missing {artifact_type} artifact: {path}"
                )
            actual = verified.get(os.path.abspath(path))
            if actual is None:
                actual = compute_sha256(path)
                verified[os.path.abspath(path)] = actual
            if actual != expected:
                raise ValueError(
                    f"[{row.sequence} frame {row.frame_id}] {artifact_type} SHA-256 mismatch: "
                    f"expected={expected}, actual={actual}, path={path}"
                )
    print(f"Manifest artifact verification passed: {len(verified)} unique files")

def get_episode_df(manifest, seq):
    episode_df = manifest[manifest["sequence"] == seq].copy()
    if len(episode_df) == 0:
        raise ValueError(f"Manifest中找不到序列: {seq}")
    episode_df["frame_id"] = episode_df["frame_id"].astype(int)
    episode_df["sequence_index"] = episode_df["sequence_index"].astype(int)
    if episode_df["frame_id"].duplicated().any():
        dup = episode_df.loc[episode_df["frame_id"].duplicated(), "frame_id"].tolist()
        raise ValueError(f"{seq} 存在重复frame_id: {dup}")
    if episode_df["sequence_index"].duplicated().any():
        dup = episode_df.loc[episode_df["sequence_index"].duplicated(), "sequence_index"].tolist()
        raise ValueError(f"{seq} 存在重复sequence_index: {dup}")
    episode_df = episode_df.sort_values("sequence_index", kind="stable").reset_index(drop=True)
    expected_indices = list(range(len(episode_df)))
    if episode_df["sequence_index"].tolist() != expected_indices:
        raise ValueError(
            f"{seq} manifest sequence_index不连续: "
            f"expected={expected_indices[:10]}, actual={episode_df['sequence_index'].tolist()[:10]}"
        )
    return episode_df

def load_frame_from_manifest(row, args):
    pred_path = resolve_path(row["pred_path"], args.res_dir)
    gt_path = resolve_path(row["gt_path"], args.ycb_dir)
    depth_path = resolve_path(row["depth_path"], args.data_dir)
    T_obs = np.loadtxt(pred_path).reshape(4, 4)
    T_gt = np.loadtxt(gt_path).reshape(4, 4)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"无法读取Depth: {depth_path}")
    depth_real = depth_raw.astype(np.float32) / 1000.0
    return T_obs, T_gt, depth_real

def cad_depth_geometry_inconsistency(
    T_pose,
    depth_real,
    model_pts,
    K_mat,
    inlier_threshold_m=0.02,
    min_projected_pixels=20,
):
    """
    Deployable CAD-depth geometric inconsistency for ONE pose hypothesis.

    Steps:
      1) transform CAD points by T_pose;
      2) project them to the current depth image;
      3) use a simple z-buffer (nearest CAD depth per pixel);
      4) compare rendered CAD depth with current observed depth.

    x5 = 1 - geometric inlier ratio, so larger means worse.

    This uses only current depth + CAD + K + the pose being evaluated.
    No GT is used.
    """
    T_pose = np.asarray(T_pose, dtype=np.float64).reshape(4, 4)
    pts = np.asarray(model_pts, dtype=np.float64).reshape(-1, 3)
    depth = np.asarray(depth_real, dtype=np.float32)
    K_arr = np.asarray(K_mat, dtype=np.float64).reshape(3, 3)

    h, w = depth.shape[:2]

    pts_cam = (T_pose[:3, :3] @ pts.T).T + T_pose[:3, 3]
    z = pts_cam[:, 2]

    valid_z = np.isfinite(z) & (z > 1e-8)
    if np.count_nonzero(valid_z) < min_projected_pixels:
        return 1.0

    pts_cam = pts_cam[valid_z]
    z = pts_cam[:, 2]

    u = np.rint(
        K_arr[0, 0] * pts_cam[:, 0] / z + K_arr[0, 2]
    ).astype(np.int64)
    v = np.rint(
        K_arr[1, 1] * pts_cam[:, 1] / z + K_arr[1, 2]
    ).astype(np.int64)

    inside = (
        (u >= 0) & (u < w)
        & (v >= 0) & (v < h)
        & np.isfinite(z)
    )

    if np.count_nonzero(inside) < min_projected_pixels:
        return 1.0

    u = u[inside]
    v = v[inside]
    z = z[inside]

    # z-buffer: same pixel -> keep nearest CAD surface depth
    flat_idx = v * w + u
    cad_depth_flat = np.full(h * w, np.inf, dtype=np.float64)
    np.minimum.at(cad_depth_flat, flat_idx, z)

    projected_mask = np.isfinite(cad_depth_flat)
    projected_pixels = int(np.count_nonzero(projected_mask))
    if projected_pixels < min_projected_pixels:
        return 1.0

    projected_indices = np.flatnonzero(projected_mask)
    cad_z = cad_depth_flat[projected_indices]
    obs_z = depth.reshape(-1)[projected_indices].astype(np.float64)

    valid_obs = (
        np.isfinite(obs_z)
        & (obs_z > 0.05)
        & (obs_z < 5.0)
    )

    if np.count_nonzero(valid_obs) == 0:
        return 1.0

    residual = np.abs(cad_z[valid_obs] - obs_z[valid_obs])
    inlier_count = int(
        np.count_nonzero(residual < inlier_threshold_m)
    )

    # Denominator uses ALL projected CAD pixels.
    # Missing-depth pixels therefore do not artificially inflate the score.
    geo_inlier_ratio = inlier_count / float(projected_pixels)
    return float(1.0 - geo_inlier_ratio)


def extract_pose_conditioned_features(
    T_pose,
    depth_real,
    obj_idx,
    models_pts,
    scenes,
    renders_obj,
    mesh_nodes,
    include_support=True,
):
    """
    Extract deployable features conditioned on the pose being evaluated.

    For T_obs:
      x1_obs, x2_obs, x4_obs, x5_obs

    For T_prior:
      x1_prior, x2_prior, x5_prior
      (x4_prior intentionally not used in the learned prior-risk model)

    x1/x2/x5 are recomputed independently for T_obs and T_prior.
    """
    x1_depth_residual = reliability_depth_residual(
        depth_real,
        T_pose,
        scenes[obj_idx],
        renders_obj[obj_idx],
        mesh_nodes[obj_idx],
    )

    model_pts = models_pts[obj_idx]
    R_p, t_p = T_pose[:3, :3], T_pose[:3, 3]
    pts_cam = (R_p @ model_pts.T).T + t_p

    X = pts_cam[:, 0]
    Y = pts_cam[:, 1]
    Z = pts_cam[:, 2]

    h, w = depth_real.shape[:2]

    valid_z = Z > 1e-8
    u = np.zeros(len(Z), dtype=int)
    v = np.zeros(len(Z), dtype=int)

    u[valid_z] = np.round(
        (K[0, 0] * X[valid_z] / Z[valid_z]) + K[0, 2]
    ).astype(int)

    v[valid_z] = np.round(
        (K[1, 1] * Y[valid_z] / Z[valid_z]) + K[1, 2]
    ).astype(int)

    valid_bounds = (
        valid_z
        & (u >= 0) & (u < w)
        & (v >= 0) & (v < h)
    )

    u_v = u[valid_bounds]
    v_v = v[valid_bounds]
    Z_p = Z[valid_bounds]

    if len(u_v) > 0:
        Z_real = depth_real[v_v, u_v]
        x2_inlier_error = reliability_inlier_ratio(
            Z_real > 0,
            Z_p,
            Z_real,
        )
        x4_support_ratio = (
            1.0
            - (
                np.sum(Z_real > 0.1)
                / (len(Z_real) + 1e-5)
            )
        )
    else:
        x2_inlier_error = 1.0
        x4_support_ratio = 1.0

    x5_geometry_inconsistency = cad_depth_geometry_inconsistency(
        T_pose=T_pose,
        depth_real=depth_real,
        model_pts=model_pts,
        K_mat=K,
        inlier_threshold_m=0.02,
        min_projected_pixels=20,
    )

    return {
        "x1": float(x1_depth_residual),
        "x2": float(x2_inlier_error),
        "x4": (
            float(x4_support_ratio)
            if include_support
            else None
        ),
        "x5": float(x5_geometry_inconsistency),
    }


def build_label_row(
    seq,
    frame_id,
    T_obs,
    T_prior,
    T_gt,
    obj_idx,
    d_objs,
    open3d_models,
    obs_features,
    prior_features,
    x3_trans,
    x3_rot,
    p_obs_bad,
    p_prior_bad,
    mode,
):
    """
    GT is used ONLY to create supervision labels / diagnostics.
    It never enters any learned feature.
    """
    E_update_cm = U.adi(
        T_obs,
        T_gt,
        open3d_models[obj_idx],
    ) * 100

    E_prior_cm = U.adi(
        T_prior,
        T_gt,
        open3d_models[obj_idx],
    ) * 100

    e_update_norm = E_update_cm / d_objs[obj_idx]
    e_prior_norm = E_prior_cm / d_objs[obj_idx]

    return {
        "sequence": seq,
        "frame_id": frame_id,

        "E_update_cm": E_update_cm,
        "E_prior_cm": E_prior_cm,
        "e_update_norm": e_update_norm,
        "e_prior_norm": e_prior_norm,

        "obs_risk_label":
            int(E_update_cm > ARGS_RISK_THRESHOLD_CM),
        "prior_risk_label":
            int(E_prior_cm > ARGS_RISK_THRESHOLD_CM),

        # ---------------- Observation-specific ----------------
        "x1_obs_depth_residual":
            obs_features["x1"],
        "x2_obs_inlier_error":
            obs_features["x2"],
        "x4_obs_support_ratio":
            obs_features["x4"],
        "x5_obs_geometry_inconsistency":
            obs_features["x5"],

        # ---------------- Prior-specific ----------------
        "x1_prior_depth_residual":
            prior_features["x1"],
        "x2_prior_inlier_error":
            prior_features["x2"],
        "x5_prior_geometry_inconsistency":
            prior_features["x5"],

        # ---------------- Shared temporal innovation ----------------
        # Translation unit: meter
        # Rotation unit: radian
        "x3_trans_innovation":
            float(x3_trans),
        "x3_rot_innovation":
            float(x3_rot),

        "p_obs_bad_rollout":
            p_obs_bad,
        "p_prior_bad_rollout":
            (
                p_prior_bad
                if p_prior_bad is not None
                else np.nan
            ),

        "rollout_mode": mode,
        "D_obj": d_objs[obj_idx],
    }


def _first_existing_file(candidates):
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    return None


def resolve_foundationpose_mesh_file(args, obj_idx):
    """
    Resolve object-specific CAD mesh for FoundationPose.
    Prefer textured_simple.obj; fall back to textured.obj / textured.ply.
    """
    model_seq = args.cad_models_seq[obj_idx]
    model_dir = os.path.join(args.mesh_path_root, model_seq)
    candidates = [
        os.path.join(model_dir, "textured_simple.obj"),
        os.path.join(model_dir, "textured.obj"),
        os.path.join(model_dir, "textured.ply"),
    ]
    path = _first_existing_file(candidates)
    if path is None:
        raise FileNotFoundError(
            f"[FoundationPose recovery] 找不到 CAD mesh: model_dir={model_dir}"
        )
    return path


def load_foundationpose_recovery_rgb(seq, row, args):
    """Load only current recovery-frame RGB; no mask is required."""
    rgb_path = resolve_path(row["rgb_path"], args.data_dir)
    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(
            f"[FoundationPose recovery][{seq}] RGB 不存在: {rgb_path}"
        )
    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(
            f"[FoundationPose recovery][{seq}] 无法读取 RGB: {rgb_path}"
        )
    rgb_real = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    return rgb_real, rgb_path


def resolve_initial_mask_file_for_episode(
    episode_df,
    args,
):
    """
    Resolve the official YCBInEOAT init_mask.png for this base sequence.

    Example:
      base_sequence = mustard0
      -> ./datasets/YCBInEOAT/mustard0/init_mask.png
    """
    if len(episode_df) == 0:
        raise ValueError(
            "Cannot resolve init_mask.png from an empty episode."
        )

    if "base_sequence" not in episode_df.columns:
        raise ValueError(
            "Manifest缺少 base_sequence，无法定位官方 init_mask.png"
        )

    base_sequence = str(
        episode_df.iloc[0][
            "base_sequence"
        ]
    )

    path = os.path.abspath(
        os.path.join(
            args.ycb_dir,
            base_sequence,
            "init_mask.png",
        )
    )

    if not os.path.isfile(path):
        raise FileNotFoundError(
            "[Recovery][Template1] 官方 init_mask.png 不存在: "
            f"{path}"
        )

    return path


def load_initial_template_inputs(
    initial_rgb_file,
    initial_mask_file,
):
    """
    Load the first RGB and official YCBInEOAT init_mask.png.
    rgb_template1 itself is cropped inside b5_policy.
    """
    initial_rgb_file = os.path.abspath(
        str(initial_rgb_file)
    )
    initial_mask_file = os.path.abspath(
        str(initial_mask_file)
    )

    rgb_bgr = cv2.imread(
        initial_rgb_file,
        cv2.IMREAD_COLOR,
    )

    if rgb_bgr is None:
        raise RuntimeError(
            "[Recovery][Template1] 无法读取 initial RGB: "
            f"{initial_rgb_file}"
        )

    initial_rgb_real = cv2.cvtColor(
        rgb_bgr,
        cv2.COLOR_BGR2RGB,
    )

    mask_raw = cv2.imread(
        initial_mask_file,
        cv2.IMREAD_UNCHANGED,
    )

    if mask_raw is None:
        raise RuntimeError(
            "[Recovery][Template1] 无法读取 init_mask.png: "
            f"{initial_mask_file}"
        )

    if mask_raw.ndim == 3:
        initial_mask = np.any(
            mask_raw > 0,
            axis=2,
        )
    else:
        initial_mask = (
            mask_raw > 0
        )

    if (
        initial_mask.shape[:2]
        != initial_rgb_real.shape[:2]
    ):
        raise ValueError(
            "[Recovery][Template1] initial RGB/mask 尺寸不一致: "
            f"rgb={initial_rgb_real.shape}, "
            f"mask={initial_mask.shape}"
        )

    if int(
        np.count_nonzero(
            initial_mask
        )
    ) < 20:
        raise ValueError(
            "[Recovery][Template1] init_mask 前景像素太少: "
            f"{int(np.count_nonzero(initial_mask))}"
        )

    return (
        initial_rgb_real,
        initial_mask.astype(bool),
        initial_rgb_file,
        initial_mask_file,
    )


def rollout_episode(
    episode_df,
    seq,
    obj_idx,
    args,
    models_pts,
    scenes,
    renders_obj,
    mesh_nodes,
    d_objs,
    open3d_models,
    clf_obs,
    scaler_obs,
    clf_prior=None,
    scaler_prior=None,
    use_prior_predictor=False,
    p_obs_threshold=None,
    p_prior_threshold=None,
):
    rows = []
    T_B5_history = []
    b5_state = init_b5_state()

    # Recovery Template1 source for this episode:
    # first RGB comes from the frozen manifest; init_mask.png comes from the
    # original YCBInEOAT base sequence, e.g.
    # ./datasets/YCBInEOAT/mustard0/init_mask.png
    initial_rgb_file_template1 = resolve_path(
        episode_df.iloc[0]["rgb_path"],
        args.data_dir,
    )
    initial_mask_file_template1 = (
        resolve_initial_mask_file_for_episode(
            episode_df,
            args,
        )
    )
    initial_rgb_real_template1 = None
    initial_mask_template1 = None

    for frame_index, (_, row) in enumerate(
        episode_df.iterrows()
    ):
        frame_id = int(row["frame_id"])
        T_obs, T_gt, depth_real = load_frame_from_manifest(
            row,
            args,
        )

        # Load the exact current RGB associated with this frame in the frozen
        # manifest on EVERY frame. b5_policy caches the last non-blackout
        # RGB/T_final pair; during blackout that cache remains frozen and is
        # later used to build rgb_template2.
        (
            rgb_real,
            current_rgb_path,
        ) = load_foundationpose_recovery_rgb(
            seq,
            row,
            args,
        )

        # -------------------------------------------------------------
        # Temporal prior: generated from recursive B5 history.
        # The first two frames have no two-step history, so T_prior=T_obs.
        # -------------------------------------------------------------
        if len(T_B5_history) < 2:
            T_prior = T_obs
        else:
            T_prior = compute_se3_prior(
                T_B5_history[-1],
                T_B5_history[-2],
            )

        # -------------------------------------------------------------
        # IMPORTANT: observation and prior pose-conditioned features
        # are computed INDEPENDENTLY.
        # -------------------------------------------------------------
        obs_features = extract_pose_conditioned_features(
            T_pose=T_obs,
            depth_real=depth_real,
            obj_idx=obj_idx,
            models_pts=models_pts,
            scenes=scenes,
            renders_obj=renders_obj,
            mesh_nodes=mesh_nodes,
            include_support=True,
        )

        prior_features = extract_pose_conditioned_features(
            T_pose=T_prior,
            depth_real=depth_real,
            obj_idx=obj_idx,
            models_pts=models_pts,
            scenes=scenes,
            renders_obj=renders_obj,
            mesh_nodes=mesh_nodes,
            include_support=False,
        )

        # -------------------------------------------------------------
        # Shared temporal innovation:
        # disagreement between temporal prediction and current observation.
        # It is NOT treated as a prior-only feature.
        # -------------------------------------------------------------
        innovation_vec = se3_log_map(
            np.linalg.inv(T_prior) @ T_obs
        )

        x3_trans = float(
            np.linalg.norm(
                innovation_vec[:3]
            )
        )

        x3_rot = float(
            np.linalg.norm(
                innovation_vec[3:]
            )
        )

        # -------------------------------------------------------------
        # Observation-risk predictor.
        #
        # Warm-start model: 4 dims, no temporal innovation yet.
        # Final model:      6 dims, includes shared temporal innovation.
        # -------------------------------------------------------------
        if clf_obs.n_features_in_ == 4:
            obs_raw = [[
                obs_features["x1"],
                obs_features["x2"],
                obs_features["x4"],
                obs_features["x5"],
            ]]

        elif clf_obs.n_features_in_ == 6:
            obs_raw = [[
                obs_features["x1"],
                obs_features["x2"],
                obs_features["x4"],
                obs_features["x5"],
                x3_trans,
                x3_rot,
            ]]

        else:
            raise ValueError(
                "Unexpected obs predictor dimension: "
                f"{clf_obs.n_features_in_}"
            )

        obs_feat = scaler_obs.transform(
            obs_raw
        )

        p_obs_bad = float(
            clf_obs.predict_proba(
                obs_feat
            )[0, 1]
        )

        # -------------------------------------------------------------
        # Prior-risk predictor.
        #
        # 5 dims:
        #   x1_prior, x2_prior, x5_prior,
        #   x3_trans, x3_rot
        #
        # No x4_prior. No observation-conditioned x1/x2/x5 leakage.
        # -------------------------------------------------------------
        p_prior_bad = None

        if use_prior_predictor:
            if clf_prior is None or scaler_prior is None:
                raise ValueError(
                    "Prior predictor requested but "
                    "clf_prior/scaler_prior is None."
                )

            if clf_prior.n_features_in_ != 5:
                raise ValueError(
                    "Unexpected prior predictor dimension: "
                    f"{clf_prior.n_features_in_}"
                )

            prior_raw = [[
                prior_features["x1"],
                prior_features["x2"],
                prior_features["x5"],
                x3_trans,
                x3_rot,
            ]]

            prior_feat = scaler_prior.transform(
                prior_raw
            )

            p_prior_bad = float(
                clf_prior.predict_proba(
                    prior_feat
                )[0, 1]
            )

        x4_obs = obs_features["x4"]

        # Preview the SAME recovery trigger as deployment evaluation.
        # Importantly, this does not use x4. A true blackout is detected from
        # the full depth image, and the 5-frame prior-reliance trigger is also
        # shared with b5_policy.
        (
            will_attempt_recovery,
            recovery_trigger_preview,
            _recovery_depth_diag,
        ) = b5_recovery_needed(
            depth_real=depth_real,
            state=b5_state,
            blackout_min_frames=(
                args.blackout_min_frames
            ),
        )

        if will_attempt_recovery:
            if (
                initial_rgb_real_template1 is None
                or initial_mask_template1 is None
            ):
                (
                    initial_rgb_real_template1,
                    initial_mask_template1,
                    _initial_rgb_loaded,
                    _initial_mask_loaded,
                ) = load_initial_template_inputs(
                    initial_rgb_file=(
                        initial_rgb_file_template1
                    ),
                    initial_mask_file=(
                        initial_mask_file_template1
                    ),
                )

                print(
                    f"[Recovery][Template1] initial RGB : "
                    f"{_initial_rgb_loaded}"
                )
                print(
                    f"[Recovery][Template1] initial mask: "
                    f"{_initial_mask_loaded}"
                )

            print(
                f"[Template2+FoundationPose label recovery] "
                f"seq={seq} frame={frame_id} | "
                f"trigger={recovery_trigger_preview} | "
                f"rgb={current_rgb_path}"
            )

        T_final_B5, mode, b5_state, _ = b5_transition(
            T_obs=T_obs,
            T_prior=T_prior,
            p_obs_bad=p_obs_bad,
            p_prior_bad=p_prior_bad,
            support=x4_obs,
            depth_real=depth_real,
            model_pts=models_pts[obj_idx],
            K=K,
            p_obs_threshold=p_obs_threshold,
            p_prior_threshold=p_prior_threshold,
            frame_index=frame_index,
            frame_id=frame_id,
            state=b5_state,
            blackout_min_frames=args.blackout_min_frames,
            use_prior_predictor=use_prior_predictor,

            rgb_real=rgb_real,
            initial_rgb_real=(
                initial_rgb_real_template1
            ),
            initial_mask=(
                initial_mask_template1
            ),
            mesh_file=resolve_foundationpose_mesh_file(
                args,
                obj_idx,
            ),
            foundationpose_python=(
                args.foundationpose_python
            ),
            foundationpose_dir=(
                args.foundationpose_dir
            ),
            foundationpose_refiner_weight=(
                args.foundationpose_refiner_weight
            ),
            foundationpose_refine_iter=(
                args.foundationpose_refine_iter
            ),
        )

        T_B5_history.append(
            T_final_B5
        )

        rows.append(
            build_label_row(
                seq=seq,
                frame_id=frame_id,
                T_obs=T_obs,
                T_prior=T_prior,
                T_gt=T_gt,
                obj_idx=obj_idx,
                d_objs=d_objs,
                open3d_models=open3d_models,
                obs_features=obs_features,
                prior_features=prior_features,
                x3_trans=x3_trans,
                x3_rot=x3_rot,
                p_obs_bad=p_obs_bad,
                p_prior_bad=p_prior_bad,
                mode=mode,
            )
        )

    return rows


def main(args):
    global ARGS_RISK_THRESHOLD_CM
    ARGS_RISK_THRESHOLD_CM = args.risk_threshold

    manifest = pd.read_csv(args.manifest_path)
    print(f"Consuming frozen reference manifest: {args.manifest_path}")
    required_cols = {
        "base_sequence", "condition", "sequence", "sequence_index", "frame_id",
        "rgb_path", "depth_path", "gt_path", "pred_path",
        "rgb_sha256", "depth_sha256", "gt_sha256", "pred_sha256",
        "association_method", "association_reference", "association_description",
    }
    missing_cols = required_cols - set(manifest.columns)
    if missing_cols:
        raise ValueError(f"Manifest缺少字段: {sorted(missing_cols)}")
    if manifest.duplicated(["sequence", "frame_id"]).any():
        raise ValueError("Manifest中存在重复(sequence, frame_id)")
    if manifest.duplicated(["sequence", "sequence_index"]).any():
        raise ValueError("Manifest中存在重复(sequence, sequence_index)")
    association_methods = manifest["association_method"].dropna().unique().tolist()
    if association_methods != ["official_ycbineoat_reference_sorted_index"]:
        raise ValueError(f"Manifest association_method不受支持: {association_methods}")
    verify_manifest_artifacts(manifest, args)

    renders_obj, d_objs, scenes, mesh_nodes, models_pts, open3d_models = [], [], [], [], [], []
    for model_seq in args.cad_models_seq[:len(args.target_seqs)]:
        points_path = os.path.join(args.mesh_path_root, model_seq, "points.xyz")
        mesh_path = os.path.join(args.mesh_path_root, model_seq, "textured.obj")
        mesh = trimesh.load(mesh_path)
        render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene = pyrender.Scene()
        mesh_node = scene.add(render_mesh)
        mesh_nodes.append(mesh_node)
        camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
        scene.add(camera, pose=np.eye(4))
        scenes.append(scene)
        renders_obj.append(pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480))
        with open(points_path, 'r') as f:
            model_pts = np.array([list(map(float, line.rstrip().split())) for line in f.readlines()])
        models_pts.append(model_pts)
        open3d_models.append(U.toOpen3dCloud(model_pts, colors=np.zeros(model_pts.shape, dtype=np.float64)))
        bbox_min, bbox_max = np.min(model_pts, axis=0), np.max(model_pts, axis=0)
        d = np.linalg.norm(bbox_max - bbox_min) * 100
        d_objs.append(d)
        print(f"物体 3D 直径 d_obj = {d:.2f} cm")


    print(
        "阶段 1: 训练 observation-risk warm-start predictor "
        "(obs-specific x1/x2/x4/x5)..."
    )

    stage1_X_obs = []
    stage1_y_obs = []

    for obj_idx, seq_target in enumerate(
        args.target_seqs
    ):
        for dot in args.corruption_lists:
            seq = seq_target + dot
            episode_df = get_episode_df(
                manifest,
                seq,
            )

            for _, row in episode_df.iterrows():
                (
                    T_obs,
                    T_gt,
                    depth_real,
                ) = load_frame_from_manifest(
                    row,
                    args,
                )

                obs_features = (
                    extract_pose_conditioned_features(
                        T_pose=T_obs,
                        depth_real=depth_real,
                        obj_idx=obj_idx,
                        models_pts=models_pts,
                        scenes=scenes,
                        renders_obj=renders_obj,
                        mesh_nodes=mesh_nodes,
                        include_support=True,
                    )
                )

                E_update_cm = U.adi(
                    T_obs,
                    T_gt,
                    open3d_models[obj_idx],
                ) * 100

                obs_risk_label = int(
                    E_update_cm
                    > args.risk_threshold
                )

                stage1_X_obs.append([
                    obs_features["x1"],
                    obs_features["x2"],
                    obs_features["x4"],
                    obs_features["x5"],
                ])

                stage1_y_obs.append(
                    obs_risk_label
                )

    scaler_obs_warm = MinMaxScaler()
    X_obs_warm = scaler_obs_warm.fit_transform(
        np.asarray(
            stage1_X_obs,
            dtype=np.float64,
        )
    )

    clf_obs_warm = LogisticRegression(
        max_iter=1000
    ).fit(
        X_obs_warm,
        np.asarray(stage1_y_obs),
    )

    # Bootstrap stage has no prior-risk predictor yet.
    warm_obs_probs = clf_obs_warm.predict_proba(
        X_obs_warm
    )[:, 1]
    warm_p_obs_threshold, warm_obs_balanced_accuracy = (
        select_risk_threshold(
            np.asarray(stage1_y_obs),
            warm_obs_probs,
        )
    )

    print(
        "阶段 1 完成。warm-start obs features = "
        "[x1_obs, x2_obs, x4_obs, x5_obs]"
    )
    print(
        f"[Warm threshold] p_obs_threshold="
        f"{warm_p_obs_threshold:.3f} | "
        f"balanced_accuracy="
        f"{warm_obs_balanced_accuracy:.4f}"
    )


    print("阶段 2: bootstrap B5 rollout，生成初始prior-risk labels...")
    bootstrap_rows = []
    for obj_idx, seq_target in enumerate(args.target_seqs):
        for dot in args.corruption_lists:
            seq = seq_target + dot
            episode_df = get_episode_df(manifest, seq)
            bootstrap_rows += rollout_episode(
                episode_df, seq, obj_idx, args, models_pts, scenes, renders_obj, mesh_nodes,
                d_objs, open3d_models, clf_obs_warm, scaler_obs_warm,
                use_prior_predictor=False,
                p_obs_threshold=warm_p_obs_threshold,
                p_prior_threshold=None,
            )

    bootstrap_df = pd.DataFrame(bootstrap_rows)

    # Final observation-risk model:
    # observation-conditioned geometry/visibility + shared temporal innovation.
    feature_cols_obs = [
        'x1_obs_depth_residual',
        'x2_obs_inlier_error',
        'x4_obs_support_ratio',
        'x5_obs_geometry_inconsistency',
        'x3_trans_innovation',
        'x3_rot_innovation',
    ]

    # Final prior-risk model:
    # prior-conditioned geometry + shared temporal innovation.
    # No x4_prior and no observation-conditioned x1/x2/x5 leakage.
    feature_cols_prior = [
        'x1_prior_depth_residual',
        'x2_prior_inlier_error',
        'x5_prior_geometry_inconsistency',
        'x3_trans_innovation',
        'x3_rot_innovation',
    ]


    scaler_obs_full = MinMaxScaler()
    X_obs_boot = scaler_obs_full.fit_transform(bootstrap_df[feature_cols_obs].values)
    scaler_prior_full = MinMaxScaler()
    X_prior_boot = scaler_prior_full.fit_transform(bootstrap_df[feature_cols_prior].values)

    y_obs_boot = bootstrap_df['obs_risk_label'].values
    y_prior_boot = bootstrap_df['prior_risk_label'].values
    if len(np.unique(y_obs_boot)) < 2:
        raise ValueError("bootstrap obs_risk_label只有一个类别，无法训练obs predictor")
    if len(np.unique(y_prior_boot)) < 2:
        raise ValueError(
            "bootstrap prior_risk_label只有一个类别，无法训练prior predictor。"
            "请检查risk_threshold或训练数据；不能通过周期性重置B5 history改变部署状态分布。"
        )

    clf_obs_full = LogisticRegression(max_iter=1000).fit(
        X_obs_boot,
        y_obs_boot,
    )
    clf_prior_boot = LogisticRegression(max_iter=1000).fit(
        X_prior_boot,
        y_prior_boot,
    )

    print(
        "阶段 2 完成: 已得到用于最终 label rollout 的 "
        "obs/prior predictor。"
    )

    print("\n[Standardized LogisticRegression coefficients]")
    print("Observation-risk:")
    for name, coef in zip(
        feature_cols_obs,
        clf_obs_full.coef_[0],
    ):
        print(f"  {name:35s}: {coef:+.6f}")

    print("Prior-risk:")
    for name, coef in zip(
        feature_cols_prior,
        clf_prior_boot.coef_[0],
    ):
        print(f"  {name:35s}: {coef:+.6f}")

    # Learn two independent operating thresholds from bootstrap data.
    bootstrap_p_obs = clf_obs_full.predict_proba(
        X_obs_boot
    )[:, 1]
    bootstrap_p_prior = clf_prior_boot.predict_proba(
        X_prior_boot
    )[:, 1]

    p_obs_threshold, obs_threshold_score = (
        select_risk_threshold(
            y_obs_boot,
            bootstrap_p_obs,
        )
    )
    p_prior_threshold, prior_threshold_score = (
        select_risk_threshold(
            y_prior_boot,
            bootstrap_p_prior,
        )
    )

    obs_threshold_json = {
        "p_obs_threshold": p_obs_threshold,
        "balanced_accuracy": obs_threshold_score,
        "warm_start_p_obs_threshold": warm_p_obs_threshold,
        "warm_start_balanced_accuracy":
            warm_obs_balanced_accuracy,
        "selection_protocol":
            "label_bootstrap_balanced_accuracy",
        "feature_columns": feature_cols_obs,
        "risk_label_threshold_cm":
            float(args.risk_threshold),
    }
    prior_threshold_json = {
        "p_prior_threshold": p_prior_threshold,
        "balanced_accuracy": prior_threshold_score,
        "selection_protocol":
            "label_bootstrap_balanced_accuracy",
        "feature_columns": feature_cols_prior,
        "risk_label_threshold_cm":
            float(args.risk_threshold),
    }

    with open(
        "label_p_obs_threshold.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obs_threshold_json,
            f,
            indent=4,
            ensure_ascii=False,
        )

    with open(
        "label_p_prior_threshold.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            prior_threshold_json,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\n[Frozen label-rollout probability thresholds]")
    print(
        f"  p_obs_threshold   = "
        f"{p_obs_threshold:.3f} "
        f"(balanced_acc={obs_threshold_score:.4f})"
    )
    print(
        f"  p_prior_threshold = "
        f"{p_prior_threshold:.3f} "
        f"(balanced_acc={prior_threshold_score:.4f})"
    )
    print("  saved: label_p_obs_threshold.json")
    print("  saved: label_p_prior_threshold.json")

    print("阶段 3: 使用完整 B5 transition 重新rollout并生成最终labels...")
    csv_rows = []

    for obj_idx, seq_target in enumerate(args.target_seqs):
        for dot in args.corruption_lists:
            seq = seq_target + dot
            episode_df = get_episode_df(manifest, seq)
            csv_rows += rollout_episode(
                episode_df, seq, obj_idx, args, models_pts, scenes, renders_obj, mesh_nodes,
                d_objs, open3d_models, clf_obs_full, scaler_obs_full,
                clf_prior=clf_prior_boot, scaler_prior=scaler_prior_full,
                use_prior_predictor=True,
                p_obs_threshold=p_obs_threshold,
                p_prior_threshold=p_prior_threshold,
            )


    ci_obj_idx = args.target_seqs.index(args.ci_object)
    for episode in args.ci_episode:
        seq = args.ci_object + episode
        episode_df = get_episode_df(manifest, seq)
        csv_rows += rollout_episode(
            episode_df, seq, ci_obj_idx, args, models_pts, scenes, renders_obj, mesh_nodes,
            d_objs, open3d_models, clf_obs_full, scaler_obs_full,
            clf_prior=clf_prior_boot, scaler_prior=scaler_prior_full,
            use_prior_predictor=True,
            p_obs_threshold=p_obs_threshold,
            p_prior_threshold=p_prior_threshold,
        )

    df = pd.DataFrame(csv_rows)
    output_csv = f"./per_frame_label_threshold{args.risk_threshold}.csv"
    df.to_csv(output_csv, index=False)

    balance_df = df.groupby('sequence').agg(
        Total_Frames=('obs_risk_label', 'count'),
        Obs_Risk_Positive_Ratio=('obs_risk_label', lambda x: f"{x.mean()*100:.2f}%"),
        Prior_Risk_Positive_Ratio=('prior_risk_label', lambda x: f"{x.mean()*100:.2f}%")
    ).reset_index()
    balance_csv_path = f"./class_balance_summary_threshold{args.risk_threshold}.csv"
    balance_df.to_csv(balance_csv_path, index=False)

    # -------------------------------------------------------------
    # Decoupling diagnostics:
    # These do NOT affect training/inference; they only report whether
    # the two risk predictors still collapse into the same state.
    # -------------------------------------------------------------
    valid_prob = (
        np.isfinite(
            df["p_obs_bad_rollout"].values
        )
        & np.isfinite(
            df["p_prior_bad_rollout"].values
        )
    )

    if np.count_nonzero(valid_prob) > 1:
        prob_corr = float(
            np.corrcoef(
                df.loc[
                    valid_prob,
                    "p_obs_bad_rollout",
                ].values,
                df.loc[
                    valid_prob,
                    "p_prior_bad_rollout",
                ].values,
            )[0, 1]
        )
    else:
        prob_corr = np.nan

    label_quadrant = pd.crosstab(
        df["obs_risk_label"],
        df["prior_risk_label"],
        rownames=["obs_risk_label"],
        colnames=["prior_risk_label"],
        dropna=False,
    )

    prob_quadrant_df = df.loc[
        valid_prob,
        [
            "p_obs_bad_rollout",
            "p_prior_bad_rollout",
        ],
    ].copy()

    if len(prob_quadrant_df) > 0:
        prob_quadrant_df["obs_bad_pred"] = (
            prob_quadrant_df[
                "p_obs_bad_rollout"
            ] > p_obs_threshold
        ).astype(int)

        prob_quadrant_df["prior_bad_pred"] = (
            prob_quadrant_df[
                "p_prior_bad_rollout"
            ] > p_prior_threshold
        ).astype(int)

        prob_quadrant = pd.crosstab(
            prob_quadrant_df[
                "obs_bad_pred"
            ],
            prob_quadrant_df[
                "prior_bad_pred"
            ],
            rownames=["obs_bad_pred"],
            colnames=["prior_bad_pred"],
            dropna=False,
        )
    else:
        prob_quadrant = pd.DataFrame()

    coupling_csv_path = (
        f"./risk_quadrant_summary_threshold"
        f"{args.risk_threshold}.csv"
    )

    quadrant_rows = []
    for obs_state in [0, 1]:
        for prior_state in [0, 1]:
            count = int(
                (
                    (df["obs_risk_label"] == obs_state)
                    & (
                        df["prior_risk_label"]
                        == prior_state
                    )
                ).sum()
            )
            quadrant_rows.append({
                "obs_risk_label": obs_state,
                "prior_risk_label": prior_state,
                "count": count,
            })

    pd.DataFrame(
        quadrant_rows
    ).to_csv(
        coupling_csv_path,
        index=False,
    )

    print("\\n[Obs/Prior decoupling diagnostics]")
    print(
        "Pearson corr("
        "p_obs_bad_rollout, p_prior_bad_rollout"
        f") = {prob_corr:.6f}"
        if np.isfinite(prob_corr)
        else "Probability correlation unavailable."
    )
    print("\\nGT-label quadrants:")
    print(label_quadrant)

    if len(prob_quadrant) > 0:
        print(
            f"\\nPredicted probability quadrants "
            f"(p_obs_threshold={p_obs_threshold:.3f}, p_prior_threshold={p_prior_threshold:.3f}):"
        )
        print(prob_quadrant)

    print(
        f"Risk quadrant summary saved: "
        f"{coupling_csv_path}"
    )

    print("\n" + "=" * 60)
    print(f"数据总行数: {len(df)}")
    print(f"obs_risk=1 占比: {df['obs_risk_label'].mean()*100:.2f}%")
    print(f"prior_risk=1 占比: {df['prior_risk_label'].mean()*100:.2f}%")
    print(f"逐帧标签: {output_csv}")
    print(f"类别平衡: {balance_csv_path}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 YCBInEOAT observation/prior risk labels")
    parser.add_argument('--manifest_path', type=str, default="./reference_manifest.csv",
                        help="冻结的 reference manifest；本程序不会重建或覆盖它")
    parser.add_argument('--ycb_dir', type=str, default="./datasets/YCBInEOAT", help="GT根目录")
    parser.add_argument('--data_dir', type=str, default="./datasets/YCBInEOAT_Corrupted", help="RGB/Depth根目录")
    parser.add_argument('--res_dir', type=str, default="./results_collection", help="Prediction根目录")
    parser.add_argument('--mesh_path_root', type=str, default="./datasets/YCB_Video_Models/CADmodels")
    parser.add_argument('--target_seqs', nargs='+', default=["mustard0", "bleach_hard_00_03_chaitanya", "bleach0"])
    parser.add_argument('--corruption_lists', nargs='+', default=["_occ40", "_black10", "_clean", "_drop60", "_occ60"])
    parser.add_argument('--ci_object', type=str, default="bleach0")
    parser.add_argument('--ci_episode', nargs='+', default=["_black10_2", "_black10_3", "_black10_4", "_black10_5"])
    parser.add_argument('--cad_models_seq', nargs='+', default=["006_mustard_bottle", "021_bleach_cleanser", "021_bleach_cleanser"])
    parser.add_argument('--risk_threshold', type=float, default=1.0, help="ADD-S风险阈值(cm)")
    parser.add_argument('--blackout_min_frames', type=int, default=10)

    # ==================== FoundationPose independent recovery ====================
    parser.add_argument(
        '--foundationpose_python',
        type=str,
        default="/home/wyg/anaconda3/envs/foundationpose/bin/python",
        help="FoundationPose conda 环境 Python 的绝对路径"
    )
    parser.add_argument(
        '--foundationpose_dir',
        type=str,
        default="/home/wyg/FoundationPose",
        help="FoundationPose repository 根目录"
    )
    parser.add_argument(
        '--foundationpose_refiner_weight',
        type=str,
        default="/home/wyg/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth",
        help="冻结的 FoundationPose PoseRefinePredictor 权重"
    )
    parser.add_argument(
        '--foundationpose_refine_iter',
        type=int,
        default=5,
        help="FoundationPose PoseRefinePredictor refinement iterations"
    )
    args = parser.parse_args()

    print("\n[FoundationPose label-rollout configuration]")
    print("  python :", args.foundationpose_python)
    print("  repo   :", args.foundationpose_dir)
    print("  refiner weight:", args.foundationpose_refiner_weight)
    print("  refine_iter:", args.foundationpose_refine_iter)

    main(args)
