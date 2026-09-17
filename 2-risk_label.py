import runtime_settings
from online_observer import RestartableObserver, close_episode_observers
from perception_runtime import PerceptionSession, close_episode_perception
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
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
import joblib
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


SHARED_FEATURE_COLUMNS = [
    "x1_norm",
    "x2_inlier_error",
    "x4_support_ratio",
    "x5_geometry_inconsistency",
]


def shared_feature_vector(features, d_obj_cm):
    """Return the SAME 4-D pose-quality feature vector for obs or prior."""
    d_obj_cm = float(d_obj_cm)
    if not np.isfinite(d_obj_cm) or d_obj_cm <= 0:
        raise ValueError(f"Invalid object diameter: {d_obj_cm}")
    return np.asarray([
        float(features["x1"]) / d_obj_cm,
        float(features["x2"]),
        float(features["x4"]),
        float(features["x5"]),
    ], dtype=np.float64)


def predict_shared_quality(
    features,
    d_obj_cm,
    scaler,
    regressor,
    risk_calibrator,
):
    """Predict normalized error, cm error, and absolute risk for one pose."""
    x = shared_feature_vector(features, d_obj_cm).reshape(1, -1)
    x_scaled = scaler.transform(x)
    e_hat_norm = max(float(regressor.predict(x_scaled)[0]), 0.0)
    E_hat_cm = float(e_hat_norm * float(d_obj_cm))
    p_risk = float(np.clip(risk_calibrator.predict([E_hat_cm])[0], 0.0, 1.0))
    return e_hat_norm, E_hat_cm, p_risk


def _assign_temporal_split(df, train_fraction):
    """Per sequence: first fraction = fit, last fraction = calibration."""
    out = df.copy()
    out["split"] = "cal"
    for seq, sub in out.groupby("sequence", sort=False):
        unique_idx = np.sort(sub["sequence_index"].astype(int).unique())
        if len(unique_idx) < 2:
            raise ValueError(f"Sequence {seq} too short for train/cal split")
        cut = max(1, min(len(unique_idx) - 1, int(len(unique_idx) * train_fraction)))
        train_idx = set(unique_idx[:cut].tolist())
        out.loc[
            (out["sequence"] == seq)
            & out["sequence_index"].astype(int).isin(train_idx),
            "split",
        ] = "train"
    return out


def _hypothesis_samples_from_rollout(rows_df):
    """Convert one per-frame row into two same-space hypothesis samples."""
    samples = []
    for row in rows_df.itertuples(index=False):
        common = {
            "sequence": str(row.sequence),
            "sequence_index": int(row.sequence_index),
            "frame_id": int(row.frame_id),
            "D_obj_cm": float(row.D_obj_cm),
        }
        samples.append({
            **common,
            "hypothesis": "obs",
            "x1_norm": float(row.x1_obs_norm),
            "x2_inlier_error": float(row.x2_obs_inlier_error),
            "x4_support_ratio": float(row.x4_obs_support_ratio),
            "x5_geometry_inconsistency": float(row.x5_obs_geometry_inconsistency),
            "target_e_norm": float(row.e_obs_norm),
            "target_E_cm": float(row.E_obs_cm),
        })
        samples.append({
            **common,
            "hypothesis": "prior",
            "x1_norm": float(row.x1_prior_norm),
            "x2_inlier_error": float(row.x2_prior_inlier_error),
            "x4_support_ratio": float(row.x4_prior_support_ratio),
            "x5_geometry_inconsistency": float(row.x5_prior_geometry_inconsistency),
            "target_e_norm": float(row.e_prior_norm),
            "target_E_cm": float(row.E_prior_cm),
        })
    return pd.DataFrame(samples)


def fit_shared_quality_model(samples_df, risk_threshold_cm, train_fraction=0.7):
    """
    Fit ONE shared pose-error regressor and ONE monotonic risk calibrator.

    The regressor target is normalized ADD-S error E/D_obj.  Calibration uses
    predicted error in cm and the single absolute risk event E > threshold.
    """
    if len(samples_df) == 0:
        raise ValueError("No samples supplied to shared quality fit")
    samples_df = _assign_temporal_split(samples_df, train_fraction)
    feature_cols = SHARED_FEATURE_COLUMNS
    train = samples_df[samples_df["split"] == "train"].copy()
    cal = samples_df[samples_df["split"] == "cal"].copy()
    if len(train) == 0 or len(cal) == 0:
        raise ValueError("Empty train/cal split for shared quality model")

    scaler = RobustScaler()
    X_train = scaler.fit_transform(train[feature_cols].values.astype(np.float64))
    y_train = train["target_e_norm"].values.astype(np.float64)
    regressor = HuberRegressor(
        epsilon=1.35,
        alpha=1e-4,
        max_iter=2000,
        tol=1e-6,
    )
    regressor.fit(X_train, y_train)

    X_cal = scaler.transform(cal[feature_cols].values.astype(np.float64))
    e_hat_cal = np.maximum(regressor.predict(X_cal), 0.0)
    E_hat_cal_cm = e_hat_cal * cal["D_obj_cm"].values.astype(np.float64)
    E_cal_cm = cal["target_E_cm"].values.astype(np.float64)
    y_cal_risk = (E_cal_cm > float(risk_threshold_cm)).astype(np.int64)
    if len(np.unique(y_cal_risk)) < 2:
        raise ValueError(
            "Calibration split has only one absolute-risk class; cannot calibrate."
        )

    # Monotonic calibration guarantees: larger predicted error cannot imply
    # smaller absolute risk probability.
    risk_calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        increasing=True,
        out_of_bounds="clip",
    )
    risk_calibrator.fit(E_hat_cal_cm, y_cal_risk)
    p_cal = np.asarray(risk_calibrator.predict(E_hat_cal_cm), dtype=np.float64)
    p_risk_threshold, balanced_acc = select_risk_threshold(y_cal_risk, p_cal)

    # Compact fit diagnostics.
    train_hat = np.maximum(regressor.predict(X_train), 0.0)
    train_hat_cm = train_hat * train["D_obj_cm"].values.astype(np.float64)
    train_true_cm = train["target_E_cm"].values.astype(np.float64)
    metrics = {
        "train_samples": int(len(train)),
        "cal_samples": int(len(cal)),
        "train_mae_cm": float(mean_absolute_error(train_true_cm, train_hat_cm)),
        "cal_mae_cm": float(mean_absolute_error(E_cal_cm, E_hat_cal_cm)),
        "cal_risk_auroc": float(roc_auc_score(y_cal_risk, p_cal)),
        "p_risk_threshold": float(p_risk_threshold),
        "threshold_balanced_accuracy": float(balanced_acc),
    }
    return scaler, regressor, risk_calibrator, float(p_risk_threshold), metrics


def build_label_row(
    seq,
    base_sequence,
    sequence_index,
    frame_id,
    T_obs,
    T_prior,
    T_gt,
    obj_idx,
    d_objs,
    open3d_models,
    obs_features,
    prior_features,
    obs_prediction,
    prior_prediction,
    mode,
    risk_threshold_cm,
    prior_advantage_margin_cm,
    policy_model_stage,
):
    """GT is used only here, after the deployable B5 decision quantities exist."""
    E_obs_cm = U.adi(T_obs, T_gt, open3d_models[obj_idx]) * 100.0
    E_prior_cm = U.adi(T_prior, T_gt, open3d_models[obj_idx]) * 100.0
    d_obj_cm = float(d_objs[obj_idx])
    e_obs_norm = float(E_obs_cm / d_obj_cm)
    e_prior_norm = float(E_prior_cm / d_obj_cm)
    delta_E_gt_cm = float(E_prior_cm - E_obs_cm)
    margin = float(prior_advantage_margin_cm)
    if delta_E_gt_cm > margin:
        pair_state_gt = "PRIOR_WORSE"
    elif delta_E_gt_cm < -margin:
        pair_state_gt = "PRIOR_BETTER"
    else:
        pair_state_gt = "TIE"

    e_obs_hat_norm, E_obs_hat_cm, p_obs_risk = obs_prediction
    e_prior_hat_norm, E_prior_hat_cm, p_prior_risk = prior_prediction

    obs_abs_risk = int(E_obs_cm > float(risk_threshold_cm))
    prior_abs_risk = int(E_prior_cm > float(risk_threshold_cm))

    return {
        "sequence": seq,
        "base_sequence": base_sequence,
        "sequence_index": int(sequence_index),
        "frame_id": int(frame_id),
        "D_obj_cm": d_obj_cm,

        # Continuous supervision: these are the primary labels.
        "E_obs_cm": float(E_obs_cm),
        "E_prior_cm": float(E_prior_cm),
        "e_obs_norm": e_obs_norm,
        "e_prior_norm": e_prior_norm,

        # Compatibility aliases used by older merge/report scripts.
        "E_update_cm": float(E_obs_cm),
        "e_update_norm": e_obs_norm,

        # Absolute-risk diagnostics only. Same definition/direction for both.
        "obs_abs_risk_label": obs_abs_risk,
        "prior_abs_risk_label": prior_abs_risk,
        "obs_risk_label": obs_abs_risk,
        "prior_risk_label": prior_abs_risk,

        # Pairwise diagnostic; NOT a second learned classifier label.
        "delta_E_gt_cm": delta_E_gt_cm,
        "pair_state_gt": pair_state_gt,
        "prior_advantage_margin_cm": margin,

        # SAME pose-conditioned feature space for observation and prior.
        "x1_obs_depth_residual": float(obs_features["x1"]),
        "x1_obs_norm": float(obs_features["x1"] / d_obj_cm),
        "x2_obs_inlier_error": float(obs_features["x2"]),
        "x4_obs_support_ratio": float(obs_features["x4"]),
        "x5_obs_geometry_inconsistency": float(obs_features["x5"]),

        "x1_prior_depth_residual": float(prior_features["x1"]),
        "x1_prior_norm": float(prior_features["x1"] / d_obj_cm),
        "x2_prior_inlier_error": float(prior_features["x2"]),
        "x4_prior_support_ratio": float(prior_features["x4"]),
        "x5_prior_geometry_inconsistency": float(prior_features["x5"]),

        # Frozen shared-estimator outputs used by B5 in this exact rollout.
        "e_obs_hat_norm": float(e_obs_hat_norm),
        "e_prior_hat_norm": float(e_prior_hat_norm),
        "E_obs_hat_cm": float(E_obs_hat_cm),
        "E_prior_hat_cm": float(E_prior_hat_cm),
        "p_obs_risk_rollout": float(p_obs_risk),
        "p_prior_risk_rollout": float(p_prior_risk),
        "delta_E_hat_cm": float(E_prior_hat_cm - E_obs_hat_cm),
        "rollout_mode": mode,
        "policy_model_stage": str(policy_model_stage),
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


@close_episode_perception
@close_episode_observers
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
    scaler,
    regressor,
    risk_calibrator,
    p_risk_threshold,
    policy_model_stage,
):
    """
    Closed-loop B5 rollout.  This is the same B5 transition interface used at
    deployment; GT is consulted only after each transition to write supervision.
    """
    rows = []
    T_B5_history = []
    from b5_revision import make_prior, advance_history
    b5_state = init_b5_state()
    init_mask_path = resolve_initial_mask_file_for_episode(episode_df, args)
    mesh_file = resolve_foundationpose_mesh_file(args, obj_idx)
    base_sequence = str(episode_df.iloc[0]["base_sequence"])
    observer = RestartableObserver(args.observer_config, base_sequence, "observer_" + seq + ".log")
    perception = PerceptionSession(args,
        [resolve_path(p, args.data_dir) for p in episode_df["rgb_path"]],
        init_mask_path, mesh_file, "perception_" + seq + "_" + policy_model_stage)

    def read_frame(row):
        T_obs, T_gt, depth_real = load_frame_from_manifest(row, args)
        rgb_real, rgb_path = load_foundationpose_recovery_rgb(seq, row, args)
        return row, T_obs, T_gt, depth_real, rgb_real, rgb_path

    for frame_index, loaded in enumerate(perception.prefetch(read_frame, (r for _, r in episode_df.iterrows()))):
        row, T_obs, T_gt, depth_real, rgb_real, rgb_path = loaded
        frame_id = int(row["frame_id"])
        perception.advance(frame_index, rgb_path)
        T_obs = observer.observe(T_obs, rgb_path, resolve_path(row["depth_path"], args.data_dir))

        T_prior = make_prior(T_B5_history, T_obs, b5_state, compute_se3_prior)

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
            include_support=True,
        )

        obs_prediction = predict_shared_quality(
            obs_features, d_objs[obj_idx], scaler, regressor, risk_calibrator
        )
        prior_prediction = predict_shared_quality(
            prior_features, d_objs[obj_idx], scaler, regressor, risk_calibrator
        )
        _, E_obs_hat_cm, p_obs_risk = obs_prediction
        _, E_prior_hat_cm, p_prior_risk = prior_prediction

        T_final_B5, mode, b5_state, _ = b5_transition(
            T_obs=T_obs,
            T_prior=T_prior,
            support=obs_features["x4"],
            depth_real=depth_real,
            model_pts=models_pts[obj_idx],
            K=K,
            frame_index=frame_index,
            frame_id=frame_id,
            state=b5_state,
            blackout_min_frames=args.blackout_min_frames,
            rgb_real=rgb_real,
            init_mask_path=init_mask_path,
            base_sequence=base_sequence,
            ycbineoat_root=args.ycb_dir,
            mesh_file=mesh_file,
            foundationpose_python=args.foundationpose_python,
            foundationpose_dir=args.foundationpose_dir,
            foundationpose_refiner_weight=args.foundationpose_refiner_weight,
            foundationpose_refine_iter=args.foundationpose_refine_iter,
            rgb_path=rgb_path,
            sam2_python=args.sam2_python,
            sam2_dir=args.sam2_dir,
            sam2_config=args.sam2_config,
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_cache_root=args.sam2_cache_root,
            E_obs_hat_cm=E_obs_hat_cm,
            E_prior_hat_cm=E_prior_hat_cm,
            p_obs_risk=p_obs_risk,
            p_prior_risk=p_prior_risk,
            p_risk_threshold=p_risk_threshold,
            prior_advantage_margin_cm=args.prior_advantage_margin_cm,
        )
        T_B5_history = advance_history(T_B5_history, T_final_B5, b5_state)
        if b5_state.get("restart_observer"):
            observer.restart(T_final_B5)

        rows.append(build_label_row(
            seq=seq,
            base_sequence=base_sequence,
            sequence_index=int(row["sequence_index"]),
            frame_id=frame_id,
            T_obs=T_obs,
            T_prior=T_prior,
            T_gt=T_gt,
            obj_idx=obj_idx,
            d_objs=d_objs,
            open3d_models=open3d_models,
            obs_features=obs_features,
            prior_features=prior_features,
            obs_prediction=obs_prediction,
            prior_prediction=prior_prediction,
            mode=mode,
            risk_threshold_cm=args.risk_threshold,
            prior_advantage_margin_cm=args.prior_advantage_margin_cm,
            policy_model_stage=policy_model_stage,
        ))
        rows[-1].update(observer_source=observer.source, observer_wall_ms=observer.wall_ms,
            sam2_frame_wall_ms=perception.wall_ms,
            perception_runtime_version=__import__("perception_runtime").CONFIG["version"],
            observer_restarted=bool(b5_state.get("restart_observer")),
            relocalization_attempted=bool(b5_state.get("relocalization_attempted")),
            relocalization_used=bool(b5_state.get("relocalization_used")),
            output_quality_unverified=bool(b5_state.get("output_quality_unverified")),
            policy_version=b5_state.get("policy_version"),
            fusion_alpha=b5_state.get("last_fusion_alpha"),
            forced_streak_reset=bool(b5_state.get("last_forced_streak_reset")),
            motion_history_reset=bool(b5_state.get("reset_motion_history")),
            output_uncertain=bool(b5_state.get("output_uncertain")))
    return rows


def collect_observation_seed_rows(
    manifest,
    train_bases,
    args,
    models_pts,
    scenes,
    renders_obj,
    mesh_nodes,
    d_objs,
    open3d_models,
):
    """Observation-only seed supervision; no recursive prior is needed."""
    rows = []
    base_to_idx = {base: i for i, base in enumerate(args.target_seqs)}
    for base in train_bases:
        obj_idx = base_to_idx[base]
        for suffix in args.corruption_lists:
            seq = base + suffix
            episode_df = get_episode_df(manifest, seq)
            for _, row in episode_df.iterrows():
                T_obs, T_gt, depth_real = load_frame_from_manifest(row, args)
                feat = extract_pose_conditioned_features(
                    T_pose=T_obs,
                    depth_real=depth_real,
                    obj_idx=obj_idx,
                    models_pts=models_pts,
                    scenes=scenes,
                    renders_obj=renders_obj,
                    mesh_nodes=mesh_nodes,
                    include_support=True,
                )
                E_obs_cm = U.adi(T_obs, T_gt, open3d_models[obj_idx]) * 100.0
                d_obj_cm = float(d_objs[obj_idx])
                x = shared_feature_vector(feat, d_obj_cm)
                rows.append({
                    "sequence": seq,
                    "sequence_index": int(row["sequence_index"]),
                    "frame_id": int(row["frame_id"]),
                    "D_obj_cm": d_obj_cm,
                    "hypothesis": "obs",
                    "x1_norm": float(x[0]),
                    "x2_inlier_error": float(x[1]),
                    "x4_support_ratio": float(x[2]),
                    "x5_geometry_inconsistency": float(x[3]),
                    "target_e_norm": float(E_obs_cm / d_obj_cm),
                    "target_E_cm": float(E_obs_cm),
                })
    return pd.DataFrame(rows)


def save_shared_artifacts(
    args,
    held_out_base,
    train_bases,
    scaler,
    regressor,
    risk_calibrator,
    p_risk_threshold,
    metrics,
):
    paths = {
        "scaler": os.path.abspath(args.shared_scaler_out),
        "model": os.path.abspath(args.shared_model_out),
        "calibrator": os.path.abspath(args.shared_calibrator_out),
        "config": os.path.abspath(args.shared_config_out),
    }
    joblib.dump(scaler, paths["scaler"])
    joblib.dump(regressor, paths["model"])
    joblib.dump(risk_calibrator, paths["calibrator"])
    cfg = {
        "version": "shared_pose_quality_v1",
        "perception_runtime_config": __import__("perception_runtime").CONFIG.copy(),
        "execution_settings": runtime_settings.execution_config(),
        "observer_config_sha256": __import__("online_observer").digest(args.observer_config),
        "recovery_gate_config": __import__("recovery_gate").CONFIG.copy(),
        "b5_policy_config": __import__("b5_revision").CONFIG.copy(),
        "training_mode": "final_development_fit" if getattr(args, "final_fit", False) else "leave_one_sequence_out",
        "seed": int(getattr(args, "seed", 42)),
        "fit_conditions": list(args.corruption_lists),
        "held_out_base": held_out_base,
        "train_bases": list(train_bases),
        "feature_columns": list(SHARED_FEATURE_COLUMNS),
        "target": "normalized_ADD-S_error_E_over_D_obj",
        "risk_definition": f"E_cm > {float(args.risk_threshold):.6g}",
        "risk_threshold_cm": float(args.risk_threshold),
        "p_risk_threshold": float(p_risk_threshold),
        "prior_advantage_margin_cm": float(args.prior_advantage_margin_cm),
        "train_fraction": float(args.train_fraction),
        "on_policy_refine_rounds": int(args.on_policy_refine_rounds),
        "manifest_sha256": compute_sha256(args.manifest_path),
        "fit_metrics": metrics,
    }
    with open(paths["config"], "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    for key in ("scaler", "model", "calibrator"):
        cfg[f"{key}_sha256"] = compute_sha256(paths[key])
    with open(paths["config"], "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # Compatibility JSONs: there is now ONE shared threshold, not two learned
    # thresholds.  These files exist only so older archival scripts do not fail.
    compat = {
        "p_risk_threshold": float(p_risk_threshold),
        "p_obs_threshold": float(p_risk_threshold),
        "p_prior_threshold": float(p_risk_threshold),
        "shared_threshold": True,
        "deprecated_compatibility_artifact": True,
        "config_path": paths["config"],
    }
    with open("label_p_obs_threshold.json", "w", encoding="utf-8") as f:
        json.dump(compat, f, indent=2, ensure_ascii=False)
    with open("label_p_prior_threshold.json", "w", encoding="utf-8") as f:
        json.dump(compat, f, indent=2, ensure_ascii=False)
    return paths


def select_training_bases(args):
    if getattr(args, "final_fit", False):
        expected = {"mustard_easy_00_02","mustard0", "bleach0", "bleach_hard_00_03_chaitanya"}
        if len(args.target_seqs) != 4 or set(args.target_seqs) != expected:
            raise ValueError("final_fit requires exactly the four development sequences")
        return None, list(args.target_seqs)
    held_out_base = args.ci_object
    if held_out_base not in args.target_seqs:
        raise ValueError(f"ci_object/held-out base {held_out_base} not in target_seqs")
    return held_out_base, [x for x in args.target_seqs if x != held_out_base]


def main(args):
    global ARGS_RISK_THRESHOLD_CM
    ARGS_RISK_THRESHOLD_CM = args.risk_threshold
    held_out_base, train_bases = select_training_bases(args)
    np.random.seed(getattr(args, "seed", 42))
    if hasattr(o3d.utility, "random"):
        o3d.utility.random.seed(getattr(args, "seed", 42))

    manifest = pd.read_csv(args.manifest_path)
    if getattr(args, "final_fit", False):
        selected = {base + suffix for base in train_bases for suffix in args.corruption_lists}
        manifest = manifest[manifest["sequence"].isin(selected)].copy()
        if set(manifest["sequence"]) != selected:
            raise ValueError("Final-fit manifest is missing development conditions")
    required_cols = {
        "base_sequence", "condition", "sequence", "sequence_index", "frame_id",
        "rgb_path", "depth_path", "gt_path", "pred_path",
        "rgb_sha256", "depth_sha256", "gt_sha256", "pred_sha256",
        "association_method", "association_reference", "association_description",
    }
    missing_cols = required_cols - set(manifest.columns)
    if missing_cols:
        raise ValueError(f"Manifest missing columns: {sorted(missing_cols)}")
    if manifest.duplicated(["sequence", "frame_id"]).any():
        raise ValueError("Manifest contains duplicate (sequence, frame_id)")
    verify_manifest_artifacts(manifest, args)

    # Object-specific rendering assets.
    renders_obj, d_objs, scenes, mesh_nodes, models_pts, open3d_models = [], [], [], [], [], []
    for model_seq in args.cad_models_seq[:len(args.target_seqs)]:
        points_path = os.path.join(args.mesh_path_root, model_seq, "points.xyz")
        mesh_path = os.path.join(args.mesh_path_root, model_seq, "textured.obj")
        mesh = trimesh.load(mesh_path)
        render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene = pyrender.Scene()
        mesh_node = scene.add(render_mesh)
        mesh_nodes.append(mesh_node)
        camera = pyrender.IntrinsicsCamera(
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
        scene.add(camera, pose=np.eye(4))
        scenes.append(scene)
        renders_obj.append(pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480))
        model_pts = np.loadtxt(points_path, dtype=np.float64).reshape(-1, 3)
        models_pts.append(model_pts)
        open3d_models.append(U.toOpen3dCloud(
            model_pts, colors=np.zeros(model_pts.shape, dtype=np.float64)
        ))
        d = np.linalg.norm(np.max(model_pts, axis=0) - np.min(model_pts, axis=0)) * 100.0
        d_objs.append(float(d))
        print(f"Object {model_seq}: D_obj={d:.3f} cm")

    print("\n=== Shared pose-quality training: final development fit or held-out fold ===")
    print("held-out:", held_out_base)
    print("train bases:", train_bases)

    # --------------------------------------------------------------
    # Stage 0: bootstrap q0 from observation hypotheses ONLY.
    # This breaks the chicken-and-egg loop without any GT in B5 decisions.
    # --------------------------------------------------------------
    seed_df = collect_observation_seed_rows(
        manifest, train_bases, args,
        models_pts, scenes, renders_obj, mesh_nodes, d_objs, open3d_models,
    )
    scaler, regressor, calibrator, p_risk_threshold, metrics = fit_shared_quality_model(
        seed_df,
        risk_threshold_cm=args.risk_threshold,
        train_fraction=args.train_fraction,
    )
    print("Stage 0 q0 (obs-only seed):", metrics)

    base_to_idx = {base: i for i, base in enumerate(args.target_seqs)}

    # --------------------------------------------------------------
    # On-policy refinement on NON-HELD-OUT objects only.
    # Every rollout uses the same b5_transition shared-quality policy.
    # --------------------------------------------------------------
    for round_idx in range(1, int(args.on_policy_refine_rounds) + 1):
        train_rollout_rows = []
        for base in train_bases:
            obj_idx = base_to_idx[base]
            for suffix in args.corruption_lists:
                seq = base + suffix
                episode_df = get_episode_df(manifest, seq)
                print(f"[on-policy round {round_idx}] {seq}")
                train_rollout_rows += rollout_episode(
                    episode_df, seq, obj_idx, args,
                    models_pts, scenes, renders_obj, mesh_nodes,
                    d_objs, open3d_models,
                    scaler, regressor, calibrator, p_risk_threshold,
                    policy_model_stage=f"refine_input_q{round_idx-1}",
                )
        train_rollout_df = pd.DataFrame(train_rollout_rows)
        hypothesis_df = _hypothesis_samples_from_rollout(train_rollout_df)
        scaler, regressor, calibrator, p_risk_threshold, metrics = fit_shared_quality_model(
            hypothesis_df,
            risk_threshold_cm=args.risk_threshold,
            train_fraction=args.train_fraction,
        )
        print(f"Stage {round_idx} q{round_idx} fit:", metrics)

    # --------------------------------------------------------------
    # Freeze q_final, then generate FINAL labels under EXACTLY that frozen
    # B5 policy.  These rows are diagnostics/supervision; deployment should
    # load the same saved model/scaler/calibrator/config.
    # --------------------------------------------------------------
    final_rows = []
    for base in args.target_seqs:
        obj_idx = base_to_idx[base]
        for suffix in args.corruption_lists:
            seq = base + suffix
            episode_df = get_episode_df(manifest, seq)
            print(f"[FINAL frozen rollout] {seq}")
            final_rows += rollout_episode(
                episode_df, seq, obj_idx, args,
                models_pts, scenes, renders_obj, mesh_nodes,
                d_objs, open3d_models,
                scaler, regressor, calibrator, p_risk_threshold,
                policy_model_stage="final_frozen_shared_quality",
            )

    # Additional CI blackout episodes only for the held-out object, preserving
    # the existing 19-episode-per-pass workflow.
    held_idx = base_to_idx.get(held_out_base)
    for suffix in (args.ci_episode if held_out_base is not None else []):
        seq = held_out_base + suffix
        episode_df = get_episode_df(manifest, seq)
        print(f"[FINAL frozen rollout] {seq}")
        final_rows += rollout_episode(
            episode_df, seq, held_idx, args,
            models_pts, scenes, renders_obj, mesh_nodes,
            d_objs, open3d_models,
            scaler, regressor, calibrator, p_risk_threshold,
            policy_model_stage="final_frozen_shared_quality",
        )

    df = pd.DataFrame(final_rows)
    if df.duplicated(["sequence", "frame_id"]).any():
        raise ValueError("Final label dataframe has duplicate (sequence, frame_id)")
    output_csv = f"./per_frame_label_threshold{args.risk_threshold}.csv"
    df.to_csv(output_csv, index=False)

    balance_df = df.groupby("sequence").agg(
        Total_Frames=("obs_risk_label", "count"),
        Obs_Risk_Positive_Ratio=("obs_risk_label", lambda x: f"{x.mean()*100:.2f}%"),
        Prior_Risk_Positive_Ratio=("prior_risk_label", lambda x: f"{x.mean()*100:.2f}%"),
    ).reset_index()
    balance_path = f"./class_balance_summary_threshold{args.risk_threshold}.csv"
    balance_df.to_csv(balance_path, index=False)

    # Keep the historical filename, but report absolute-risk coupling plus the
    # three-way pair state; no second learned classifier exists.
    quadrant_rows = []
    for obs_state in [0, 1]:
        for prior_state in [0, 1]:
            quadrant_rows.append({
                "obs_risk_label": obs_state,
                "prior_risk_label": prior_state,
                "count": int(((df["obs_risk_label"] == obs_state) &
                              (df["prior_risk_label"] == prior_state)).sum()),
            })
    quadrant_path = f"./risk_quadrant_summary_threshold{args.risk_threshold}.csv"
    pd.DataFrame(quadrant_rows).to_csv(quadrant_path, index=False)

    pair_counts = df["pair_state_gt"].value_counts(dropna=False).to_dict()
    pair_path = f"./pair_state_summary_threshold{args.risk_threshold}.csv"
    pd.DataFrame([
        {"pair_state_gt": key, "count": int(value)}
        for key, value in pair_counts.items()
    ]).to_csv(pair_path, index=False)

    artifact_paths = save_shared_artifacts(
        args, held_out_base, train_bases,
        scaler, regressor, calibrator, p_risk_threshold, metrics,
    )

    print("\n" + "=" * 72)
    print("FINAL shared-quality label generation complete")
    print("held-out base      :", held_out_base)
    print("train bases        :", train_bases)
    print("rows / episodes    :", len(df), "/", df["sequence"].nunique())
    print("p_risk_threshold   :", f"{p_risk_threshold:.4f}")
    print("advantage margin cm:", f"{args.prior_advantage_margin_cm:.4f}")
    print("label CSV          :", output_csv)
    print("shared artifacts   :", artifact_paths)
    print("IMPORTANT: GT generated E_obs/E_prior only; it never entered B5 decisions.")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate continuous pose-error supervision with a fold-specific shared B5 pose-quality estimator"
    )
    parser.add_argument('--manifest_path', type=str, default="./reference_manifest_all27.csv")
    parser.add_argument('--ycb_dir', type=str, default="./datasets/YCBInEOAT")
    parser.add_argument('--data_dir', type=str, default="./datasets/YCBInEOAT_Corrupted")
    parser.add_argument('--res_dir', type=str, default="./results_collection")
    parser.add_argument('--mesh_path_root', type=str, default="./datasets/YCB_Video_Models/CADmodels")
    parser.add_argument('--target_seqs', nargs='+', default=["mustard_easy_00_02", "mustard0", "bleach_hard_00_03_chaitanya", "bleach0"])
    parser.add_argument('--corruption_lists', nargs='+', default=["_occ40", "_black10", "_clean", "_drop60", "_occ60"])
    parser.add_argument('--ci_object', type=str, default="bleach0", help="Held-out base object for this fold")
    parser.add_argument('--final_fit', action='store_true', help="Fit all four development sequences; no held-out fold or extra CI episodes")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ci_episode', nargs='+', default=["_black10_2", "_black10_3", "_black10_4", "_black10_5"])
    parser.add_argument('--cad_models_seq', nargs='+', default=["006_mustard_bottle", "006_mustard_bottle", "021_bleach_cleanser", "021_bleach_cleanser"])
    parser.add_argument('--risk_threshold', type=float, default=1.0, help="Absolute ADD-S risk threshold in cm")
    parser.add_argument(
        '--prior_advantage_margin_cm', type=float, default=0.1,
        help='Offline pair-label margin only; v2 mode routing uses both absolute risks'
    )
    parser.add_argument('--train_fraction', type=float, default=0.7)
    parser.add_argument('--on_policy_refine_rounds', type=int, default=1)
    parser.add_argument('--blackout_min_frames', type=int, default=10)

    parser.add_argument('--foundationpose_python', type=str, default="/home/wyg/anaconda3/envs/foundationpose/bin/python")
    parser.add_argument('--foundationpose_dir', type=str, default="/home/wyg/FoundationPose")
    parser.add_argument('--foundationpose_refiner_weight', type=str, default="/home/wyg/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth")
    parser.add_argument('--foundationpose_refine_iter', type=int, default=5)

    parser.add_argument('--sam2_python', type=str, default="/home/wyg/anaconda3/envs/sam2/bin/python")
    parser.add_argument('--sam2_dir', type=str, default="/home/wyg/sam2")
    parser.add_argument('--sam2_config', type=str, default=runtime_settings.SAM2_DEFAULT_CONFIG)
    parser.add_argument('--sam2_checkpoint', type=str, default="/home/wyg/sam2/checkpoints/sam2.1_hiera_small.pt")
    parser.add_argument('--sam2_cache_root', type=str, default="./sam2_recovery_cache")

    parser.add_argument('--shared_model_out', type=str, default="./shared_pose_quality_model.joblib")
    parser.add_argument('--shared_scaler_out', type=str, default="./shared_pose_quality_scaler.joblib")
    parser.add_argument('--shared_calibrator_out', type=str, default="./shared_risk_calibrator.joblib")
    parser.add_argument('--shared_config_out', type=str, default="./shared_quality_config.json")

    parser.add_argument("--observer_config", required=True, help="Frozen restartable SE3 config")
    args = parser.parse_args()
    __import__("online_observer").validate_config(args.observer_config)
    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError("train_fraction must be in (0,1)")
    if args.on_policy_refine_rounds < 1:
        raise ValueError("on_policy_refine_rounds must be >= 1")
    main(args)
