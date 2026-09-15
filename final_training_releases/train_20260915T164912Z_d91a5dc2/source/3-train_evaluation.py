import os
import time
import importlib.util
import json
import numpy as np
import open3d as o3d
import cv2
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from b5_policy import se3_log_map, se3_exp_map, compute_se3_prior, init_b5_state, b5_transition
import Utils as U
from scipy.spatial.transform import Rotation as R_sci
import numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, mean_absolute_error, balanced_accuracy_score
import argparse
import joblib
import trimesh
import pyrender
from scipy.stats import t, spearmanr
import hashlib
import pandas as pd


K = np.array([
    [3.195820007324218750e+02,    0.0,   3.202149847676955687e+02],
    [   0.0,  4.171186828613281250e+02, 2.443486680871046701e+02],
    [   0.0,      0.0,     1.0   ]
], dtype=np.float64)

# ==================== 1. SE(3) 李群辅助函数 ====================
def calc_auc(errors_cm, max_threshold_cm=10.0):
    errors_array = np.array(errors_cm)
    if len(errors_array) == 0: return 0.0
    thresholds = np.linspace(0, max_threshold_cm, 1000)
    accs = [np.mean(errors_array <= th) for th in thresholds]
    return (np.trapz(accs, thresholds) / max_threshold_cm) * 100.0

def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = ((probs >= bin_boundaries[i]) if i == 0 else (probs > bin_boundaries[i])) & (probs <= bin_boundaries[i+1])
        if np.sum(in_bin) > 0:
            ece += np.abs(np.mean(labels[in_bin]) - np.mean(probs[in_bin])) * (np.sum(in_bin) / len(probs))
    return ece


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


def compute_episode_level_ci(scores, n_bootstraps=2000, confidence_level=0.95):
    """按 Episode 计算 95% 置信区间 """
    scores_arr = np.array(scores)
    if len(scores_arr) < 2:
        m = np.mean(scores_arr)
        return m, m, m # 单 Episode 直接返回均值

    bootstrapped_means = []
    np.random.seed(42)
    for _ in range(n_bootstraps):
        sample = np.random.choice(scores_arr, size=len(scores_arr), replace=True)
        bootstrapped_means.append(np.mean(sample))

    alpha = (1.0 - confidence_level) / 2.0
    lower_bound = np.percentile(bootstrapped_means, alpha * 100)
    upper_bound = np.percentile(bootstrapped_means, (1.0 - alpha) * 100)
    mean_score = np.mean(scores_arr)

    return mean_score, lower_bound, upper_bound



def compute_paired_bootstrap_ci(differences, n_bootstraps=10000, confidence_level=0.95, seed=42):
    """对 episode-level paired differences 做 bootstrap CI。"""
    diffs = np.asarray(differences, dtype=np.float64)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return np.nan, np.nan, np.nan
    mean_diff = float(np.mean(diffs))
    if len(diffs) == 1:
        return mean_diff, mean_diff, mean_diff
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstraps, dtype=np.float64)
    for b in range(n_bootstraps):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot_means[b] = np.mean(sample)
    alpha = (1.0 - confidence_level) / 2.0
    return mean_diff, float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1.0 - alpha) * 100))

def compute_full_sha256(filepath):
    """计算文件的完整 64 位 SHA-256 哈希值 (读全量数据，绝不截断)"""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def resolve_manifest_path(path_value, manifest_path):
    """兼容 manifest 中的绝对路径或相对路径。"""
    path_value = str(path_value)
    if os.path.isabs(path_value) or os.path.exists(path_value):
        return path_value
    candidate = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), path_value)
    return candidate

def load_episode_manifest(manifest_path, sequence, data_dir, gt_dir, result_dir):
    """
    从冻结的 reference manifest 中按 exact sequence 取出一个 episode。
    本函数只消费 reference，并再次检查实际 path + SHA-256；不会重建映射。
    """
    df_manifest = pd.read_csv(manifest_path)
    required = {
        "sequence", "sequence_index", "frame_id",
        "rgb_path", "depth_path", "gt_path", "pred_path",
        "rgb_sha256", "depth_sha256", "gt_sha256", "pred_sha256",
        "association_method", "association_reference", "association_description",
    }
    missing = required - set(df_manifest.columns)
    if missing:
        raise ValueError(f"Master manifest 缺少字段: {sorted(missing)}")
    episode = df_manifest[df_manifest["sequence"] == sequence].copy()
    if len(episode) == 0:
        raise ValueError(f"Master manifest 中找不到 episode: {sequence}")
    episode["frame_id"] = episode["frame_id"].astype(int)
    episode["sequence_index"] = episode["sequence_index"].astype(int)
    if episode["frame_id"].duplicated().any():
        dup = episode.loc[episode["frame_id"].duplicated(), "frame_id"].tolist()
        raise ValueError(f"[{sequence}] manifest 存在重复 frame_id: {dup}")
    if episode["sequence_index"].duplicated().any():
        dup = episode.loc[episode["sequence_index"].duplicated(), "sequence_index"].tolist()
        raise ValueError(f"[{sequence}] manifest 存在重复 sequence_index: {dup}")
    episode = episode.sort_values("sequence_index", kind="stable").reset_index(drop=True)
    expected_indices = list(range(len(episode)))
    if episode["sequence_index"].tolist() != expected_indices:
        raise ValueError(
            f"[{sequence}] manifest sequence_index不连续: "
            f"expected={expected_indices[:10]}, actual={episode['sequence_index'].tolist()[:10]}"
        )
    methods = episode["association_method"].dropna().unique().tolist()
    if methods != ["official_ycbineoat_reference_sorted_index"]:
        raise ValueError(f"[{sequence}] manifest association_method不受支持: {methods}")
    episode["seq_idx"] = episode["sequence_index"]
    episode["frame_idx"] = episode["frame_id"]

    # RGB / depth 都严格使用 frozen reference manifest 当前行记录的路径。
    episode["rgb_path"] = episode["rgb_path"].map(
        lambda x: x if os.path.isabs(str(x))
        else os.path.normpath(os.path.join(data_dir, str(x)))
    )
    episode["depth_path"] = episode["depth_path"].map(
        lambda x: x if os.path.isabs(str(x))
        else os.path.normpath(os.path.join(data_dir, str(x)))
    )
    episode["gt_path"] = episode["gt_path"].map(
        lambda x: os.path.join(gt_dir, os.path.basename(str(x)))
    )
    episode["pred_path"] = episode["pred_path"].map(
        lambda x: os.path.join(result_dir, os.path.basename(str(x)))
    )

    for col in ["rgb_path", "depth_path", "gt_path", "pred_path"]:
        missing_paths = [p for p in episode[col].tolist() if not os.path.exists(p)]
        if missing_paths:
            raise FileNotFoundError(f"[{sequence}] {col} 中存在不存在的文件，例如: {missing_paths[0]}")
    for row in episode.itertuples(index=False):
        for artifact_type in ["rgb", "depth", "gt", "pred"]:
            path = getattr(row, f"{artifact_type}_path")
            expected = str(getattr(row, f"{artifact_type}_sha256")).lower()
            actual = compute_full_sha256(path)
            if actual != expected:
                raise ValueError(
                    f"[{sequence} frame {row.frame_id}] {artifact_type} SHA-256 mismatch: "
                    f"expected={expected}, actual={actual}, path={path}"
                )
    print(f"[{sequence}] manifest paths and SHA-256 values verified: {len(episode)} frames")
    return episode


def load_foundationpose_recovery_rgb(rgb_file):
    """
    Load the exact RGB file associated with the current frame by the
    frozen reference manifest.

    No basename matching and no directory guessing are performed.
    """
    rgb_file = os.path.abspath(str(rgb_file))

    if not os.path.isfile(rgb_file):
        raise FileNotFoundError(
            "[FoundationPose recovery] reference_manifest.csv 中当前帧对应的 "
            f"RGB 文件不存在:\n{rgb_file}"
        )

    rgb_bgr = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(
            f"[FoundationPose recovery] 无法读取 manifest RGB: {rgb_file}"
        )

    rgb_real = cv2.cvtColor(
        rgb_bgr,
        cv2.COLOR_BGR2RGB,
    )
    return rgb_real, rgb_file


# =====================================================================
# Shared pose-quality deployment features.
# These definitions intentionally mirror 2-risk_label.py.
# =====================================================================
cv_to_gl = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1],
], dtype=np.float64)

SHARED_FEATURE_COLUMNS = [
    "x1_norm",
    "x2_inlier_error",
    "x4_support_ratio",
    "x5_geometry_inconsistency",
]


def reliability_depth_residual(depth_real, pred_pose, scene, renderer, mesh_node):
    pose_render = cv_to_gl @ pred_pose
    scene.set_pose(mesh_node, pose_render)
    depth_render = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    valid = depth_render > 0
    if np.sum(valid) > 20:
        residual = np.abs(depth_render[valid] - depth_real[valid])
        return float(np.mean(residual) * 100.0)
    return 20.0


def reliability_inlier_ratio(valid_depth, Z_pred, Z_real):
    if valid_depth.sum() > 0:
        Z_pred_valid = Z_pred[valid_depth]
        Z_real_valid = Z_real[valid_depth]
        depth_diff = np.abs(Z_pred_valid - Z_real_valid) * 100.0
        return float(1.0 - np.mean(depth_diff < 2.0))
    return 1.0


def cad_depth_geometry_inconsistency(
    T_pose,
    depth_real,
    model_pts,
    K_mat,
    inlier_threshold_m=0.02,
    min_projected_pixels=20,
):
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
    u = np.rint(K_arr[0, 0] * pts_cam[:, 0] / z + K_arr[0, 2]).astype(np.int64)
    v = np.rint(K_arr[1, 1] * pts_cam[:, 1] / z + K_arr[1, 2]).astype(np.int64)
    inside = (
        (u >= 0) & (u < w) & (v >= 0) & (v < h) & np.isfinite(z)
    )
    if np.count_nonzero(inside) < min_projected_pixels:
        return 1.0
    u, v, z = u[inside], v[inside], z[inside]
    flat_idx = v * w + u
    cad_depth_flat = np.full(h * w, np.inf, dtype=np.float64)
    np.minimum.at(cad_depth_flat, flat_idx, z)
    projected_mask = np.isfinite(cad_depth_flat)
    projected_pixels = int(np.count_nonzero(projected_mask))
    if projected_pixels < min_projected_pixels:
        return 1.0
    idx = np.flatnonzero(projected_mask)
    cad_z = cad_depth_flat[idx]
    obs_z = depth.reshape(-1)[idx].astype(np.float64)
    valid_obs = np.isfinite(obs_z) & (obs_z > 0.05) & (obs_z < 5.0)
    if np.count_nonzero(valid_obs) == 0:
        return 1.0
    residual = np.abs(cad_z[valid_obs] - obs_z[valid_obs])
    inlier_count = int(np.count_nonzero(residual < inlier_threshold_m))
    return float(1.0 - inlier_count / float(projected_pixels))


def extract_pose_conditioned_features(
    T_pose,
    depth_real,
    model_pts,
    scene,
    renderer,
    mesh_node,
):
    x1 = reliability_depth_residual(
        depth_real, T_pose, scene, renderer, mesh_node
    )
    pts = np.asarray(model_pts, dtype=np.float64)
    R_p, t_p = T_pose[:3, :3], T_pose[:3, 3]
    pts_cam = (R_p @ pts.T).T + t_p
    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    h, w = depth_real.shape[:2]
    valid_z = Z > 1e-8
    u = np.zeros(len(Z), dtype=int)
    v = np.zeros(len(Z), dtype=int)
    u[valid_z] = np.round(K[0, 0] * X[valid_z] / Z[valid_z] + K[0, 2]).astype(int)
    v[valid_z] = np.round(K[1, 1] * Y[valid_z] / Z[valid_z] + K[1, 2]).astype(int)
    valid_bounds = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u_v, v_v, Z_p = u[valid_bounds], v[valid_bounds], Z[valid_bounds]
    if len(u_v) > 0:
        Z_real = depth_real[v_v, u_v]
        x2 = reliability_inlier_ratio(Z_real > 0, Z_p, Z_real)
        x4 = 1.0 - (np.sum(Z_real > 0.1) / (len(Z_real) + 1e-5))
    else:
        x2, x4 = 1.0, 1.0
    x5 = cad_depth_geometry_inconsistency(
        T_pose, depth_real, model_pts, K,
        inlier_threshold_m=0.02,
        min_projected_pixels=20,
    )
    return {"x1": float(x1), "x2": float(x2), "x4": float(x4), "x5": float(x5)}


def shared_feature_vector(features, d_obj_cm):
    return np.asarray([
        float(features["x1"]) / float(d_obj_cm),
        float(features["x2"]),
        float(features["x4"]),
        float(features["x5"]),
    ], dtype=np.float64)


def predict_shared_quality(features, d_obj_cm, scaler, regressor, calibrator):
    x = shared_feature_vector(features, d_obj_cm).reshape(1, -1)
    e_hat = max(float(regressor.predict(scaler.transform(x))[0]), 0.0)
    E_hat_cm = float(e_hat * float(d_obj_cm))
    p_risk = float(np.clip(calibrator.predict([E_hat_cm])[0], 0.0, 1.0))
    return e_hat, E_hat_cm, p_risk


def load_shared_artifacts(args):
    with open(args.shared_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("version") != "shared_pose_quality_v1":
        raise ValueError(f"Unsupported shared-quality config: {cfg.get('version')}")
    if getattr(args, "frozen_test", False):
        if cfg.get("training_mode") != "final_development_fit" or cfg.get("held_out_base") is not None:
            raise ValueError("Frozen test requires a final-development model, not a held-out fold")
        if args.test_base_seq in cfg.get("train_bases", []):
            raise ValueError("Development sequence cannot enter frozen new-sequence test")
        if cfg.get("manifest_sha256") != compute_full_sha256(args.manifest_path):
            raise ValueError("Frozen test must use the original training release's reference manifest")
    elif cfg.get("held_out_base") != args.test_base_seq:
        raise ValueError(
            "Shared model held-out base mismatch: "
            f"config={cfg.get('held_out_base')} eval={args.test_base_seq}"
        )
    cfg_train = set(cfg.get("train_bases", []))
    if cfg_train != set(args.train_seqs):
        raise ValueError(
            f"Shared model train bases mismatch: config={sorted(cfg_train)}, "
            f"eval={sorted(set(args.train_seqs))}"
        )
    if not np.isclose(float(cfg["risk_threshold_cm"]), float(args.risk_threshold)):
        raise ValueError("Shared config risk_threshold_cm mismatch")
    if not np.isclose(
        float(cfg["prior_advantage_margin_cm"]),
        float(args.prior_advantage_margin_cm),
    ):
        raise ValueError("Shared config prior_advantage_margin_cm mismatch")
    if cfg.get("feature_columns") != SHARED_FEATURE_COLUMNS:
        raise ValueError(
            f"Shared feature schema mismatch: {cfg.get('feature_columns')}"
        )

    # Verify frozen artifact bytes if hashes were written by 2-risk_label.py.
    for key, path in [
        ("scaler", args.shared_scaler_path),
        ("model", args.shared_model_path),
        ("calibrator", args.shared_calibrator_path),
    ]:
        expected = cfg.get(f"{key}_sha256")
        if getattr(args, "frozen_test", False) and expected is None:
            raise ValueError("Frozen config is missing artifact hash: " + key)
        if expected is not None:
            actual = compute_full_sha256(path)
            if actual != expected:
                raise ValueError(
                    f"Frozen {key} SHA-256 mismatch: expected={expected}, actual={actual}"
                )

    scaler = joblib.load(args.shared_scaler_path)
    regressor = joblib.load(args.shared_model_path)
    calibrator = joblib.load(args.shared_calibrator_path)
    p_risk_threshold = float(cfg["p_risk_threshold"])
    return cfg, scaler, regressor, calibrator, p_risk_threshold


def _safe_prob_metrics(labels, probs):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if len(labels) == 0:
        return {"auroc": np.nan, "auprc": np.nan, "brier": np.nan, "ece": np.nan}
    if len(np.unique(labels)) < 2:
        return {"auroc": np.nan, "auprc": np.nan,
                "brier": float(brier_score_loss(labels, probs)),
                "ece": float(compute_ece(probs, labels))}
    precision, recall, _ = precision_recall_curve(labels, probs)
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(auc(recall, precision)),
        "brier": float(brier_score_loss(labels, probs)),
        "ece": float(compute_ece(probs, labels)),
    }


def _latency_string(errors, start_index, threshold_cm=0.5):
    if start_index is None or not np.isfinite(start_index):
        return "N/A"
    start_index = int(start_index)
    if start_index < 0 or start_index >= len(errors):
        return "N/A"
    for i in range(start_index, len(errors)):
        if errors[i] < threshold_cm:
            return f"{i - start_index:.1f} frames"
    return "N/A (Failed)"


def build_eval_renderer(mesh_file):
    mesh = trimesh.load(mesh_file)
    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene()
    mesh_node = scene.add(render_mesh)
    camera = pyrender.IntrinsicsCamera(
        fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    scene.add(camera, pose=np.eye(4))
    renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
    return scene, renderer, mesh_node


def decision_inputs(variant, obs_error, prior_error, obs_risk, prior_risk):
    """Predeclared controls; GT never enters these decisions or frozen B5 code.

    simple: accept observations outside blackout, extrapolate own prior during
    blackout, use the SAME SAM2/FP/gate and prior fallback at blackout exit.
    Both ablations keep the trained predictor, geometry gate and state machine.
    """
    if variant == "simple":
        return 0.0, 0.0, 0.0, 0.0
    if variant == "no_absolute_gate":
        return obs_error, prior_error, 1.0, prior_risk
    if variant == "no_relative_advantage":
        return obs_error, obs_error, obs_risk, prior_risk
    if variant != "full":
        raise ValueError("Unknown policy variant: " + variant)
    return obs_error, prior_error, obs_risk, prior_risk


def evaluate_episode(
    args,
    result_dir,
    labels_df,
    model_pts,
    open3d_model,
    d_obj_cm,
    scene,
    renderer,
    mesh_node,
    scaler,
    regressor,
    calibrator,
    p_risk_threshold,
):
    last_name = os.path.basename(result_dir)
    test_df = None
    if not getattr(args, "frozen_test", False):
        test_df = labels_df[labels_df["sequence"] == last_name].copy()
        if len(test_df) == 0:
            raise ValueError(f"Label CSV contains no test sequence {last_name}")
        test_df["frame_id"] = test_df["frame_id"].astype(int)
        test_df = test_df.set_index("frame_id")

    df_manifest = load_episode_manifest(
        args.manifest_path,
        last_name,
        args.data_dir,
        args.gt_dir,
        result_dir,
    )
    if test_df is not None and set(df_manifest["frame_id"].astype(int)) != set(test_df.index.astype(int)):
        raise ValueError(f"[{last_name}] label/manifest frame IDs mismatch")
    if "base_sequence" in df_manifest and set(df_manifest["base_sequence"].astype(str)) != {args.test_base_seq}:
        raise ValueError("Manifest base_sequence and initial-mask base do not match")

    b1_errs, b2_errs, b3_errs, b4_errs, b5_errs, b6_errs = [], [], [], [], [], []
    b5_modes, matched_frames = [], []
    T_history2, T_history3, T_history4, T_history5, T_history6 = [], [], [], [], []
    revised_history = bool(getattr(b5_transition, "__globals__", {}).get("B5_POLICY_CONFIG"))
    if revised_history:
        from b5_revision import make_prior, advance_history
    b5_state = init_b5_state()
    recovery_record = None
    recovery_events = []
    quality_records = []
    label_consistency_failures = []

    for row in df_manifest.itertuples():
        i = int(row.seq_idx)
        frame_id = int(row.frame_idx)
        matched_frames.append(frame_id)
        T_obs = np.loadtxt(row.pred_path).reshape(4, 4)
        T_gt = np.loadtxt(row.gt_path).reshape(4, 4)
        depth_raw = cv2.imread(row.depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(row.depth_path)
        depth_real = depth_raw.astype(np.float32) / 1000.0
        # row.rgb_path is already resolved by load_episode_manifest. Preserve
        # that exact artifact/order for SAM2 while retaining existing RGB I/O.
        rgb_real, rgb_path = load_foundationpose_recovery_rgb(row.rgb_path)

        if i < 2:
            T_prior2 = T_prior3 = T_prior4 = T_prior5 = T_prior6 = T_obs
        else:
            T_prior2 = compute_se3_prior(T_history2[-1], T_history2[-2])
            T_prior3 = compute_se3_prior(T_history3[-1], T_history3[-2])
            T_prior4 = compute_se3_prior(T_history4[-1], T_history4[-2])
            if not revised_history:
                T_prior5 = compute_se3_prior(T_history5[-1], T_history5[-2])
            T_prior6 = compute_se3_prior(T_history6[-1], T_history6[-2])

        restart_zero_velocity = bool(revised_history and len(T_history5) == 1
                                     and b5_state.get("motion_history_restarted", False))
        if revised_history:
            T_prior5 = make_prior(T_history5, T_obs, b5_state, compute_se3_prior)

        quality_start = time.perf_counter()
        obs_features = extract_pose_conditioned_features(
            T_obs, depth_real, model_pts, scene, renderer, mesh_node
        )
        prior_features = extract_pose_conditioned_features(
            T_prior5, depth_real, model_pts, scene, renderer, mesh_node
        )
        e_obs_hat, E_obs_hat_cm, p_obs_risk = predict_shared_quality(
            obs_features, d_obj_cm, scaler, regressor, calibrator
        )
        e_prior_hat, E_prior_hat_cm, p_prior_risk = predict_shared_quality(
            prior_features, d_obj_cm, scaler, regressor, calibrator
        )
        quality_wall_ms = (time.perf_counter() - quality_start) * 1000.0

        # B1: observation only.
        E_obs_gt_cm = U.adi(T_obs, T_gt, open3d_model) * 100.0
        E_prior_gt_cm = U.adi(T_prior5, T_gt, open3d_model) * 100.0
        b1_errs.append(E_obs_gt_cm)

        # B2: fixed alpha.
        delta = se3_log_map(np.linalg.inv(T_prior2) @ T_obs)
        T_final2 = T_prior2 @ se3_exp_map(args.alpha * delta)
        T_history2.append(T_final2)
        b2_errs.append(U.adi(T_final2, T_gt, open3d_model) * 100.0)

        # B3: hard observation-support threshold (legacy baseline only).
        if obs_features["x4"] < 0.4:
            T_final3 = T_obs
        else:
            T_final3 = T_prior3
        T_history3.append(T_final3)
        b3_errs.append(U.adi(T_final3, T_gt, open3d_model) * 100.0)

        # B4: Huber innovation weighting.
        innovation = se3_log_map(np.linalg.inv(T_prior4) @ T_obs)
        r = np.linalg.norm(innovation)
        huber_delta = 0.1
        huber_alpha = 1.0 if r <= huber_delta else huber_delta / r
        T_huber = T_prior4 @ se3_exp_map(huber_alpha * innovation)
        T_history4.append(T_huber)
        b4_errs.append(U.adi(T_huber, T_gt, open3d_model) * 100.0)

        # B5: exact same shared-quality B5 transition as final label rollout.
        decision = decision_inputs(getattr(args, "policy_variant", "full"),
                                   E_obs_hat_cm, E_prior_hat_cm, p_obs_risk, p_prior_risk)
        transition_start = time.perf_counter()
        T_final, current_mode, b5_state, recovery_info = b5_transition(
            T_obs=T_obs,
            T_prior=T_prior5,
            support=obs_features["x4"],
            depth_real=depth_real,
            model_pts=model_pts,
            K=K,
            frame_index=i,
            frame_id=frame_id,
            state=b5_state,
            blackout_min_frames=args.blackout_min_frames,
            rgb_real=rgb_real,
            base_sequence=args.test_base_seq,
            ycbineoat_root=args.ycbineoat_root,
            mesh_file=os.path.abspath(args.foundationpose_mesh_file),
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
            E_obs_hat_cm=decision[0],
            E_prior_hat_cm=decision[1],
            p_obs_risk=decision[2],
            p_prior_risk=decision[3],
            p_risk_threshold=p_risk_threshold,
            prior_advantage_margin_cm=args.prior_advantage_margin_cm,
        )
        transition_wall_ms = (time.perf_counter() - transition_start) * 1000.0
        if revised_history:
            T_history5 = advance_history(T_history5, T_final, b5_state)
        else:
            T_history5.append(T_final)
        b5_error_current = U.adi(T_final, T_gt, open3d_model) * 100.0
        b5_errs.append(b5_error_current)
        b5_modes.append(current_mode)

        # Optional exact-consistency check against the final frozen label rollout.
        if not getattr(args, "frozen_test", False) and args.strict_label_rollout_check:
            stored = test_df.loc[frame_id]
            checks = {
                "E_obs_hat_cm": E_obs_hat_cm,
                "E_prior_hat_cm": E_prior_hat_cm,
                "p_obs_risk_rollout": p_obs_risk,
                "p_prior_risk_rollout": p_prior_risk,
            }
            for key, actual in checks.items():
                if key in stored.index:
                    expected = float(stored[key])
                    if not np.isclose(
                        expected, actual,
                        rtol=args.label_rollout_rtol,
                        atol=args.label_rollout_atol,
                    ):
                        label_consistency_failures.append(
                            (frame_id, key, expected, actual)
                        )
            if "rollout_mode" in stored.index and str(stored["rollout_mode"]) != current_mode:
                label_consistency_failures.append(
                    (frame_id, "rollout_mode", str(stored["rollout_mode"]), current_mode)
                )

        if recovery_info is not None:
            print(
            f"\n[Recovery] episode={last_name} frame={frame_id}"
            )
            print(
            "  trigger =",
            recovery_info.get("recovery_trigger")
            )
            print(
            "  raw =",
            recovery_info.get("raw_recovery_generated")
            )
            print(
            "  accepted =",
            recovery_info.get("accepted_recovery")
            )
            print(
            "  used =",
            recovery_info.get("recovery_used")
            )

            print(
            "  recovery_failure_reason =",
            recovery_info.get("recovery_failure_reason")
            )
            print(
            "  template2_match_reason =",
            recovery_info.get("template2_match_reason")
            )
            print(
            "  template2_candidate_count =",
            recovery_info.get("template2_candidate_count")
            )
            print(
            "  foundationpose_error =",
            recovery_info.get("foundationpose_error")
            )

            if recovery_info.get("recovery_trigger") == "blackout_exit":
                print("\n========== RECOVERY VALIDITY ==========")
                gate = (recovery_info.get("recovery_soft_evidence") or {}).get("gate", {})
                if gate:
                    print("gate_version:", gate.get("recovery_gate_version"))
                    print("gate_status:", gate.get("recovery_gate_status"))
                    print("acceptance_path:", gate.get("recovery_acceptance_path"))
                    print("decision_category:", gate.get("recovery_decision_category"))
                    print("occlusion / conflict / inlier fractions:",
                          gate.get("recovery_occlusion_fraction"), gate.get("recovery_conflict_fraction"),
                          gate.get("recovery_inlier_fraction"))
                    print("positive inliers / visible support / spatial bins:",
                          gate.get("recovery_inlier_pixels"), gate.get("recovery_visible_support_pixels"),
                          gate.get("recovery_inlier_spatial_bins"))
                    print("History jumps, legacy IoU/depth residual below are DIAGNOSTICS ONLY.")
                print(
                    "rejection_reasons:",
                    recovery_info.get("recovery_rejection_reasons")
                )
                print(
                    "IoU:",
                    recovery_info.get("recovery_cad_mask_iou")
                )
                print(
                    "coverage:",
                    recovery_info.get("recovery_cad_coverage")
                )
                print(
                    "center_error_norm:",
                    recovery_info.get(
                        "recovery_reprojection_center_error_norm"
                    )
                )
                print(
                    "depth_residual_m:",
                    recovery_info.get(
                        "recovery_rendered_depth_median_residual_m"
                    )
                )
                print(
                    "depth_support:",
                    recovery_info.get(
                        "recovery_rendered_depth_support"
                    )
                )
                print(
                    "template_score_margin:",
                    recovery_info.get(
                        "recovery_template_score_margin"
                    )
                )
                print(
                    "translation_jump_m:",
                    recovery_info.get(
                        "recovery_translation_jump_m"
                    )
                )
                print(
                    "rotation_jump_deg:",
                    recovery_info.get(
                        "recovery_rotation_jump_deg"
                    )
                )
                print("=======================================\n")



        if recovery_info is not None:
            raw_generated = bool(recovery_info.get("raw_recovery_generated", False))
            accepted = bool(recovery_info.get("accepted_recovery", False))
            used = bool(recovery_info.get("recovery_used", False))
            raw_pose = recovery_info.get("T_raw_recovery")
            accepted_pose = recovery_info.get("T_accepted_recovery")
            raw_error_cm = (
                U.adi(raw_pose, T_gt, open3d_model) * 100.0
                if raw_generated and raw_pose is not None else np.nan
            )
            accepted_error_cm = (
                U.adi(accepted_pose, T_gt, open3d_model) * 100.0
                if accepted and accepted_pose is not None else np.nan
            )
            b1_recovery_error_cm = b1_errs[-1]
            recovery_event = {
                "episode": last_name,
                "recovery_frame": frame_id,
                "recovery_trigger": recovery_info.get("recovery_trigger"),
                "recovery_index": i,
                "sam2_called": bool(recovery_info.get("sam2_called", False)),
                "sam2_cache_hit": bool(recovery_info.get("sam2_cache_hit", False)),
                "fallback_prior_error_cm": float(E_prior_gt_cm),
                "failure_reason": recovery_info.get("recovery_failure_reason") or "",
                "rejection_reasons": json.dumps(recovery_info.get("recovery_rejection_reasons", []), ensure_ascii=False),
                "raw_recovery_generated": raw_generated,
                "raw_recovery_error_cm": float(raw_error_cm) if np.isfinite(raw_error_cm) else np.nan,
                "accepted_recovery": accepted,
                "accepted_recovery_error_cm": float(accepted_error_cm) if np.isfinite(accepted_error_cm) else np.nan,
                "recovery_used": used,
                "B1_error_cm": float(b1_recovery_error_cm),
                "B5_operational_error_cm": float(b5_error_current),
                "raw_minus_B1_cm": float(raw_error_cm - b1_recovery_error_cm) if np.isfinite(raw_error_cm) else np.nan,
                "accepted_minus_B1_cm": float(accepted_error_cm - b1_recovery_error_cm) if np.isfinite(accepted_error_cm) else np.nan,
                "operational_minus_B1_cm": float(b5_error_current - b1_recovery_error_cm),
            }
            recovery_events.append(recovery_event)
            gate_diag = recovery_info.get("recovery_soft_evidence") or {}
            gate_diag = gate_diag.get("gate", {})
            recovery_event["recovery_gate_diagnostics"] = json.dumps(gate_diag, ensure_ascii=False)
            for key in ("recovery_gate_version", "recovery_gate_status", "recovery_acceptance_path",
                        "recovery_decision_category", "recovery_inlier_fraction",
                        "recovery_conflict_fraction", "recovery_occlusion_fraction",
                        "recovery_visible_support_pixels", "recovery_inlier_pixels",
                        "recovery_visible_mask_explained", "recovery_visible_cad_coverage",
                        "recovery_inlier_spatial_bins", "recovery_global_inlier_fraction"):
                recovery_event[key] = gate_diag.get(key)
            # Preserve the original single-event export for existing consumers.
            if recovery_event["recovery_trigger"] == "blackout_exit" and recovery_record is None:
                recovery_record = recovery_event

        # B6 oracle upper bound with independent recursive history.
        err_obs = U.adi(T_obs, T_gt, open3d_model)
        err_prior = U.adi(T_prior6, T_gt, open3d_model)
        T_final6 = T_obs if err_obs < err_prior else T_prior6
        T_history6.append(T_final6)
        b6_errs.append(U.adi(T_final6, T_gt, open3d_model) * 100.0)

        quality_records.append({
            "episode": last_name,
            "frame_id": frame_id,
            "E_obs_cm": float(E_obs_gt_cm),
            "E_prior_cm": float(E_prior_gt_cm),
            "E_obs_hat_cm": float(E_obs_hat_cm),
            "E_prior_hat_cm": float(E_prior_hat_cm),
            "e_obs_hat_norm": float(e_obs_hat),
            "e_prior_hat_norm": float(e_prior_hat),
            "p_obs_risk": float(p_obs_risk),
            "p_prior_risk": float(p_prior_risk),
            "delta_E_gt_cm": float(E_prior_gt_cm - E_obs_gt_cm),
            "delta_E_hat_cm": float(E_prior_hat_cm - E_obs_hat_cm),
            "selected_mode": current_mode,
            "policy_variant": getattr(args, "policy_variant", "full"),
            "policy_version": b5_state.get("policy_version", "legacy_frozen"),
            "fusion_alpha": b5_state.get("last_fusion_alpha"),
            "forced_streak_reset": int(bool(b5_state.get("last_forced_streak_reset"))),
            "motion_history_reset": int(bool(b5_state.get("reset_motion_history"))),
            "restart_zero_velocity_prior": int(restart_zero_velocity),
            "output_uncertain": b5_state.get("output_uncertain"),
            "both_candidates_bad": int(E_obs_gt_cm > args.risk_threshold and E_prior_gt_cm > args.risk_threshold),
            "local_candidate_oracle_cm": float(min(E_obs_gt_cm, E_prior_gt_cm)),
            "is_blackout": int(b5_state.get("is_depth_blackout", False)),
            "recovery_attempted": int(recovery_info is not None),
            "sam2_cache_hit": int(bool(recovery_info and recovery_info.get("sam2_cache_hit", False))),
            "quality_wall_ms": quality_wall_ms,
            "transition_wall_ms": transition_wall_ms,
            "policy_wall_ms": transition_wall_ms + (0.0 if getattr(args, "policy_variant", "full") == "simple" else quality_wall_ms),
        })

    if label_consistency_failures:
        preview = label_consistency_failures[:10]
        raise ValueError(f"[{last_name}] strict label/rollout mismatch: {preview}")

    blackout_intervals = []
    for interval in b5_state.get("blackout_intervals", []):
        rec = {"episode": last_name}
        rec.update(interval)
        blackout_intervals.append(rec)

    recovery_start = None
    if blackout_intervals:
        recovery_start = blackout_intervals[0]["recovery_index"]
    latency_scores = [
        _latency_string(b1_errs, recovery_start),
        _latency_string(b2_errs, recovery_start),
        _latency_string(b3_errs, recovery_start),
        _latency_string(b4_errs, recovery_start),
        _latency_string(b5_errs, recovery_start),
        _latency_string(b6_errs, recovery_start),
    ]
    false_triggers = ["N/A"] * 6  # False triggers are not measured here.

    # Per-frame deployment log.
    qdf = pd.DataFrame(quality_records)
    qdf["error_b1_obs_cm"] = b1_errs
    qdf["error_b2_cm"] = b2_errs
    qdf["error_b3_cm"] = b3_errs
    qdf["error_b4_cm"] = b4_errs
    qdf["error_b5_ours_cm"] = b5_errs
    qdf["error_b6_oracle_cm"] = b6_errs
    log_path = f"./checkpoint2_per_frame_{last_name}_log_threshold{args.risk_threshold}.csv"
    qdf.to_csv(log_path, index=False)

    if blackout_intervals:
        start = int(blackout_intervals[0]["blackout_start_index"])
        end = int(blackout_intervals[0]["recovery_index"])
        plt.figure(figsize=(10, 5))
        plt.plot(b1_errs, label="B1: Obs-Only")
        plt.plot(b5_errs, label="Policy: " + getattr(args, "policy_variant", "full"))
        plt.axvspan(start, end, alpha=0.3, label="Blackout")
        plt.xlabel("Frame index")
        plt.ylabel("ADD-S error (cm)")
        plt.legend()
        plt.grid(True, linestyle="--")
        plt.savefig(
            f"trajectory_recovery_plot_{last_name}_threshold{args.risk_threshold}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    adds_scores = [calc_auc(x) for x in [b1_errs, b2_errs, b3_errs, b4_errs, b5_errs, b6_errs]]
    fail_rates = [float(np.mean(np.asarray(x) > 2.0) * 100.0) for x in [b1_errs, b2_errs, b3_errs, b4_errs, b5_errs, b6_errs]]
    recovery_metrics = build_recovery_decomposition(
        last_name, blackout_intervals, recovery_events,
        dict(zip(("B1", "B2", "B3", "B4", "B5"),
                 (b1_errs, b2_errs, b3_errs, b4_errs, b5_errs))),
        threshold_cm=args.risk_threshold,
    )
    return adds_scores, fail_rates, latency_scores, false_triggers, recovery_record, blackout_intervals, quality_records, recovery_metrics


def build_recovery_decomposition(episode, intervals, events, errors_by_method,
                                 threshold_cm=1.0, window_frames=60):
    """One row per blackout exit, including rejected/missing recoveries.
    Window includes the exit frame. GT is used only for offline evaluation.
    """
    rows = []
    for interval in intervals:
        start = int(interval["recovery_index"])
        event = next((e for e in events
                      if e["recovery_index"] == start
                      and e["recovery_trigger"] == "blackout_exit"), None)
        row = {
            "row_type": "event", "episode": episode,
            "recovery_index": start, "recovery_frame": interval["recovery_frame"],
            "threshold_cm": threshold_cm, "window_requested_frames": window_frames,
            "blackout_event_count": 1, "trigger_count": int(event is not None),
            "sam2_requested_count": int(bool(event and event["sam2_called"])),
            "sam2_cache_hit_count": int(bool(event and event["sam2_cache_hit"])),
        }
        generated = bool(event and event["raw_recovery_generated"])
        accepted = bool(event and event["accepted_recovery"])
        used = bool(event and event["recovery_used"])
        raw_error = event["raw_recovery_error_cm"] if generated else np.nan
        accepted_error = event["accepted_recovery_error_cm"] if accepted else np.nan
        prior_error = event["fallback_prior_error_cm"] if event else np.nan
        if generated and not np.isfinite(raw_error):
            raw_error = np.inf
        if accepted and not np.isfinite(accepted_error):
            accepted_error = np.inf
        if event and not np.isfinite(prior_error):
            prior_error = np.inf
        row.update({
            "generated_count": int(generated), "accepted_count": int(accepted),
            "used_count": int(used), "rejected_count": int(generated and not accepted),
            "not_generated_count": int(not generated),
            "raw_good_count": int(generated and raw_error <= threshold_cm),
            "accepted_good_count": int(accepted and accepted_error <= threshold_cm),
            "raw_recovery_error_cm": raw_error,
            "accepted_recovery_error_cm": accepted_error,
            "fallback_prior_error_cm": prior_error,
            "raw_minus_fallback_prior_cm": raw_error - prior_error if generated else np.nan,
            "outcome": ("accepted" if accepted else "rejected" if generated else
                        "not_generated" if event else "not_triggered"),
            "failure_reason": event["failure_reason"] if event else "no_recovery_event",
            "rejection_reasons": event["rejection_reasons"] if event else "",
        })
        if event:
            for key, value in event.items():
                if key.startswith("recovery_") and key not in row:
                    row[key] = value
        for method, errors in errors_by_method.items():
            errors = np.asarray(errors, dtype=np.float64)
            # Invalid outputs must not be silently removed or counted as accurate.
            errors = np.where(np.isfinite(errors), errors, np.inf)
            window = errors[start:start + window_frames]
            n = len(window)
            row[method + "_window_frames"] = n
            row[method + "_window_complete"] = int(n == window_frames)
            row[method + "_window_failure_percent"] = (
                float(np.mean(window > threshold_cm) * 100.0) if n else np.nan)
            row[method + "_window_auc_percent"] = calc_auc(window) if n else np.nan
            row[method + "_full_sequence_auc_percent"] = calc_auc(errors) if len(errors) else np.nan
            # Latency is confirmed only after five consecutive good frames.
            # Search the same fixed post-exit window; never encode no-success as zero.
            good = window <= threshold_cm
            confirmation = next((j for j in range(4, n) if np.all(good[j-4:j+1])), None)
            row[method + "_latency_success"] = int(confirmation is not None)
            row[method + "_latency_confirmed_frames"] = confirmation if confirmation is not None else np.nan
            row[method + "_latency_censored"] = int(confirmation is None)
            row[method + "_latency_followup_frames"] = n
            row[method + "_latency_required_good_frames"] = 5
        rows.append(row)
    return rows


def summarize_recovery_decomposition(rows, base_sequence):
    """Descriptive within-base event summary, NOT an independent-object estimate.
    Incomplete windows remain in event rows but are excluded from fixed-window
    summary metrics; their counts are explicitly reported.
    """
    summary = {
        "row_type": "summary", "episode": base_sequence,
        "aggregation": "within_base_event_mean_not_independent_samples",
        "threshold_cm": rows[0]["threshold_cm"] if rows else np.nan,
        "window_requested_frames": 60,
    }
    counts = ("blackout_event_count", "trigger_count", "sam2_requested_count",
              "sam2_cache_hit_count", "generated_count", "accepted_count",
              "used_count", "rejected_count", "not_generated_count",
              "raw_good_count", "accepted_good_count")
    for key in counts:
        summary[key] = sum(row[key] for row in rows)
    for output, numerator, denominator in (
        ("raw_accuracy_percent", "raw_good_count", "generated_count"),
        ("accepted_accuracy_percent", "accepted_good_count", "accepted_count"),
        ("acceptance_percent", "accepted_count", "generated_count"),
        ("rejection_percent", "rejected_count", "generated_count"),
    ):
        summary[output] = (100.0 * summary[numerator] / summary[denominator]
                           if summary[denominator] else np.nan)
    for key, count_key in (("raw_recovery_error_cm", "generated_count"),
                           ("accepted_recovery_error_cm", "accepted_count")):
        values = [r[key] for r in rows if r[count_key]]
        summary[key] = float(np.mean(values)) if values else np.nan
    for method in ("B1", "B2", "B3", "B4", "B5"):
        complete = [r for r in rows if r[method + "_window_complete"]]
        summary[method + "_complete_window_count"] = len(complete)
        summary[method + "_incomplete_window_count"] = len(rows) - len(complete)
        for suffix in ("_window_failure_percent", "_window_auc_percent"):
            summary[method + suffix] = (
                float(np.mean([r[method + suffix] for r in complete]))
                if complete else np.nan)
        # Each original evaluation episode contributes only once here.
        per_episode = {r["episode"]: r[method + "_full_sequence_auc_percent"] for r in rows}
        summary[method + "_full_sequence_auc_percent"] = (
            float(np.mean(list(per_episode.values()))) if per_episode else np.nan)
    return summary



def save_reliability_diagram(probs, labels, title, xlabel, ylabel, path, ece):
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    boundaries = np.linspace(0, 1, 11)
    accs, confs = [], []
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    for i in range(10):
        mask = ((probs >= boundaries[i]) if i == 0 else (probs > boundaries[i])) & (probs <= boundaries[i + 1])
        if np.any(mask):
            accs.append(np.mean(labels[mask]))
            confs.append(np.mean(probs[mask]))
    plt.plot(confs, accs, 's-', linewidth=2, label=f'Shared calibrator (ECE={ece:.3f})')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def bind_frozen_quality_implementation():
    """Reuse training-snapshot feature extraction/K exactly, without calling its main."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2-risk_label.py")
    spec = importlib.util.spec_from_file_location("frozen_quality_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def extract(T_pose, depth_real, model_pts, scene, renderer, mesh_node):
        return module.extract_pose_conditioned_features(
            T_pose, depth_real, 0, [model_pts], [scene], [renderer], [mesh_node], include_support=True)
    global K, cv_to_gl, SHARED_FEATURE_COLUMNS, extract_pose_conditioned_features, predict_shared_quality
    K, cv_to_gl = module.K, module.cv_to_gl
    SHARED_FEATURE_COLUMNS = module.SHARED_FEATURE_COLUMNS
    extract_pose_conditioned_features = extract
    predict_shared_quality = module.predict_shared_quality


def main(args):
    if getattr(args, "frozen_test", False):
        bind_frozen_quality_implementation()
    np.random.seed(args.seed)
    try:
        o3d.utility.random.seed(args.seed)
    except Exception:
        pass

    labels_df = None if getattr(args, "frozen_test", False) else pd.read_csv(args.csv_path)
    required = {
        "sequence", "frame_id", "E_obs_cm", "E_prior_cm",
        "E_obs_hat_cm", "E_prior_hat_cm",
        "p_obs_risk_rollout", "p_prior_risk_rollout", "rollout_mode",
    }
    missing = required - set(labels_df.columns) if labels_df is not None else set()
    if missing:
        raise ValueError(
            f"Label CSV missing shared-quality columns {sorted(missing)}; "
            "regenerate it with the new 2-risk_label.py."
        )

    cfg, scaler, regressor, calibrator, p_risk_threshold = load_shared_artifacts(args)
    print("\n[Frozen shared pose-quality model]")
    print("held-out base       :", cfg["held_out_base"])
    print("train bases         :", cfg["train_bases"])
    print("p_risk_threshold    :", p_risk_threshold)
    print("risk threshold (cm) :", cfg["risk_threshold_cm"])
    print("advantage margin cm :", cfg["prior_advantage_margin_cm"])
    if getattr(args, "preflight_only", False):
        print("Frozen feature implementation and model imports validated; no inference executed.")
        return

    model_pts = np.loadtxt(args.point_path, dtype=np.float64).reshape(-1, 3)
    open3d_model = U.toOpen3dCloud(
        model_pts, colors=np.zeros(model_pts.shape, dtype=np.float64)
    )
    d_obj_cm = float(
        np.linalg.norm(np.max(model_pts, axis=0) - np.min(model_pts, axis=0)) * 100.0
    )
    scene, renderer, mesh_node = build_eval_renderer(args.foundationpose_mesh_file)

    baseline_names = [
        "B1: Obs-Only se(3)-TrackNet",
        "B2: Fixed-Alpha (0.5) Interpolation",
        "B3: Hard Depth Threshold",
        "B4: Robust Huber Weighting",
        "B5: Proposed Shared Pose-Quality Policy",
        "B6: Recursive Obs/Prior Oracle (diagnostic, not a global upper bound)",
    ]
    baseline_names[4] = "B5 slot: " + getattr(args, "policy_variant", "full")

    ADDS, FAIL, LATENCY, TRIGGER = {}, {}, {}, {}
    RECOVERY, BLACKOUT = {}, {}
    all_quality_records = []
    all_recovery_metrics = []

    for result_dir in args.result_dir:
        ep = os.path.basename(result_dir)
        print(f"\n=== Evaluating {ep} ===")
        (
            adds, fails, lats, triggers,
            recovery_record, blackout_intervals, quality_records, recovery_metrics,
        ) = evaluate_episode(
            args, result_dir, labels_df,
            model_pts, open3d_model, d_obj_cm,
            scene, renderer, mesh_node,
            scaler, regressor, calibrator, p_risk_threshold,
        )
        ADDS[ep], FAIL[ep], LATENCY[ep], TRIGGER[ep] = adds, fails, lats, triggers
        RECOVERY[ep], BLACKOUT[ep] = recovery_record, blackout_intervals
        all_quality_records.extend(quality_records)
        all_recovery_metrics.extend(recovery_metrics)

    qdf = pd.DataFrame(all_quality_records)
    # Base-specific filename prevents run.sh's subsequent folds overwriting it.
    recovery_decomposition_path = (
        f"./checkpoint2_recovery_decomposition_{args.test_base_seq}"
        f"_threshold{args.risk_threshold}.csv"
    )
    recovery_summary = summarize_recovery_decomposition(all_recovery_metrics, args.test_base_seq)
    pd.DataFrame(all_recovery_metrics + [recovery_summary]).to_csv(
        recovery_decomposition_path, index=False, encoding="utf-8-sig")
    print("recovery decomposition:", recovery_decomposition_path)
    obs_labels = (qdf["E_obs_cm"].values > args.risk_threshold).astype(int)
    prior_labels = (qdf["E_prior_cm"].values > args.risk_threshold).astype(int)
    obs_metrics = _safe_prob_metrics(obs_labels, qdf["p_obs_risk"].values)
    prior_metrics = _safe_prob_metrics(prior_labels, qdf["p_prior_risk"].values)

    obs_mae = float(mean_absolute_error(qdf["E_obs_cm"], qdf["E_obs_hat_cm"]))
    prior_mae = float(mean_absolute_error(qdf["E_prior_cm"], qdf["E_prior_hat_cm"]))
    obs_med = float(np.median(np.abs(qdf["E_obs_cm"] - qdf["E_obs_hat_cm"])))
    prior_med = float(np.median(np.abs(qdf["E_prior_cm"] - qdf["E_prior_hat_cm"])))
    obs_spear = float(spearmanr(qdf["E_obs_cm"], qdf["E_obs_hat_cm"]).statistic)
    prior_spear = float(spearmanr(qdf["E_prior_cm"], qdf["E_prior_hat_cm"]).statistic)

    margin = float(args.prior_advantage_margin_cm)
    non_tie = np.abs(qdf["delta_E_gt_cm"].values) > margin
    pair_auroc = np.nan
    pair_bal_acc = np.nan
    pair_order_acc = np.nan
    if np.count_nonzero(non_tie) > 1:
        pair_y = (qdf.loc[non_tie, "delta_E_gt_cm"].values > margin).astype(int)
        pair_score = qdf.loc[non_tie, "delta_E_hat_cm"].values.astype(float)
        if len(np.unique(pair_y)) == 2:
            pair_auroc = float(roc_auc_score(pair_y, pair_score))
            pair_pred = (pair_score > 0.0).astype(int)
            pair_bal_acc = float(balanced_accuracy_score(pair_y, pair_pred))
            pair_order_acc = float(np.mean(pair_pred == pair_y))

    # Probability/regression/ranking metrics.
    metric_rows = [
        {"Metric": "Shared error MAE_obs_cm", "Value": obs_mae},
        {"Metric": "Shared error MedianAE_obs_cm", "Value": obs_med},
        {"Metric": "Shared error Spearman_obs", "Value": obs_spear},
        {"Metric": "AUROC_obs_abs_risk", "Value": obs_metrics["auroc"]},
        {"Metric": "AUPRC_obs_abs_risk", "Value": obs_metrics["auprc"]},
        {"Metric": "Brier_obs_abs_risk", "Value": obs_metrics["brier"]},
        {"Metric": "ECE_obs_abs_risk", "Value": obs_metrics["ece"]},
        {"Metric": "Shared error MAE_prior_cm", "Value": prior_mae},
        {"Metric": "Shared error MedianAE_prior_cm", "Value": prior_med},
        {"Metric": "Shared error Spearman_prior", "Value": prior_spear},
        {"Metric": "AUROC_prior_abs_risk", "Value": prior_metrics["auroc"]},
        {"Metric": "AUPRC_prior_abs_risk", "Value": prior_metrics["auprc"]},
        {"Metric": "Brier_prior_abs_risk", "Value": prior_metrics["brier"]},
        {"Metric": "ECE_prior_abs_risk", "Value": prior_metrics["ece"]},
        {"Metric": "Pairwise_AUROC_prior_worse", "Value": pair_auroc},
        {"Metric": "Pairwise_balanced_accuracy", "Value": pair_bal_acc},
        {"Metric": "Pairwise_ordering_accuracy", "Value": pair_order_acc},
        {"Metric": "p_risk_threshold_shared", "Value": p_risk_threshold},
        {"Metric": "prior_advantage_margin_cm", "Value": margin},
    ]
    metrics_path = f"./checkpoint2_probability_calibration_metrics_threshold{args.risk_threshold}.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)

    save_reliability_diagram(
        qdf["p_obs_risk"], obs_labels,
        "Reliability Diagram for Observation Absolute Risk",
        "Predicted P(Observation Risk)", "Empirical Risk Frequency",
        f"reliability_diagram_observation_risk_threshold{args.risk_threshold}.png",
        obs_metrics["ece"],
    )
    save_reliability_diagram(
        qdf["p_prior_risk"], prior_labels,
        "Reliability Diagram for Prior Absolute Risk",
        "Predicted P(Prior Risk)", "Empirical Risk Frequency",
        f"reliability_diagram_prior_risk_threshold{args.risk_threshold}.png",
        prior_metrics["ece"],
    )

    # Episode summary.
    summary_rows = []
    for b_idx, b_name in enumerate(baseline_names):
        row = {"Baseline / Method": b_name}
        vals = [ADDS[ep][b_idx] for ep in ADDS]
        for ep in ADDS:
            row[f"ADD-S ({ep})"] = f"{ADDS[ep][b_idx]:.3f}%"
            row[f"Failure Rate ({ep})"] = f"{FAIL[ep][b_idx]:.3f}%"
            row[f"Legacy latency <0.5cm single-frame ({ep})"] = LATENCY[ep][b_idx]
            row[f"False Triggers ({ep})"] = TRIGGER[ep][b_idx]
        row["Mean ADD-S (%)"] = f"{np.mean(vals):.3f}%"
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    first_ep = os.path.basename(args.result_dir[0])
    summary_path = f"./checkpoint2_full_metrics_episode_summary_{first_ep}_threshold{args.risk_threshold}.csv"
    summary_df.to_csv(summary_path, index=False)

    blackout_eps = [ep for ep in ADDS if "black" in ep.lower()]
    interval_rows = []
    for ep in blackout_eps:
        ints = BLACKOUT.get(ep, [])
        if len(ints) != 1:
            raise RuntimeError(f"[{ep}] expected exactly one blackout interval, got {len(ints)}")
        interval_rows.extend(ints)
    blackout_path = f"./checkpoint2_blackout_frame_intervals_threshold{args.risk_threshold}.csv"
    pd.DataFrame(interval_rows).to_csv(blackout_path, index=False)

    # Paired AUC.
    paired_rows, diffs = [], []
    for ep in blackout_eps:
        b1, b5 = float(ADDS[ep][0]), float(ADDS[ep][4])
        diff = b5 - b1
        diffs.append(diff)
        paired_rows.append({
            "episode": ep,
            "AUC_B1_percent": b1,
            "AUC_B5_percent": b5,
            "Delta_B5_minus_B1_percentage_points": diff,
        })
    mean_diff, low, high = compute_paired_bootstrap_ci(
        diffs, n_bootstraps=args.bootstrap_samples, seed=args.seed
    )
    paired_rows.append({
        "episode": "PAIRED_MEAN",
        "AUC_B1_percent": np.nan,
        "AUC_B5_percent": np.nan,
        "Delta_B5_minus_B1_percentage_points": mean_diff,
        "Paired_95CI_low": low,
        "Paired_95CI_high": high,
    })
    pd.DataFrame(paired_rows).to_csv(
        f"./checkpoint2_paired_auc_B5_vs_B1_threshold{args.risk_threshold}.csv",
        index=False,
    )


    compact_cols = [
        "episode", "recovery_frame", "recovery_trigger",
        "raw_recovery_generated", "raw_recovery_error_cm",
        "accepted_recovery", "accepted_recovery_error_cm",
        "recovery_used", "B1_error_cm", "B5_operational_error_cm",
        "raw_minus_B1_cm", "accepted_minus_B1_cm", "operational_minus_B1_cm",
    ]
    recovery_rows = []
    for ep in blackout_eps:
        rec = RECOVERY.get(ep)
        if rec is None:
            rec = {
                "episode": ep,
                "recovery_frame": np.nan,
                "recovery_trigger": "blackout_exit_not_recorded",
                "raw_recovery_generated": False,
                "raw_recovery_error_cm": np.nan,
                "accepted_recovery": False,
                "accepted_recovery_error_cm": np.nan,
                "recovery_used": False,
                "B1_error_cm": np.nan,
                "B5_operational_error_cm": np.nan,
                "raw_minus_B1_cm": np.nan,
                "accepted_minus_B1_cm": np.nan,
                "operational_minus_B1_cm": np.nan,
            }
        recovery_rows.append({k: rec.get(k, np.nan) for k in compact_cols})
    recovery_path = f"./checkpoint2_paired_recovery_B5_vs_B1_threshold{args.risk_threshold}.csv"
    pd.DataFrame(recovery_rows, columns=compact_cols).to_csv(recovery_path, index=False)

    print("\n=== Shared pose-quality evaluation complete ===")
    print("metrics          :", metrics_path)
    print("episode summary  :", summary_path)
    print("blackout intervals:", blackout_path)
    print("paired recovery  :", recovery_path)
    print(f"Pairwise AUROC prior-worse = {pair_auroc:.4f}")
    print(f"Obs risk AUROC            = {obs_metrics['auroc']:.4f}")
    print(f"Prior risk AUROC          = {prior_metrics['auroc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen_test', action='store_true', help='No test label CSV; final model only; no fitting')
    parser.add_argument('--preflight_only', action='store_true', help='Load frozen features/artifacts only; no rendering or rollout')
    parser.add_argument('--policy_variant', choices=['full', 'simple', 'no_absolute_gate', 'no_relative_advantage'], default='full')
    parser.add_argument('--csv_path', type=str, default="./per_frame_label_threshold1.0.csv")
    parser.add_argument('--manifest_path', type=str, default="./reference_manifest_all27.csv")
    parser.add_argument('--result_dir', nargs='+', type=str, default=[
        "./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10",
        "./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_2",
        "./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_3",
        "./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_4",
        "./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_5",
    ])
    parser.add_argument('--gt_dir', type=str, default="./datasets/YCBInEOAT/bleach_hard_00_03_chaitanya/annotated_poses")
    parser.add_argument('--point_path', type=str, default="./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/points.xyz")   #021_bleach_cleanser, 006_mustard_bottle
    parser.add_argument('--train_seqs', nargs='+', default=["mustard0", "bleach0"])
    parser.add_argument('--test_base_seq', type=str, default="bleach_hard_00_03_chaitanya")
    parser.add_argument('--data_dir', type=str, default="./datasets/YCBInEOAT_Corrupted")
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--risk_threshold', type=float, default=1.0)
    parser.add_argument(
        '--prior_advantage_margin_cm', type=float, default=0.1,
    )
    parser.add_argument('--blackout_min_frames', type=int, default=10)
    parser.add_argument('--ycbineoat_root', type=str, default="./datasets/YCBInEOAT")

    parser.add_argument('--foundationpose_python', type=str, default="/home/wyg/anaconda3/envs/foundationpose/bin/python")
    parser.add_argument('--foundationpose_dir', type=str, default="/home/wyg/FoundationPose")
    parser.add_argument('--foundationpose_mesh_file', type=str, default="./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/textured.obj")
    parser.add_argument('--foundationpose_refiner_weight', type=str, default="/home/wyg/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth")
    parser.add_argument('--foundationpose_refine_iter', type=int, default=5)

    parser.add_argument('--sam2_python', type=str, default="/home/wyg/anaconda3/envs/sam2/bin/python")
    parser.add_argument('--sam2_dir', type=str, default="/home/wyg/sam2")
    parser.add_argument('--sam2_config', type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument('--sam2_checkpoint', type=str, default="/home/wyg/sam2/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument('--sam2_cache_root', type=str, default="./sam2_recovery_cache")

    parser.add_argument('--shared_model_path', type=str, default="./shared_pose_quality_model.joblib")
    parser.add_argument('--shared_scaler_path', type=str, default="./shared_pose_quality_scaler.joblib")
    parser.add_argument('--shared_calibrator_path', type=str, default="./shared_risk_calibrator.joblib")
    parser.add_argument('--shared_config_path', type=str, default="./shared_quality_config.json")

    parser.add_argument('--strict_label_rollout_check', action='store_true', default=True)
    parser.add_argument('--label_rollout_rtol', type=float, default=1e-5)
    parser.add_argument('--label_rollout_atol', type=float, default=1e-5)
    parser.add_argument('--bootstrap_samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.policy_variant != 'full' and not args.frozen_test:
        parser.error('Policy controls require --frozen_test')
    main(args)
