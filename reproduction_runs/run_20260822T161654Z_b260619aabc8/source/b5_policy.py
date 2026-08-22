import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R_sci

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
        "blackout_intervals": []
    }

def actual_recovery_action(current_depth_real, model_pts_3d, K):
    y_idx, x_idx = np.where((current_depth_real > 0.5) & (current_depth_real < 1.2))
    if len(y_idx) < 300:
        print("[Recovery] valid depth points are insufficient.")
        return None, False
    z = current_depth_real[y_idx, x_idx]
    x = (x_idx - K[0, 2]) * z / K[0, 0]
    y = (y_idx - K[1, 2]) * z / K[1, 1]
    scene_pts = np.vstack([x, y, z]).T

    pcd_scene = o3d.geometry.PointCloud()
    pcd_scene.points = o3d.utility.Vector3dVector(scene_pts)
    _, inliers = pcd_scene.segment_plane(
        distance_threshold=0.015, ransac_n=3, num_iterations=500
    )
    pcd_objects = pcd_scene.select_by_index(inliers, invert=True)
    if len(pcd_objects.points) < 50:
        pcd_objects = pcd_scene

    obj_pts = np.asarray(pcd_objects.points)
    real_object_center = np.median(obj_pts, axis=0)
    model_center = np.mean(model_pts_3d, axis=0)

    T_coarse = np.eye(4)
    T_coarse[:3, 3] = real_object_center - model_center

    pcd_model = o3d.geometry.PointCloud()
    pcd_model.points = o3d.utility.Vector3dVector(model_pts_3d)

    icp_result = o3d.pipelines.registration.registration_icp(
        pcd_model,
        pcd_objects,
        max_correspondence_distance=0.10,
        init=T_coarse,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    if icp_result.fitness < 0.15:
        print("[Recovery] ICP recovery failed.")
        return None, False
    print("[Recovery] ICP recovery succeeded. Fitness:", icp_result.fitness)
    return icp_result.transformation, True

def b5_transition(T_obs, T_prior, p_obs_bad, p_prior_bad, support, depth_real, model_pts,
                  K, p_risk_threshold, frame_index, frame_id, state,
                  blackout_min_frames=10, use_prior_predictor=True):
    """
    Shared B5 transition used by BOTH risk-label rollout and deployment evaluation.

    Final deployed policy:
      1) blackout -> prior propagation
      2) first frame after a valid blackout -> independent recovery
      3) low observation risk -> observation
      4) low prior risk -> prior
      5) both risky -> SE(3) uncertain fusion

    Bootstrap label stage uses the SAME state-transition function with
    use_prior_predictor=False; before a prior predictor exists, an unreliable
    observation falls back to the recursive prior.
    """
    state = dict(state)
    state["blackout_intervals"] = [dict(item) for item in state.get("blackout_intervals", [])]
    recovery_info = None

    if support == 1:
        state["consecutive_blackout"] += 1
        if state["consecutive_blackout"] == 1:
            state["blackout_start_idx"] = frame_index
            state["blackout_start_frame"] = frame_id
        state["last_blackout_idx"] = frame_index
        state["last_blackout_frame"] = frame_id
    else:
        if state["consecutive_blackout"] >= blackout_min_frames:
            state["exited_blackout"] = True
            state["blackout_end_idx"] = frame_index
            state["blackout_end_frame"] = state["last_blackout_frame"]
            state["recovery_frame"] = frame_id
            state["blackout_intervals"].append({
                "blackout_start_index": state["blackout_start_idx"],
                "blackout_end_index": state["last_blackout_idx"],
                "recovery_index": frame_index,
                "blackout_start_frame": state["blackout_start_frame"],
                "blackout_end_frame": state["last_blackout_frame"],
                "recovery_frame": frame_id,
            })
        state["consecutive_blackout"] = 0

    if support == 1:
        current_mode = "MODE_3_BLACKOUT_WAITING"
        T_final = T_prior

    elif state["exited_blackout"]:
        T_recovery, recovery_ok = actual_recovery_action(depth_real, model_pts, K)
        recovery_info = {
            "recovery_frame": frame_id,
            "recovery_frame_index": frame_index,
            "recovery_success": bool(recovery_ok),
            "T_recovery": T_recovery.copy() if recovery_ok else None,
            "blackout_interval": dict(state["blackout_intervals"][-1]),
        }
        if recovery_ok:
            T_final = T_recovery
        else:
            T_final = T_prior
        current_mode = "MODE_3_RECOVERY_EXECUTE"
        state["exited_blackout"] = False

    elif p_obs_bad <= p_risk_threshold:
        current_mode = "MODE_1_ACCEPT"
        T_final = T_obs

    elif use_prior_predictor and p_prior_bad is not None and p_prior_bad <= p_risk_threshold:
        current_mode = "MODE_2_PRIOR"
        T_final = T_prior

    elif use_prior_predictor:
        current_mode = "MODE_3_UNCERTAIN_FUSION"
        T_delta = np.linalg.inv(T_prior) @ T_obs
        alpha = 0.5
        T_final = T_prior @ se3_exp_map(alpha * se3_log_map(T_delta))

    else:
        current_mode = "MODE_BOOTSTRAP_PRIOR"
        T_final = T_prior

    return T_final, current_mode, state, recovery_info
