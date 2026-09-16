"""Development-only, current-frame recovery admission. No GT or history inputs.

CAD depths use a sparse nearest-point z-buffer, not a mesh renderer. Thresholds
are starting values for development validation, NOT calibrated guarantees.
"""
import numpy as np

VERSION = "occlusion_aware_v1_development"
CONFIG = dict(version=VERSION, status="development_unvalidated",
    min_support=50, min_inliers=50, min_depth_m=0.05, max_depth_m=5.0,
    depth_tolerance_floor_m=0.01, depth_tolerance_diameter_fraction=0.05,
    max_conflict_fraction=0.10, min_global_inlier_fraction=0.10,
    min_mask_explained=0.60, normal_min_cad_coverage=0.30,
    normal_min_inlier_fraction=0.60, normal_min_spatial_bins=3,
    occluded_min_cad_coverage=0.10, occluded_min_inlier_fraction=0.80,
    occluded_min_spatial_bins=4, min_occlusion_fraction=0.10,
    spatial_grid_size=4, rotation_tolerance=0.001)


def evaluate_current_frame(T, points, K, depth, mask, silhouette):
    """Classify foreground occlusion separately from free-space contradiction.

    Invalid depths never count as occlusion or matching evidence. Foreground
    depth is only an occlusion *hypothesis*, not proof of a robot-arm identity.
    Positive inliers must support a nontrivial fraction of the whole candidate.
    """
    c = CONFIG
    out = dict(recovery_gate_version=VERSION, recovery_gate_status=c["status"],
        recovery_gate_config=dict(c), recovery_acceptance_path="none",
        recovery_decision_category="insufficient_evidence", recovery_hard_gate_pass=False,
        recovery_rejection_reasons=[], recovery_pose_valid=False, accepted_recovery=False,
        raw_recovery_generated=T is not None)

    def reject(reason, category="insufficient_evidence"):
        out["recovery_rejection_reasons"].append(reason)
        out["recovery_decision_category"] = category
        return out

    if any(x is None for x in (T, points, K, depth, mask, silhouette)):
        return reject("missing_pose_depth_mask_or_projection")
    T, points, K = np.asarray(T, float), np.asarray(points, float), np.asarray(K, float)
    depth, mask, silhouette = np.asarray(depth, float), np.asarray(mask)>0, np.asarray(silhouette)>0
    if (T.shape != (4, 4) or K.shape != (3, 3) or points.ndim != 2 or points.shape[1] != 3
            or depth.ndim != 2 or mask.shape != depth.shape or silhouette.shape != depth.shape):
        return reject("invalid_input_shape", "invalid_input")
    if not all(np.isfinite(x).all() for x in (T, points, K)) or not len(points):
        return reject("non_finite_pose_or_geometry", "invalid_input")
    R = T[:3, :3]
    if (not np.allclose(R.T @ R, np.eye(3), atol=c["rotation_tolerance"], rtol=0)
            or abs(np.linalg.det(R)-1) > c["rotation_tolerance"]
            or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6, rtol=0)):
        return reject("invalid_rigid_transform", "invalid_input")
    diameter = float(np.linalg.norm(np.ptp(points, axis=0)))
    if diameter <= 0 or K[0, 0] <= 0 or K[1, 1] <= 0:
        return reject("invalid_geometry_or_intrinsics", "invalid_input")
    tau = max(c["depth_tolerance_floor_m"], c["depth_tolerance_diameter_fraction"] * diameter)
    xyz = points @ R.T + T[:3, 3]
    xyz = xyz[xyz[:, 2] > 1e-6]
    if len(xyz) < 3 or not silhouette.any():
        return reject("cad_projection_invalid", "invalid_input")
    h, w = depth.shape
    uv = xyz @ K.T
    uv = np.rint(uv[:, :2] / uv[:, 2:3])
    inside = (uv[:, 0]>=0)&(uv[:, 0]<w)&(uv[:, 1]>=0)&(uv[:, 1]<h)
    uv, xyz = uv[inside].astype(int), xyz[inside]
    zbuf = np.full(h*w, np.inf)
    np.minimum.at(zbuf, uv[:, 1]*w+uv[:, 0], xyz[:, 2])
    rendered = zbuf.reshape(h, w)
    projected = np.isfinite(rendered)
    valid = np.isfinite(depth)&(depth>c["min_depth_m"])&(depth<c["max_depth_m"])
    comparable = projected & valid
    delta = np.zeros_like(depth)
    delta[comparable] = depth[comparable]-rendered[comparable]
    occluded = comparable & (delta < -tau)
    conflict = comparable & (delta > 2*tau)
    visible = comparable & ~occluded
    support = visible & mask
    inliers = support & (np.abs(delta)<=tau)
    n, ns, ni = int(comparable.sum()), int(support.sum()), int(inliers.sum())
    # Occluded pixels are neither good matches nor failures. Keep denominators
    # before occlusion removal too, to prevent a far-away pose gaming the test.
    occ_fraction = float(occluded.sum()/max(n, 1))
    conflict_fraction = float(conflict.sum()/max(int(visible.sum()), 1))
    inlier_fraction = ni/max(ns, 1)
    global_fraction = ni/max(n, 1)
    overlap = silhouette & mask & ~occluded
    mask_explained = float(overlap.sum()/max(int((mask & ~occluded).sum()), 1))
    coverage = float(overlap.sum()/max(int((silhouette & ~occluded).sum()), 1))
    bins = 0
    py, px = np.where(projected)
    iy, ix = np.where(inliers)
    if len(ix) and len(px):
        g = c["spatial_grid_size"]
        bx = np.minimum(g-1, (ix-px.min())*g//max(int(np.ptp(px))+1, 1))
        by = np.minimum(g-1, (iy-py.min())*g//max(int(np.ptp(py))+1, 1))
        bins = len(np.unique(by*g+bx))
    out.update(recovery_depth_tolerance_m=tau, recovery_comparable_pixels=n,
        recovery_visible_support_pixels=ns, recovery_inlier_pixels=ni,
        recovery_occlusion_fraction=occ_fraction, recovery_conflict_fraction=conflict_fraction,
        recovery_inlier_fraction=inlier_fraction, recovery_global_inlier_fraction=global_fraction,
        recovery_visible_mask_explained=mask_explained, recovery_visible_cad_coverage=coverage,
        recovery_inlier_spatial_bins=bins,
        recovery_depth_method="sparse_cad_point_zbuffer")
    if conflict_fraction > c["max_conflict_fraction"]:
        return reject("current_depth_free_space_conflict", "geometric_conflict")
    if ns < c["min_support"] or ni < c["min_inliers"] or global_fraction < c["min_global_inlier_fraction"]:
        return reject("insufficient_positive_depth_support")
    out["recovery_hard_gate_pass"] = True
    normal = (mask_explained>=c["min_mask_explained"] and coverage>=c["normal_min_cad_coverage"]
        and inlier_fraction>=c["normal_min_inlier_fraction"] and bins>=c["normal_min_spatial_bins"])
    occlusion = (occ_fraction>=c["min_occlusion_fraction"] and mask_explained>=c["min_mask_explained"]
        and coverage>=c["occluded_min_cad_coverage"] and inlier_fraction>=c["occluded_min_inlier_fraction"]
        and bins>=c["occluded_min_spatial_bins"])
    out.update(recovery_normal_path_pass=bool(normal), recovery_occlusion_path_pass=bool(occlusion))
    if not (normal or occlusion):
        return reject("insufficient_current_frame_consistency")
    out.update(recovery_acceptance_path="normal" if normal else "occlusion",
        recovery_decision_category="accepted", recovery_pose_valid=True, accepted_recovery=True)
    return out
