import numpy as np

from b5_policy import (
    _project_cad_silhouette_mask,
    _project_cad_depth_residual,
    evaluate_recovery_pose_validity,
)


def _cube_points(side=0.10, n=11):
    a = np.linspace(-side / 2.0, side / 2.0, n)
    pts = []
    # Six surfaces for a reasonably dense synthetic CAD point set.
    for x in (-side / 2.0, side / 2.0):
        for y in a:
            for z in a:
                pts.append([x, y, z])
    for y in (-side / 2.0, side / 2.0):
        for x in a:
            for z in a:
                pts.append([x, y, z])
    for z in (-side / 2.0, side / 2.0):
        for x in a:
            for y in a:
                pts.append([x, y, z])
    return np.unique(np.asarray(pts, dtype=np.float64), axis=0)


def _scene():
    K = np.array([[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]])
    model = _cube_points()
    T_ref = np.eye(4, dtype=np.float64)
    T_ref[2, 3] = 1.0
    visible = _project_cad_silhouette_mask(T_ref, model, K, (480, 640))
    assert visible is not None

    # Build a synthetic depth image directly from the same projected CAD points.
    depth = np.zeros((480, 640), dtype=np.float32)
    T = T_ref
    pts_cam = (T[:3, :3] @ model.T).T + T[:3, 3]
    z = pts_cam[:, 2]
    u = np.round(K[0, 0] * pts_cam[:, 0] / z + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * pts_cam[:, 1] / z + K[1, 2]).astype(int)
    inside = (u >= 0) & (u < 640) & (v >= 0) & (v < 480) & (z > 0)
    for uu, vv, zz in zip(u[inside], v[inside], z[inside]):
        if depth[vv, uu] == 0 or zz < depth[vv, uu]:
            depth[vv, uu] = zz
    return K, model, T_ref, visible, depth


def test_good_pose_is_accepted():
    K, model, T_ref, visible, depth = _scene()
    result = evaluate_recovery_pose_validity(
        T_recovery=T_ref,
        model_pts_3d=model,
        K=K,
        visible_mask=visible,
        depth_real=depth,
        reference_T_final=T_ref,
        template_score_margin=0.10,
    )
    assert result["accepted_recovery"] is True, result


def test_wrong_object_or_wrong_location_mask_is_rejected():
    K, model, T_ref, visible, depth = _scene()
    wrong_mask = np.roll(visible, shift=180, axis=1)
    result = evaluate_recovery_pose_validity(
        T_recovery=T_ref,
        model_pts_3d=model,
        K=K,
        visible_mask=wrong_mask,
        depth_real=depth,
        reference_T_final=T_ref,
        template_score_margin=0.10,
    )
    assert result["accepted_recovery"] is False
    assert any(
        reason in result["recovery_rejection_reasons"]
        for reason in ("visible_mask_iou_too_low", "cad_coverage_too_low", "reprojection_center_error_too_large")
    )


def test_bleach_hard_45_3cm_gross_displacement_is_rejected():
    """Regression case matching the ~45.3 cm bleach_hard failure magnitude."""
    K, model, T_ref, visible, depth = _scene()
    T_bad = T_ref.copy()
    T_bad[0, 3] += 0.453
    result = evaluate_recovery_pose_validity(
        T_recovery=T_bad,
        model_pts_3d=model,
        K=K,
        visible_mask=visible,
        depth_real=depth,
        reference_T_final=T_ref,
        template_score_margin=0.10,
    )
    assert result["accepted_recovery"] is False
    assert "translation_jump_too_large" in result["recovery_rejection_reasons"]


def test_bleach_hard_49_4cm_gross_displacement_is_rejected():
    """Regression case matching the ~49.4 cm bleach_hard failure magnitude."""
    K, model, T_ref, visible, depth = _scene()
    T_bad = T_ref.copy()
    T_bad[1, 3] += 0.494
    result = evaluate_recovery_pose_validity(
        T_recovery=T_bad,
        model_pts_3d=model,
        K=K,
        visible_mask=visible,
        depth_real=depth,
        reference_T_final=T_ref,
        template_score_margin=0.10,
    )
    assert result["accepted_recovery"] is False
    assert "translation_jump_too_large" in result["recovery_rejection_reasons"]


def test_ambiguous_template_score_margin_is_rejected():
    K, model, T_ref, visible, depth = _scene()
    result = evaluate_recovery_pose_validity(
        T_recovery=T_ref,
        model_pts_3d=model,
        K=K,
        visible_mask=visible,
        depth_real=depth,
        reference_T_final=T_ref,
        template_score_margin=0.001,
    )
    assert result["accepted_recovery"] is False
    assert "template_score_margin_too_small" in result["recovery_rejection_reasons"]
