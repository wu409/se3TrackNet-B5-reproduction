"""Numerical pose validity only. No GT or geometric quality gate."""
import numpy as np


def is_se3(pose):
    try:
        t = np.asarray(pose, dtype=float)
        return bool(t.shape == (4, 4) and np.isfinite(t).all()
            and np.allclose(t[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            and np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-3, rtol=0)
            and np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-3, rtol=0))
    except (TypeError, ValueError):
        return False


def se3_only_admission(T_recovery, *args, **kwargs):
    valid = is_se3(T_recovery)
    return dict(accepted_recovery=valid, recovery_pose_valid=valid,
        recovery_rejection_reasons=[] if valid else ["invalid_SE3_matrix"],
        recovery_gate_version="mode3_SE3_only_v2",
        recovery_admission_rule="SE3_safety_only_no_quality_gate")
