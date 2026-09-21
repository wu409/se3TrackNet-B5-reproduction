"""Evaluation-only controls. No GT, training, new triggers or registration calls."""
from functools import wraps
import numpy as np


def ungated_recovery_validity(T_recovery, *args, **kwargs):
    """Remove geometric admission, retaining only representable SE(3) safety.

    No mask/depth agreement, temporal jump or learned score is consulted.
    Absent/failed raw proposals still fail upstream; this cannot invent a pose.
    """
    try:
        t = np.asarray(T_recovery, dtype=float)
        valid = bool(t.shape == (4, 4) and np.isfinite(t).all()
                     and np.allclose(t[3], [0., 0., 0., 1.], atol=1e-6, rtol=0.)
                     and np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-3, rtol=0.)
                     and np.isclose(np.linalg.det(t[:3, :3]), 1., atol=1e-3, rtol=0.))
    except (TypeError, ValueError):
        valid = False
    diag = dict(accepted_recovery=valid, recovery_pose_valid=valid,
                recovery_rejection_reasons=[] if valid else ["invalid_SE3_matrix"],
                recovery_gate_version="ablation_no_blackout_geometric_admission_v2",
                recovery_admission_rule="SE3_safety_only_no_quality_gate")
    diag["recovery_soft_evidence"] = {"gate": dict(diag)}
    return diag


def without_recovery_admission(transition):
    """Scoped override for the single-threaded evaluation process; always restore."""
    namespace = transition.__globals__
    key = "evaluate_recovery_pose_validity"
    if key not in namespace:
        raise ValueError("Frozen policy has no compatible recovery admission hook")

    @wraps(transition)
    def controlled(*args, **kwargs):
        original = namespace[key]
        namespace[key] = ungated_recovery_validity
        try:
            return transition(*args, **kwargs)
        finally:
            namespace[key] = original
    return controlled
