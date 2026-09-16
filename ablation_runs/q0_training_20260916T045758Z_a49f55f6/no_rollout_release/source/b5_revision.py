"""Shared development policy weights/history for training and evaluation.

Initial bounds are engineering defaults, not fitted on frozen test outcomes.
No GT inputs, no extra registration calls, no change to the pose math helpers.
"""
import numpy as np

CONFIG = dict(version="relative_quality_restart_v1_1_development", status="development_unvalidated",
    epsilon_cm=0.01, mode2_min_alpha=0.60, mode2_max_alpha=0.90,
    mode3_min_alpha=0.25, mode3_max_alpha=0.75, max_prior_streak=5,
    streak_limit_action="one_legacy_weak_mode3_no_recovery",
    legacy_weak_scale=0.15, legacy_weak_min_alpha=0.10, legacy_weak_max_alpha=0.30,
    prior_drift_decay=0.90, prior_drift_increment=0.08,
    history_reset="accepted_and_used_recovery_only", restart_prediction="one_step_zero_velocity",
    history_update="every_final_output_after_restart", history_retained_poses=2)


def relative_alpha(obs_error_cm, prior_error_cm, mode):
    if mode not in (2, 3):
        raise ValueError("Expected fusion mode 2 or 3")
    if not np.isfinite([obs_error_cm, prior_error_cm]).all():
        raise ValueError("Nonfinite predicted errors")
    obs, prior = max(float(obs_error_cm), 0.), max(float(prior_error_cm), 0.)
    eps = CONFIG["epsilon_cm"]
    # Scale before summation to avoid overflow for finite large errors.
    scale = max(obs, prior, eps)
    ratio = (obs/scale+eps/scale)/(obs/scale+prior/scale+2*eps/scale)
    return float(np.clip(ratio, CONFIG["mode%d_min_alpha" % mode], CONFIG["mode%d_max_alpha" % mode]))


def make_prior(history, observation, state, extrapolate):
    if len(history) >= 2:
        return extrapolate(history[-1], history[-2])
    if len(history) == 1 and state.get("motion_history_restarted", False):
        return np.asarray(history[-1], dtype=float).copy()
    # Preserve original first-two-frame initialization before any recovery.
    return np.asarray(observation, dtype=float).copy()


def advance_history(history, final_pose, state):
    pose = np.asarray(final_pose, dtype=float).reshape(4, 4).copy()
    if not np.isfinite(pose).all():
        raise ValueError("Cannot append nonfinite operational pose")
    if state.get("reset_motion_history", False):
        return [pose]
    return [np.asarray(p, dtype=float).copy() for p in history[-1:]] + [pose]
