"""Shared development policy weights/history for training and evaluation.

Initial bounds are engineering defaults, not fitted on frozen test outcomes.
No GT inputs. MODE3 adds registration calls, explicitly logged separately.
"""
import numpy as np

CONFIG = dict(version="four_mode_relocalization_v2", status="development_unvalidated",
    epsilon_cm=0.01, mode4_min_alpha=0.25, mode4_max_alpha=0.75,
    mode2="pure_prior_no_streak_limit", mode3="both_risky_SE3_only_relocalization",
    mode3_history="append_without_reset", mode3_failure="prior_flagged_uncertain",
    mode3_budget="every_eligible_frame_no_cooldown",
    history_reset="accepted_blackout_recovery_only", restart_prediction="one_step_zero_velocity",
    observation_restart="accepted_blackout_or_mode3_then_tracker_own_predictions",
    history_update="every_final_output", history_retained_poses=2)


def relative_alpha(obs_error_cm, prior_error_cm, mode):
    if mode != 4:
        raise ValueError("Only MODE4 fuses in four-mode v2")
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
