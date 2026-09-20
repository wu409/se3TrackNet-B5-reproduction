"""Development-only, fixed-candidate diagnostics from frozen releases.

Does not mutate a release or claim an independent test. Requires the same q0/q1
pair produced by run_all.sh; every output is created exclusively.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import RobustScaler

FEATURES = ["x1_norm", "x2_inlier_error", "x4_support_ratio", "x5_geometry_inconsistency"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def release(path, expected_rounds):
    path = Path(path).resolve(strict=True)
    cfg = json.loads((path / "artifacts/shared_quality_config.json").read_text())
    frozen = json.loads((path / "FROZEN.json").read_text())
    if frozen.get("status") != "final_development_model_frozen" or cfg["on_policy_refine_rounds"] != expected_rounds:
        raise ValueError("Expected a frozen q%d final-development release" % expected_rounds)
    if cfg["feature_columns"] != FEATURES:
        raise ValueError("Feature schema mismatch")
    for key, name in [("scaler", "shared_pose_quality_scaler.joblib"),
                      ("model", "shared_pose_quality_model.joblib"),
                      ("calibrator", "shared_risk_calibrator.joblib")]:
        if digest(path / "artifacts" / name) != cfg[key + "_sha256"]:
            raise ValueError("Frozen artifact hash mismatch: " + name)
    labels = next(path.glob("per_frame_label_threshold*.csv"))
    return dict(path=path, cfg=cfg, labels=labels,
                scaler=joblib.load(path / "artifacts/shared_pose_quality_scaler.joblib"),
                model=joblib.load(path / "artifacts/shared_pose_quality_model.joblib"),
                calibrator=joblib.load(path / "artifacts/shared_risk_calibrator.joblib"))


def samples(labels):
    df = pd.read_csv(labels)
    rows = []
    for source in ("obs", "prior"):
        cols = ["x1_%s_norm" % source, "x2_%s_inlier_error" % source,
                "x4_%s_support_ratio" % source, "x5_%s_geometry_inconsistency" % source]
        sub = df[["sequence", "sequence_index", "frame_id", "D_obj_cm",
                  "E_%s_cm" % source] + cols].copy()
        sub.columns = ["sequence", "sequence_index", "frame_id", "D_obj_cm", "target_E_cm"] + FEATURES
        sub["hypothesis"] = source
        sub["target_e_norm"] = sub.target_E_cm / sub.D_obj_cm
        rows.append(sub)
    result = pd.concat(rows, ignore_index=True)
    if not np.isfinite(result[FEATURES + ["target_E_cm", "D_obj_cm"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite candidate features/labels")
    return result


def temporal_split(df, fraction):
    out = df.copy()
    out["split"] = "cal"
    for seq, group in out.groupby("sequence"):
        idx = np.sort(group.sequence_index.astype(int).unique())
        cut = max(1, min(len(idx) - 1, int(len(idx) * fraction)))
        out.loc[(out.sequence == seq) & out.sequence_index.isin(idx[:cut]), "split"] = "train"
    return out


def fit(df, threshold, fraction):
    split = temporal_split(df, fraction)
    train, cal = split[split.split == "train"], split[split.split == "cal"]
    if min(len(train), len(cal)) == 0:
        raise ValueError("Empty temporal fit/calibration partition")
    scaler = RobustScaler().fit(train[FEATURES].to_numpy(dtype=float))
    model = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000, tol=1e-6)
    model.fit(scaler.transform(train[FEATURES]), train.target_e_norm.to_numpy(dtype=float))
    error = np.maximum(model.predict(scaler.transform(cal[FEATURES])), 0) * cal.D_obj_cm.to_numpy()
    binary = (cal.target_E_cm.to_numpy() > threshold).astype(int)
    if len(np.unique(binary)) != 2:
        raise ValueError("Calibration split lacks both risk classes")
    isotonic = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip")
    isotonic.fit(error, binary)
    return dict(scaler=scaler, model=model, calibrator=isotonic)


def score(fitted, df):
    error = np.maximum(fitted["model"].predict(fitted["scaler"].transform(df[FEATURES])), 0) * df.D_obj_cm.to_numpy()
    return error, np.asarray(fitted["calibrator"].predict(error))


def metrics(name, population, df, error, probability, threshold):
    truth = (df.target_E_cm.to_numpy() > threshold).astype(int)
    return dict(model=name, population=population, n=len(df),
                mae_cm=float(mean_absolute_error(df.target_E_cm, error)),
                risk_auroc=(float(roc_auc_score(truth, probability)) if len(np.unique(truth)) == 2 else None),
                observation_mae_cm=float(mean_absolute_error(df.loc[df.hypothesis == "obs", "target_E_cm"],
                    error[df.hypothesis.to_numpy() == "obs"])),
                prior_mae_cm=float(mean_absolute_error(df.loc[df.hypothesis == "prior", "target_E_cm"],
                    error[df.hypothesis.to_numpy() == "prior"])))


def threshold_search(values, labels):
    values = np.asarray(values)
    labels = np.asarray(labels)
    unique = np.unique(values)
    candidates = np.r_[np.nextafter(unique[0], -np.inf), unique]
    # > implements the same high-risk direction used by the calibrated gate.
    ranks = [(balanced_accuracy_score(labels, values > x), x) for x in candidates]
    return float(max(ranks, key=lambda pair: (pair[0], -pair[1]))[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["source", "refit", "calibration"])
    p.add_argument("--q0", required=True)
    p.add_argument("--q1", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fixed-policy-samples", help="Optional independent observation-only-policy candidate CSV; see README")
    a = p.parse_args()
    q0, q1 = release(a.q0, 0), release(a.q1, 1)
    c0, c1 = q0["cfg"], q1["cfg"]
    for key in ("manifest_sha256", "train_bases", "fit_conditions", "risk_threshold_cm", "train_fraction", "seed"):
        if c0[key] != c1[key]:
            raise ValueError("q0/q1 incompatible: " + key)
    out = Path(a.output).resolve()
    if out.exists() or out in (q0["path"], q1["path"]) or q0["path"] in out.parents or q1["path"] in out.parents:
        raise ValueError("Output must be a NEW directory outside frozen releases")
    d0, d1 = samples(q0["labels"]), samples(q1["labels"])
    threshold = float(c1["risk_threshold_cm"])
    rows = []
    details = {}
    if a.mode == "source":
        # Fit two independent heads on the same q0-policy candidate population used for q1 fitting.
        heads = {kind: fit(d0[d0.hypothesis == kind], threshold, c1["train_fraction"])
                 for kind in ("obs", "prior")}
        for population, data in (("q0_policy", d0), ("q1_policy", d1)):
            shared_e, shared_p = score(q1, data)
            rows.append(metrics("frozen_shared_q1", population, data, shared_e, shared_p, threshold))
            separate_e = np.empty(len(data)); separate_p = np.empty(len(data))
            for kind in heads:
                mask = data.hypothesis.to_numpy() == kind
                separate_e[mask], separate_p[mask] = score(heads[kind], data.loc[mask])
            rows.append(metrics("source_specific_q0_policy_fit", population, data, separate_e, separate_p, threshold))
    elif a.mode == "refit":
        for population, data in (("q0_policy_same_candidates", d0), ("q1_policy_different_candidates", d1)):
            for name, model in (("frozen_q0", q0), ("frozen_q1", q1)):
                error, prob = score(model, data)
                rows.append(metrics(name, population, data, error, prob, threshold))
        if a.fixed_policy_samples:
            fixed_path = Path(a.fixed_policy_samples).resolve(strict=True)
            fixed = pd.read_csv(fixed_path)
            required = set(d0.columns) - {"split"}
            if not required <= set(fixed.columns) or set(fixed.hypothesis) != {"obs", "prior"}:
                raise ValueError("fixed-policy CSV must contain paired obs/prior development features and labels")
            if set(fixed.sequence) != set(d0.sequence):
                raise ValueError("fixed-policy development sequences differ")
            # Equal number of hypotheses, same architecture and temporal split. Never use q1 rollout labels here.
            prior_exposure = fit(fixed, threshold, c1["train_fraction"])
            for population, data in (("q0_policy_same_candidates", d0), ("q1_policy_different_candidates", d1)):
                error, prob = score(prior_exposure, data)
                rows.append(metrics("fixed_policy_prior_exposure", population, data, error, prob, threshold))
            details["fixed_policy_samples_sha256"] = digest(fixed_path)
        else:
            details["prior_exposure_control"] = "NOT RUN: independent fixed-policy candidate cache absent; q0-policy pairs are q1 training data and cannot isolate prior exposure"
    else:
        # Calibrated isotonic is monotone in raw error: report routing agreement, not a spurious AUC gain.
        cal = temporal_split(d0, c1["train_fraction"])
        cal = cal[cal.split == "cal"]
        error, prob = score(q1, cal)
        raw_cut = threshold_search(error, cal.target_E_cm.to_numpy() > threshold)
        p_cut = float(c1["p_risk_threshold"])
        for population, data in (("development_calibration", cal), ("q1_policy_diagnostic", d1)):
            e, risk = score(q1, data)
            truth = data.target_E_cm.to_numpy() > threshold
            for name, routed in (("isotonic_frozen_threshold", risk > p_cut),
                                 ("raw_development_tuned", e > raw_cut)):
                rows.append(dict(model=name, population=population, n=len(data),
                                 balanced_accuracy=float(balanced_accuracy_score(truth, routed)),
                                 high_risk_rate=float(np.mean(routed))))
            details[population + "_route_disagreement"] = int(np.count_nonzero((risk > p_cut) != (e > raw_cut)))
        details.update(raw_cut_cm=raw_cut, frozen_probability_cut=p_cut)
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    provenance = dict(mode=a.mode, q0=str(q0["path"]), q1=str(q1["path"]),
                      q0_label_sha256=digest(q0["labels"]), q1_label_sha256=digest(q1["labels"]),
                      status="post_development_diagnostic_not_held_out", details=details)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(out)


if __name__ == "__main__":
    main()
