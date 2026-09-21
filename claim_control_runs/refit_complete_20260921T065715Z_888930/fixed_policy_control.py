"""Complete development-only refit attribution using frozen feature/fitting code.

No tracker training, no quality-driven decisions during fixed-policy collection.
Original frozen models are anchors; newly fitted controls are separately named.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace


FEATURES = ["x1_norm", "x2_inlier_error", "x4_support_ratio", "x5_geometry_inconsistency"]
POLICY = "always_accept_frozen_precomputed_observation_no_reseed_v1"
KEYS = ["sequence", "sequence_index", "frame_id", "hypothesis"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def frozen_release(path, rounds):
    root = Path(path).resolve(strict=True)
    expected = {}
    for line in (root / "freeze.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        target = (root / name).resolve()
        if root not in target.parents:
            raise ValueError("Frozen path escapes release: " + name)
        expected[name] = digest
    needed = {"FROZEN.json", "effective_config.json", "reference_manifest.csv", "observer_config.json", "input_hashes.json",
              "artifacts/shared_quality_config.json", "source/2-risk_label.py", "source/b5_revision.py"}
    needed.update(name for name in expected if name.startswith("source/") and name.endswith(".py"))
    if rounds == 0:
        needed.add("ablation_parent.json")
    cfg = read(root / "artifacts/shared_quality_config.json")
    labels = "per_frame_label_threshold%s.csv" % float(cfg["risk_threshold_cm"])
    needed.add(labels)
    for kind, name in (("scaler", "shared_pose_quality_scaler.joblib"),
                       ("model", "shared_pose_quality_model.joblib"),
                       ("calibrator", "shared_risk_calibrator.joblib")):
        needed.add("artifacts/" + name)
        if sha(root / "artifacts" / name) != cfg[kind + "_sha256"]:
            raise ValueError("Artifact differs from model config: " + kind)
    for name in needed:
        if name not in expected or sha(root / name) != expected[name]:
            raise ValueError("Frozen source/data hash mismatch: " + name)
    if read(root / "FROZEN.json")["status"] != "final_development_model_frozen":
        raise ValueError("Release is not complete")
    if cfg["on_policy_refine_rounds"] != rounds or cfg["training_mode"] != "final_development_fit":
        raise ValueError("Wrong frozen q%d release" % rounds)
    if cfg.get("held_out_base") is not None or cfg["feature_columns"] != FEATURES:
        raise ValueError("Unsupported model schema")
    if sha(root / "reference_manifest.csv") != cfg["manifest_sha256"]:
        raise ValueError("Manifest/config mismatch")
    return dict(root=root, cfg=cfg, effective=read(root / "effective_config.json"), labels=root / labels,
                verified={str(root / name): expected[name] for name in needed})


def fixed_prior(history, observation, extrapolate):
    """Exact first-two-frame initialization, then prior from past observations only."""
    if len(history) < 2:
        return observation.copy()
    return extrapolate(history[-1], history[-2])


def validate_samples(df, manifest):
    import numpy as np
    import pandas as pd
    required = set(KEYS + FEATURES + ["D_obj_cm", "target_E_cm", "target_e_norm"])
    if not required <= set(df.columns) or df.duplicated(KEYS).any():
        raise ValueError("Missing schema or duplicate candidate keys")
    if set(df.hypothesis) != {"obs", "prior"}:
        raise ValueError("Need exactly observation and prior candidates")
    expected = pd.concat([manifest[["sequence", "sequence_index", "frame_id"]].assign(hypothesis=h)
                          for h in ("obs", "prior")], ignore_index=True)
    actual_keys = pd.MultiIndex.from_frame(df[KEYS])
    expected_keys = pd.MultiIndex.from_frame(expected[KEYS])
    if len(df) != len(expected) or len(actual_keys.difference(expected_keys)) or len(expected_keys.difference(actual_keys)):
        raise ValueError("Candidate keys differ from exact development frames; no test frames allowed")
    values = df[FEATURES + ["D_obj_cm", "target_E_cm", "target_e_norm"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (df.D_obj_cm <= 0).any() or (df.target_E_cm < 0).any():
        raise ValueError("Invalid candidate features/labels")
    if not np.allclose(df.target_e_norm * df.D_obj_cm, df.target_E_cm, rtol=1e-8, atol=1e-8):
        raise ValueError("Normalized and physical labels disagree")


def load_frozen_module(full):
    source = full["root"] / "source"
    os.environ["PYOPENGL_PLATFORM"] = full["effective"]["paths"].get("PYOPENGL_PLATFORM") or "egl"
    budgets = full["cfg"]["execution_settings"]
    os.environ["B5_NUM_THREADS"] = str(budgets["cpu_threads"])
    os.environ["B5_IO_WORKERS"] = str(budgets["io_workers"])
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("frozen_risk_labels", str(source / "2-risk_label.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("b5_policy", "b5_revision", "Utils", "runtime_settings"):
        if Path(sys.modules[name].__file__).resolve().parent != source:
            raise ValueError("Non-frozen dependency loaded: " + name)
    module.runtime_settings.apply_frozen(budgets)
    if module.SHARED_FEATURE_COLUMNS != FEATURES:
        raise ValueError("Frozen feature schema differs")
    return module


def collect(frozen, full, manifest, cache, smoke):
    np, pd = frozen.np, frozen.pd
    cfg, paths = full["cfg"], full["effective"]["paths"]
    args = SimpleNamespace(data_dir=paths["DATASET_ROOT"], ycb_dir=paths["GT_ROOT"], res_dir=paths["RESULT_ROOT"])
    if smoke:
        first = cfg["train_bases"][0] + cfg["fit_conditions"][0]
        selected = frozen.get_episode_df(manifest, first).head(4)
    else:
        selected = manifest
    frozen.verify_manifest_artifacts(selected, args)
    mapping = read(full["root"] / "observer_config.json")["sequence_objects"]
    frozen_inputs = read(full["root"] / "input_hashes.json")
    cache.mkdir()
    chunks, assets = [], {}
    for base in cfg["train_bases"]:
        if base not in set(selected.base_sequence):
            continue
        cad = Path(paths["CAD_MODEL_ROOT"]) / mapping[base]
        for asset in cad.rglob("*"):
            if asset.is_file():
                key = str(asset.resolve())
                assets[key] = sha(asset)
                if frozen_inputs.get(key) != assets[key]:
                    raise ValueError("CAD asset differs from original frozen input: " + key)
        mesh = frozen.trimesh.load(str(cad / "textured.obj"))
        points = np.loadtxt(cad / "points.xyz", dtype=np.float64).reshape(-1, 3)
        diameter = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)) * 100.)
        cloud = frozen.U.toOpen3dCloud(points, colors=np.zeros(points.shape, dtype=np.float64))
        scene = frozen.pyrender.Scene()
        node = scene.add(frozen.pyrender.Mesh.from_trimesh(mesh, smooth=False))
        k = frozen.K
        scene.add(frozen.pyrender.IntrinsicsCamera(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2]), pose=np.eye(4))
        renderer = frozen.pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
        try:
            for condition in cfg["fit_conditions"]:
                seq = base + condition
                if seq not in set(selected.sequence):
                    continue
                episode = frozen.get_episode_df(selected, seq)
                history, rows = [], []
                for _, row in episode.iterrows():
                    obs, gt, depth = frozen.load_frame_from_manifest(row, args)
                    prior = fixed_prior(history, obs, frozen.compute_se3_prior)
                    # Policy history is committed from the observation, before GT labels are computed.
                    history = (history + [obs.copy()])[-2:]
                    for hypothesis, pose in (("obs", obs), ("prior", prior)):
                        feat = frozen.extract_pose_conditioned_features(pose, depth, 0, [points], [scene],
                                                                        [renderer], [node], include_support=True)
                        vector = frozen.shared_feature_vector(feat, diameter)
                        error = float(frozen.U.adi(pose, gt, cloud) * 100.)
                        sample = dict(sequence=seq, sequence_index=int(row.sequence_index), frame_id=int(row.frame_id),
                                      hypothesis=hypothesis, D_obj_cm=diameter, target_E_cm=error,
                                      target_e_norm=error / diameter)
                        sample.update(zip(FEATURES, map(float, vector)))
                        rows.append(sample)
                chunk = pd.DataFrame(rows)
                chunk.to_csv(cache / (seq + ".csv"), index=False)
                chunks.append(chunk)
                print("Collected fixed policy:", seq, len(episode), "frames", flush=True)
        finally:
            renderer.delete()
    samples = pd.concat(chunks, ignore_index=True)
    validate_samples(samples, selected)
    samples.to_csv(cache / "samples.csv", index=False)
    for path, expected in assets.items():
        if sha(path) != expected:
            raise ValueError("CAD asset changed during collection: " + path)
    receipt = dict(status="smoke_only" if smoke else "fixed_policy_cache_complete", policy=POLICY,
                   manifest_sha256=cfg["manifest_sha256"], feature_source_sha256=sha(full["root"] / "source/2-risk_label.py"),
                   prior_source_sha256=sha(full["root"] / "source/b5_policy.py"), full_freeze_sha256=sha(full["root"] / "freeze.sha256"),
                   samples_sha256=sha(cache / "samples.csv"), candidate_count=len(samples), frame_count=len(selected),
                   train_bases=cfg["train_bases"], conditions=cfg["fit_conditions"], cad_hashes=assets,
                   collector_sha256=sha(__file__), gt_use="ADD-S labels only; never candidate generation")
    write(cache / "receipt.json", receipt)
    return samples, receipt


def use_cache(path, full, manifest):
    import pandas as pd
    cache = Path(path).resolve(strict=True)
    receipt = read(cache / "receipt.json")
    checks = dict(status="fixed_policy_cache_complete", policy=POLICY,
                  manifest_sha256=full["cfg"]["manifest_sha256"],
                  feature_source_sha256=sha(full["root"] / "source/2-risk_label.py"),
                  prior_source_sha256=sha(full["root"] / "source/b5_policy.py"),
                  full_freeze_sha256=sha(full["root"] / "freeze.sha256"),
                  samples_sha256=sha(cache / "samples.csv"), collector_sha256=sha(__file__))
    for key, value in checks.items():
        if receipt.get(key) != value:
            raise ValueError("Unverified or incompatible fixed-policy cache: " + key)
    data = pd.read_csv(cache / "samples.csv")
    validate_samples(data, manifest)
    return data, receipt


def fit_and_report(frozen, q0, full, fixed, manifest, output):
    np, pd = frozen.np, frozen.pd
    cfg = full["cfg"]
    d0 = frozen._hypothesis_samples_from_rollout(pd.read_csv(q0["labels"]))
    d1 = frozen._hypothesis_samples_from_rollout(pd.read_csv(full["labels"]))
    for data in (d0, d1):
        validate_samples(data, manifest)
        diameters = data.set_index(KEYS).D_obj_cm.sort_index()
        if not np.allclose(diameters, fixed.set_index(KEYS).D_obj_cm.sort_index(), rtol=1e-10, atol=1e-10):
            raise ValueError("Candidate populations disagree on CAD normalization")
    models, counts = {}, []
    for name, release in (("frozen_q0", q0), ("frozen_q1", full)):
        root = release["root"] / "artifacts"
        models[name] = (frozen.joblib.load(root / "shared_pose_quality_scaler.joblib"),
                        frozen.joblib.load(root / "shared_pose_quality_model.joblib"),
                        frozen.joblib.load(root / "shared_risk_calibrator.joblib"), release["cfg"]["p_risk_threshold"])
    obs = fixed[fixed.hypothesis == "obs"].copy()
    arms = {"fixed_obs_only": obs, "fixed_obs_duplicated_count_control": pd.concat([obs, obs], ignore_index=True),
            "fixed_obs_prior": fixed, "q0_policy_obs_prior_refit": d0}
    model_dir = output / "models"
    model_dir.mkdir()
    for name, data in arms.items():
        print("Fitting lightweight control:", name, len(data), "candidates", flush=True)
        scaler, regressor, calibrator, cut, fit_metrics = frozen.fit_shared_quality_model(
            data, cfg["risk_threshold_cm"], cfg["train_fraction"])
        models[name] = (scaler, regressor, calibrator, cut)
        folder = model_dir / name
        folder.mkdir()
        for kind, value in (("scaler", scaler), ("model", regressor), ("calibrator", calibrator)):
            frozen.joblib.dump(value, folder / (kind + ".joblib"))
        data_split = frozen._assign_temporal_split(data, cfg["train_fraction"])
        count = dict(model=name, candidates=len(data), fit_candidates=int((data_split.split == "train").sum()),
                     calibration_candidates=int((data_split.split == "cal").sum()),
                     observation_candidates=int((data.hypothesis == "obs").sum()),
                     prior_candidates=int((data.hypothesis == "prior").sum()))
        counts.append(count)
        write(folder / "config.json", dict(name=name, feature_columns=FEATURES, p_risk_threshold=cut,
              risk_threshold_cm=cfg["risk_threshold_cm"], train_fraction=cfg["train_fraction"], seed=cfg["seed"],
              training_counts=count, fit_metrics=fit_metrics, training_population="fixed_policy" if name.startswith("fixed_") else "cached_q0_policy",
              status="development_control_not_original_frozen_release"))
    pd.DataFrame(counts).to_csv(output / "training_counts.csv", index=False)
    reproduction = []
    for original, repeated, data in (("frozen_q0", "fixed_obs_only", obs),
                                     ("frozen_q1", "q0_policy_obs_prior_refit", d0)):
        predictions = []
        for name in (original, repeated):
            scaler, model, _, _ = models[name]
            predictions.append(np.maximum(model.predict(scaler.transform(data[FEATURES].to_numpy(dtype=float))), 0.)
                               * data.D_obj_cm.to_numpy())
        difference = np.abs(predictions[0] - predictions[1])
        reproduction.append(dict(anchor=original, new_control=repeated,
                                 max_prediction_difference_cm=float(difference.max()),
                                 mean_prediction_difference_cm=float(difference.mean()),
                                 note="Diagnostic, not assumed bitwise-identical; q0 cached rollout may differ from original q1 fit trajectory"))
    pd.DataFrame(reproduction).to_csv(output / "anchor_reproduction.csv", index=False)
    rows, sequence_rows = [], []
    from sklearn.metrics import balanced_accuracy_score
    for population, data in (("fixed_policy", fixed), ("q0_policy", d0), ("q1_policy", d1)):
        split = frozen._assign_temporal_split(data, cfg["train_fraction"])
        for name, (scaler, model, calibrator, cut) in models.items():
            error = np.maximum(model.predict(scaler.transform(split[FEATURES].to_numpy(dtype=float))), 0.) * split.D_obj_cm.to_numpy()
            probability = np.asarray(calibrator.predict(error))
            for partition in ("all", "train", "cal"):
                mask = np.ones(len(split), dtype=bool) if partition == "all" else split.split.to_numpy() == partition
                def measure(selected, sequence="ALL"):
                    truth = split.target_E_cm.to_numpy()[selected]
                    predicted, prob = error[selected], probability[selected]
                    risky = truth > cfg["risk_threshold_cm"]
                    two_classes = len(np.unique(risky)) == 2
                    return dict(model=name, candidate_population=population, partition=partition, sequence=sequence,
                                candidates=int(selected.sum()), mae_cm=float(np.mean(np.abs(truth - predicted))),
                                risk_auroc=float(frozen.roc_auc_score(risky, prob)) if two_classes else None,
                                balanced_accuracy=float(balanced_accuracy_score(risky, prob > cut)) if two_classes else None,
                                evidence="post_development_diagnostic; calibration partition used for fitting calibrator")
                rows.append(measure(mask))
                for sequence in split.sequence.unique():
                    sequence_rows.append(measure(mask & (split.sequence.to_numpy() == sequence), sequence))
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(sequence_rows).to_csv(output / "sequence_metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q0", required=True)
    parser.add_argument("--q1", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixed-policy-cache")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Render four development frames; do not fit or claim completion")
    args = parser.parse_args()
    if args.smoke and (args.check_only or args.fixed_policy_cache):
        parser.error("--smoke cannot be combined with cache reuse or --check-only")
    q0, full = frozen_release(args.q0, 0), frozen_release(args.q1, 1)
    if read(q0["root"] / "ablation_parent.json")["parent_freeze_sha256"] != sha(full["root"] / "freeze.sha256"):
        raise ValueError("q0 is not derived from the supplied q1 release")
    for key in ("manifest_sha256", "train_bases", "fit_conditions", "train_fraction", "seed", "risk_threshold_cm", "feature_columns"):
        if q0["cfg"][key] != full["cfg"][key]:
            raise ValueError("q0/q1 disagree: " + key)
    output = Path(args.output).resolve()
    if output.exists() or any(output == r["root"] or r["root"] in output.parents for r in (q0, full)):
        raise ValueError("Output must be NEW and outside frozen releases")
    if Path(sys.executable).resolve() != Path(full["effective"]["paths"]["TRAIN_PYTHON"]).resolve():
        raise ValueError("Use the exact frozen TRAIN_PYTHON")
    frozen = load_frozen_module(full)
    frozen.np.random.seed(full["cfg"]["seed"])
    if hasattr(frozen.o3d.utility, "random"):
        frozen.o3d.utility.random.seed(full["cfg"]["seed"])
    manifest = frozen.pd.read_csv(full["root"] / "reference_manifest.csv")
    expected = {b + c for b in full["cfg"]["train_bases"] for c in full["cfg"]["fit_conditions"]}
    manifest = manifest[manifest.sequence.isin(expected)].copy()
    if set(manifest.sequence) != expected or set(manifest.base_sequence) != set(full["cfg"]["train_bases"]):
        raise ValueError("Wrong development-only manifest selection")
    for sequence in expected:
        frozen.get_episode_df(manifest, sequence)
    output.mkdir(parents=True)
    shutil.copy2(__file__, output / "fixed_policy_control.py")
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), q0=str(q0["root"]), q1=str(full["root"]),
                policy=POLICY, development_frames=len(manifest), development_episodes=len(expected),
                fixed_policy_cache=args.fixed_policy_cache, test_sequences_used=False, backbone_retrained=False,
                fitted_parts="RobustScaler, HuberRegressor, isotonic and development risk threshold only",
                limitation="Offline development attribution, not independent generalization or closed-loop tracking evidence")
    write(output / "plan.json", plan)
    if args.check_only:
        paths = full["effective"]["paths"]
        frozen.verify_manifest_artifacts(manifest, SimpleNamespace(data_dir=paths["DATASET_ROOT"],
                                        ycb_dir=paths["GT_ROOT"], res_dir=paths["RESULT_ROOT"]))
        if args.fixed_policy_cache:
            use_cache(args.fixed_policy_cache, full, manifest)
        write(output / "CHECKED.json", dict(status="input_check_only_no_collection_or_fit"))
        print(output, flush=True)
        return
    if args.fixed_policy_cache:
        fixed, receipt = use_cache(args.fixed_policy_cache, full, manifest)
    else:
        fixed, receipt = collect(frozen, full, manifest, output / "fixed_policy_cache", args.smoke)
    write(output / "cache_receipt.json", receipt)
    if args.smoke:
        write(output / "SMOKE.json", dict(status="four_development_frames_verified_no_full_experiment", candidates=len(fixed)))
    else:
        fit_and_report(frozen, q0, full, fixed, manifest, output)
    verified = dict(q0["verified"], **full["verified"])
    for path, expected_hash in verified.items():
        if sha(path) != expected_hash:
            raise ValueError("Frozen input changed during experiment: " + path)
    write(output / "input_hashes.json", verified)
    if not args.smoke:
        write(output / "result_hashes.json", {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()})
        write(output / "COMPLETE.json", dict(status="development_fixed_policy_refit_attribution_complete",
              prior_exposure_control="completed", backbone_retrained=False, independent_test=False,
              development_frames=len(manifest), completed_utc=datetime.now(timezone.utc).isoformat()))
    print(output, flush=True)


if __name__ == "__main__":
    main()
