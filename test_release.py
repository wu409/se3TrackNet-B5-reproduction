"""Audit and evaluate a frozen release. Python 3.8 compatible; no training calls.

The shell is the user entry point. Five confirmed untouched sequences are the
default test population. No inference is run by --check-only.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent
TRAIN = {"mustard0", "bleach0", "bleach_hard_00_03_chaitanya"}
TEST = {
    "cracker_box_reorient": "003_cracker_box",
    "cracker_box_yalehand0": "003_cracker_box",
    "sugar_box_yalehand0": "004_sugar_box",
    "sugar_box1": "004_sugar_box",
    "tomato_soup_can_yalehand0": "005_tomato_soup_can",
}
CONDITIONS = ("_clean", "_black10", "_black10_2", "_black10_3", "_black10_4",
              "_black10_5", "_occ40", "_occ60", "_drop60")
VARIANTS = ("full", "simple", "no_absolute_gate", "no_relative_advantage")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def verify_release(release):
    if read_json(release / "FROZEN.json").get("status") != "final_development_model_frozen":
        raise ValueError("Release is not successfully frozen")
    checked = set()
    for line in (release / "freeze.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        p = (release / name).resolve()
        if release not in p.parents or sha(p) != digest:
            raise ValueError("Frozen release hash/path mismatch: " + name)
        checked.add(name)
    required = {"FROZEN.json", "effective_config.json", "reference_manifest.csv",
                "source/b5_policy.py", "source/2-risk_label.py", "artifacts/shared_quality_config.json",
                "artifacts/shared_pose_quality_model.joblib", "artifacts/shared_pose_quality_scaler.joblib",
                "artifacts/shared_risk_calibrator.joblib", "input_hashes.json"}
    if not required <= checked:
        raise ValueError("Release checksum list lacks required files: " + str(sorted(required - checked)))
    cfg = read_json(release / "artifacts/shared_quality_config.json")
    effective = read_json(release / "effective_config.json")
    if (cfg.get("training_mode") != "final_development_fit" or cfg.get("held_out_base") is not None
            or set(cfg.get("train_bases", [])) != TRAIN or set(cfg.get("fit_conditions", [])) != set(CONDITIONS)):
        raise ValueError("Unexpected final-training population/conditions")
    if cfg.get("manifest_sha256") != sha(release / "reference_manifest.csv"):
        raise ValueError("Release does not use the nine-sequence reference manifest")
    for key, name in (("model", "shared_pose_quality_model.joblib"),
                      ("scaler", "shared_pose_quality_scaler.joblib"),
                      ("calibrator", "shared_risk_calibrator.joblib")):
        if cfg.get(key + "_sha256") != sha(release / "artifacts" / name):
            raise ValueError("Shared artifact/config mismatch: " + key)
    if not 0 < float(cfg["p_risk_threshold"]) < 1:
        raise ValueError("Invalid frozen risk probability threshold")
    if set(effective.get("train_bases", [])) != TRAIN:
        raise ValueError("Training configuration mismatch")
    for key in ("risk_threshold_cm", "prior_advantage_margin_cm", "seed"):
        if cfg[key] != effective[key]:
            raise ValueError("Frozen configs disagree: " + key)
    # Includes component weights/source and the original training inputs.
    # Do not silently replace changed components or regenerate their reference.
    for path, digest in read_json(release / "input_hashes.json").items():
        if sha(path) != digest:
            raise ValueError("Frozen input/component changed: " + path)
    return cfg, effective


def audit_test_inputs(release, paths):
    with (release / "reference_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    expected = {b+c for b in TEST for c in CONDITIONS}
    selected = [r for r in rows if r["sequence"] in expected]
    if {r["sequence"] for r in selected} != expected:
        raise ValueError("Reference missing test conditions; do not rebuild it after inspecting results")
    inventory = {}
    def record(p, digest=None):
        p = Path(p).resolve()
        key = str(p)
        if key not in inventory:
            inventory[key] = sha(p)
        if digest is not None and inventory[key] != digest.lower():
            raise ValueError("Test artifact hash mismatch: " + key)
    groups = {}
    for row in selected:
        groups.setdefault(row["sequence"], []).append(row)
    for base in TEST:
        reference_ids = None
        for condition in CONDITIONS:
            seq = base + condition
            group = sorted(groups[seq], key=lambda r: int(r["sequence_index"]))
            if any(r["base_sequence"] != base for r in group):
                raise ValueError("Wrong initial-mask base in reference: " + seq)
            if [int(r["sequence_index"]) for r in group] != list(range(len(group))):
                raise ValueError("Missing/duplicate frame indices: " + seq)
            ids = [(int(r["frame_id"]), r["gt_sha256"]) for r in group]
            if len(ids) < 2 or len({i for i, _ in ids}) != len(ids):
                raise ValueError("Invalid test frame IDs: " + seq)
            if reference_ids is not None and ids != reference_ids:
                raise ValueError("Conditions have different GT/frame order: " + base)
            reference_ids = ids
            folders = {}
            for r in group:
                if r["rgb_name"] != r["depth_name"]:
                    raise ValueError("RGB/depth filename mismatch: " + seq)
                for kind, root_key in (("rgb", "DATASET_ROOT"), ("depth", "DATASET_ROOT"),
                                       ("gt", "GT_ROOT"), ("pred", "RESULT_ROOT")):
                    relative = Path(r[kind + "_path"].replace("\\", "/"))
                    p = relative if relative.is_absolute() else Path(paths[root_key]) / relative
                    # Evaluator's GT/pred directories must resolve exactly these files.
                    expected_dir = (Path(paths["GT_ROOT"]) / base / "annotated_poses" if kind == "gt" else
                                    Path(paths["RESULT_ROOT"]) / base / seq if kind == "pred" else
                                    Path(paths["DATASET_ROOT"]) / seq / kind)
                    if p.resolve().parent != expected_dir.resolve():
                        raise ValueError("Unexpected reference path: " + str(p))
                    record(p, r[kind + "_sha256"])
                    folders[kind] = p.parent
            for kind, folder in folders.items():
                pattern = "*.png" if kind in ("rgb", "depth") else "*.txt"
                if len(list(folder.glob(pattern))) != len(group):
                    raise ValueError("Extra/missing files relative to frozen manifest: " + str(folder))
        record(Path(paths["GT_ROOT"]) / base / "init_mask.png")
        cad = Path(paths["CAD_MODEL_ROOT"]) / TEST[base]
        for name in ("points.xyz", "textured.obj"):
            record(cad / name)
        for p in cad.rglob("*"):
            if p.is_file():
                record(p)
    return inventory, len(selected)


def command_for(release, output, cfg, effective, base, variant):
    paths = effective["paths"]
    work = output / variant / base
    cmd = [sys.executable, "-u", "-B", str(output / "source/3-train_evaluation.py"),
           "--frozen_test", "--policy_variant", variant,
           "--manifest_path", str(release / "reference_manifest.csv"),
           "--train_seqs", "mustard0", "bleach0", "bleach_hard_00_03_chaitanya",
           "--test_base_seq", base, "--result_dir",
           *(str(Path(paths["RESULT_ROOT"]) / base / (base+c)) for c in CONDITIONS),
           "--gt_dir", str(Path(paths["GT_ROOT"]) / base / "annotated_poses"),
           "--ycbineoat_root", paths["GT_ROOT"], "--data_dir", paths["DATASET_ROOT"],
           "--point_path", str(Path(paths["CAD_MODEL_ROOT"]) / TEST[base] / "points.xyz"),
           "--foundationpose_mesh_file", str(Path(paths["CAD_MODEL_ROOT"]) / TEST[base] / "textured.obj"),
           "--risk_threshold", str(cfg["risk_threshold_cm"]),
           "--prior_advantage_margin_cm", str(cfg["prior_advantage_margin_cm"]),
           "--blackout_min_frames", str(effective["blackout_min_frames"]),
           "--foundationpose_refine_iter", str(effective["foundationpose_refine_iter"]),
           "--seed", str(cfg["seed"]), "--bootstrap_samples", "10000",
           "--sam2_cache_root", str(work / "sam2_cache")]
    for option, key in (("foundationpose_python", "FOUNDATIONPOSE_PYTHON"),
                        ("foundationpose_dir", "FOUNDATIONPOSE_DIR"),
                        ("foundationpose_refiner_weight", "FOUNDATIONPOSE_REFINER_WEIGHT"),
                        ("sam2_python", "SAM2_PYTHON"), ("sam2_dir", "SAM2_DIR"),
                        ("sam2_config", "SAM2_CONFIG"), ("sam2_checkpoint", "SAM2_CHECKPOINT")):
        cmd += ["--" + option, paths[key]]
    for kind, name in (("model", "shared_pose_quality_model.joblib"), ("scaler", "shared_pose_quality_scaler.joblib"),
                       ("calibrator", "shared_risk_calibrator.joblib"), ("config", "shared_quality_config.json")):
        cmd += ["--shared_" + kind + "_path", str(release / "artifacts" / name)]
    return cmd, work


def run_logged(command, work, env):
    with (work / "evaluation.log").open("x", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=str(work), env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as proc:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = proc.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)


def summarize(output, variants, seed):
    import numpy as np
    import pandas as pd
    episode_rows, recoveries, calibrations = [], [], []
    for variant in variants:
        for base, obj in TEST.items():
            work = output / variant / base
            for c in CONDITIONS:
                seq = base+c
                files = list(work.glob("checkpoint2_per_frame_" + seq + "_log_threshold*.csv"))
                if len(files) != 1:
                    raise ValueError("Missing/ambiguous episode output: " + seq)
                df = pd.read_csv(files[0])
                normal = df["recovery_attempted"] == 0
                def auc10(values):
                    thresholds = np.linspace(0, 10, 1000)
                    acc = [np.mean(np.asarray(values) <= t) for t in thresholds]
                    integrate = np.trapz if hasattr(np, "trapz") else np.trapezoid
                    return float(integrate(acc, thresholds) * 10)
                episode_rows.append(dict(variant=variant, base_sequence=base, object_id=obj, episode=seq,
                    frames=len(df), auc_percent=auc10(df["error_b5_ours_cm"]),
                    b1_auc_percent=auc10(df["error_b1_obs_cm"]),
                    failure_1cm_percent=float(np.mean(df["error_b5_ours_cm"] > 1.0)*100),
                    failure_2cm_percent=float(np.mean(df["error_b5_ours_cm"] > 2.0)*100),
                    both_candidates_bad_percent=float(df["both_candidates_bad"].mean()*100),
                    ordinary_policy_ms=float(df.loc[normal, "policy_wall_ms"].mean()),
                    quality_diagnostic_ms=float(df["quality_wall_ms"].mean()),
                    recovery_transition_ms=float(df.loc[~normal, "transition_wall_ms"].mean()),
                    sam2_cache_hits=int(df["sam2_cache_hit"].sum())))
            rec = pd.read_csv(next(work.glob("checkpoint2_recovery_decomposition_*.csv")))
            rec = rec[rec.row_type == "event"].copy()
            rec["variant"], rec["base_sequence"], rec["object_id"] = variant, base, obj
            recoveries.append(rec)
            cal = pd.read_csv(next(work.glob("checkpoint2_probability_calibration_metrics_*.csv")))
            cal["variant"], cal["base_sequence"] = variant, base
            calibrations.append(cal)
    episodes = pd.DataFrame(episode_rows)
    episodes.to_csv(output / "episode_metrics.csv", index=False)
    numeric = [c for c in episodes.select_dtypes(include=[np.number]).columns if c != "frames"]
    seqs = episodes.groupby(["variant", "base_sequence", "object_id"], as_index=False)[numeric].mean()
    seqs.to_csv(output / "sequence_metrics.csv", index=False)
    pd.concat(recoveries, ignore_index=True).to_csv(output / "recovery_events.csv", index=False)
    pd.concat(calibrations, ignore_index=True).to_csv(output / "calibration_metrics.csv", index=False)
    paired = []
    if "full" in variants:
        full = seqs[seqs.variant == "full"].set_index("base_sequence")
        for control in ["B1"] + [v for v in variants if v != "full"]:
            other = full if control == "B1" else seqs[seqs.variant == control].set_index("base_sequence")
            delta = full.auc_percent - (other.b1_auc_percent if control == "B1" else other.auc_percent)
            # First average conditions within sequence, then sequences within object.
            # Only THREE distinct test objects: intervals are exploratory, not frame-level inference.
            obj_delta = delta.groupby(full.object_id).mean().to_numpy()
            rng = np.random.default_rng(seed)
            boot = rng.choice(obj_delta, size=(10000, len(obj_delta)), replace=True).mean(axis=1)
            for base, value in delta.items():
                paired.append(dict(comparison="full-minus-"+control, base_sequence=base,
                                   delta_auc_percentage_points=float(value), aggregation="sequence_macro"))
            paired.append(dict(comparison="full-minus-"+control, base_sequence="OBJECT_CLUSTER_MEAN",
                delta_auc_percentage_points=float(obj_delta.mean()), ci_low=float(np.percentile(boot, 2.5)),
                ci_high=float(np.percentile(boot, 97.5)), independent_objects=len(obj_delta),
                aggregation="object_cluster_bootstrap_exploratory_n3"))
    pd.DataFrame(paired).to_csv(output / "paired_comparisons.csv", index=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", required=True, help="Exact completed training release; never auto-select latest")
    p.add_argument("--output", help="New output directory outside the frozen release")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = p.parse_args(argv)
    if len(args.variants) != len(set(args.variants)):
        raise ValueError("Duplicate variants")
    release = Path(args.release).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output).expanduser().resolve() if args.output else ROOT / "frozen_test_runs" / (stamp+"_"+uuid.uuid4().hex[:8])
    if output == release or release in output.parents or output.exists():
        raise ValueError("Output must be new and outside the frozen training release")
    print("Verifying frozen model/source/component/input hashes...", flush=True)
    cfg, effective = verify_release(release)
    paths = effective["paths"]
    if Path(sys.executable).resolve() != Path(paths["TRAIN_PYTHON"]).resolve():
        raise ValueError("Use the frozen training Python via TEST_PYTHON=" + paths["TRAIN_PYTHON"])
    print("Auditing 5 test sequences x 9 conditions without inference...", flush=True)
    inputs, nframes = audit_test_inputs(release, paths)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(release / "source", output / "source")
    # Adapt only evaluation; B5 and training-feature implementation stay byte-identical.
    shutil.copy2(ROOT / "3-train_evaluation.py", output / "source/3-train_evaluation.py")
    shutil.copy2(Path(__file__), output / "source/test_release.py")
    shutil.copy2(ROOT / "run_test.sh", output / "source/run_test.sh")
    sources = {str(x.resolve()): sha(x) for x in (output / "source").rglob("*") if x.is_file()}
    dump(output / "test_input_hashes.json", inputs)
    dump(output / "test_source_hashes.json", sources)
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True) if shutil.which("git") else None
    status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True) if git and git.returncode == 0 else None
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), training_release=str(release),
        reference_manifest_sha256=sha(release / "reference_manifest.csv"), test_sequences=TEST,
        conditions=list(CONDITIONS), variants=args.variants, frames_per_variant=nframes,
        p_risk_threshold=cfg["p_risk_threshold"], risk_threshold_cm=cfg["risk_threshold_cm"],
        prior_advantage_margin_cm=cfg["prior_advantage_margin_cm"],
        evaluation_adapter_commit=git.stdout.strip() if git and git.returncode == 0 else None,
        evaluation_adapter_git_status=status.stdout if status else None,
        training_commit="not recorded by original release; do not infer it from current HEAD",
        recovery_window="60 frames including first valid-depth frame after >=10-frame blackout",
        latency="confirm five consecutive errors <= risk threshold; unsuccessful events censored at window/sequence end",
        rejection_denominator="raw proposals generated; also report not-generated / all blackout exits",
        timing="measured quality + transition wall time; excludes offline GT scoring, backbone inference and data I/O; not end-to-end FPS",
        timing_simple="quality is computed for diagnostics but excluded from simple-policy time; no learned score affects simple decisions",
        test_gt="offline scoring only; first-frame init_mask is an allowed input",
        uncertainty="sequence/object clustered, never independent-frame CI; primary object bootstrap has only n=3",
        additional_ablations_pending=["stateful drift penalty/streak limit", "recovery admission gate"],
        environment={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "PYOPENGL_PLATFORM")})
    dump(output / "test_protocol.json", protocol)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
               PYTHONHASHSEED=str(cfg["seed"]), PYOPENGL_PLATFORM=paths.get("PYOPENGL_PLATFORM") or "egl")
    env.setdefault("OMP_NUM_THREADS", "1")
    for name, key in (("test", "TRAIN_PYTHON"), ("sam2", "SAM2_PYTHON"), ("foundationpose", "FOUNDATIONPOSE_PYTHON")):
        pip = subprocess.run([paths[key], "-m", "pip", "freeze"], check=True, capture_output=True, text=True, env=env)
        (output / (name+"_pip_freeze.txt")).write_text(pip.stdout, encoding="utf-8")
    preflight, _ = command_for(release, output, cfg, effective, next(iter(TEST)), "full")
    preflight.append("--preflight_only")
    run_logged(preflight, output, env)
    if args.check_only:
        dump(output / "CHECKED.json", {"status": "preflight_passed_no_inference", "frames": nframes})
        print("Checks passed; no testing/training executed. Output:", output)
        return output
    subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'GPU unavailable'"], check=True, env=env)
    for variant in args.variants:
        for base in TEST:
            cmd, work = command_for(release, output, cfg, effective, base, variant)
            work.mkdir(parents=True, exist_ok=False)
            dump(work / "command.json", cmd)
            print("Evaluating", variant, base, flush=True)
            run_logged(cmd, work, env)
    for path, digest in dict(inputs, **sources).items():
        if sha(path) != digest:
            raise ValueError("Test input/source changed during evaluation: " + path)
    verify_release(release)
    summarize(output, args.variants, cfg["seed"])
    dump(output / "COMPLETE.json", {"status": "frozen_test_complete", "training_release": str(release),
                                   "variants": args.variants, "test_sequences": list(TEST)})
    print("Frozen testing complete:", output)
    return output


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print("FROZEN TEST FAILED:", exc, file=sys.stderr)
        print("No complete result claimed; preserve logs and use a new output directory.", file=sys.stderr)
        sys.exit(1)
