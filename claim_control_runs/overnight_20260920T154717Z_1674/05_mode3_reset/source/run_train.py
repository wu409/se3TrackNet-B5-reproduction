#!/usr/bin/env python3
"""Build one nine-sequence reference manifest, fit four bases, and seal the release.

Usage (activate b5-main first):
    python run_train.py --manifest-only  # no model environments or GPU required
    python run_train.py --check-only     # also validate assets/environments
    python run_train.py                  # manifest -> prepare -> fit -> seal

Each invocation owns a NEW directory. Existing runs and project-level manifests
are never overwritten. No corruption generation, backbone inference, or new-test
label generation is performed. The existing manifest builder requires pandas.
The reference covers all nine sequences x nine conditions; fitting selects only
TRAIN_BASES from this same CSV. No train/test manifest CSVs are generated.
"""
import argparse
import runtime_settings
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

REPO = Path(__file__).resolve().parent
TRAIN_BASES = ("mustard_easy_00_02","mustard0", "bleach0", "bleach_hard_00_03_chaitanya")
CONDITIONS = ("_clean", "_black10", "_black10_2", "_black10_3", "_black10_4",
              "_black10_5", "_occ40", "_occ60", "_drop60")
SEQUENCE_OBJECTS = {
    "bleach_hard_00_03_chaitanya": "021_bleach_cleanser",
    "bleach0": "021_bleach_cleanser",
    "cracker_box_reorient": "003_cracker_box",
    "cracker_box_yalehand0": "003_cracker_box",
    "mustard_easy_00_02": "006_mustard_bottle",
    "mustard0": "006_mustard_bottle",
    "sugar_box_yalehand0": "004_sugar_box",
    "sugar_box1": "004_sugar_box",
    "tomato_soup_can_yalehand0": "005_tomato_soup_can",
}


def absolute(value, root=REPO):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--manifest-only", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    p.add_argument("--release-dir", default=os.environ.get("RELEASE_DIR"),
                   help="NEW output directory; defaults to final_training_releases/train_<UTC>_<id>")
    for option, env, default in (
        ("dataset-root", "DATASET_ROOT", "datasets/YCBInEOAT_Corrupted"),
        ("gt-root", "GT_ROOT", "datasets/YCBInEOAT"),
        ("result-root", "RESULT_ROOT", "results_collection"),
        ("cad-model-root", "CAD_MODEL_ROOT", "datasets/YCB_Video_Models/CADmodels"),
        ("training-python", "TRAIN_PYTHON", sys.executable),
        ("se3-python", "SE3_PYTHON", sys.executable),
        ("se3-weight-root", "SE3_WEIGHT_ROOT", "YCBInEOAT_weights"),
        ("se3-data-root", "SE3_DATA_ROOT", "datasets/YCBInEOAT_data"),
        ("foundationpose-python", "FOUNDATIONPOSE_PYTHON", "/root/autodl-tmp/conda-envs/foundationpose/bin/python"),
        ("foundationpose-dir", "FOUNDATIONPOSE_DIR", "/root/autodl-tmp/FoundationPose"),
        ("sam2-python", "SAM2_PYTHON", "/root/autodl-tmp/conda-envs/sam2/bin/python"),
        ("sam2-dir", "SAM2_DIR", "/root/autodl-tmp/sam2"),
        ("sam2-cache-root", "SAM2_CACHE_ROOT", "all_run/sam2_masks"),
    ):
        p.add_argument("--" + option, default=os.environ.get(env, default))
    p.add_argument("--sam2-checkpoint", default=os.environ.get("SAM2_CHECKPOINT"))
    p.add_argument("--sam2-config", default=os.environ.get("SAM2_CONFIG", runtime_settings.SAM2_DEFAULT_CONFIG))
    return p.parse_args(argv)


def environment(args, release):
    env = dict(os.environ)
    runtime_settings.configure_environment(env)
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONHASHSEED="42", TRAIN_REPO=str(REPO), RELEASE_DIR=str(release),
               TRAIN_CONDITIONS_JSON=json.dumps(CONDITIONS),
               REFERENCE_MANIFEST=str(release / "reference_manifest.csv"))
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for name, key in (
        ("dataset_root", "DATASET_ROOT"), ("gt_root", "GT_ROOT"),
        ("result_root", "RESULT_ROOT"), ("cad_model_root", "CAD_MODEL_ROOT"),
        ("training_python", "TRAIN_PYTHON"), ("foundationpose_python", "FOUNDATIONPOSE_PYTHON"),
        ("foundationpose_dir", "FOUNDATIONPOSE_DIR"), ("sam2_python", "SAM2_PYTHON"),
        ("sam2_dir", "SAM2_DIR"),
        ("sam2_cache_root", "SAM2_CACHE_ROOT"),
        ("se3_python", "SE3_PYTHON"), ("se3_weight_root", "SE3_WEIGHT_ROOT"),
        ("se3_data_root", "SE3_DATA_ROOT"),
    ):
        env[key] = str(absolute(getattr(args, name)))
    env["SAM2_CONFIG"] = args.sam2_config
    env["SAM2_CHECKPOINT"] = str(absolute(args.sam2_checkpoint)) if args.sam2_checkpoint else str(
        Path(env["SAM2_DIR"]) / "checkpoints" / runtime_settings.SAM2_DEFAULT_CHECKPOINT)
    runtime_settings.validate_sam_pair(env['SAM2_CONFIG'], env['SAM2_CHECKPOINT'])
    env["FOUNDATIONPOSE_REFINER_WEIGHT"] = str(Path(env["FOUNDATIONPOSE_DIR"]) /
        "weights/2023-10-28-18-33-37/model_best.pth")
    env["FOUNDATIONPOSE_SCORER_WEIGHT"] = str(Path(env["FOUNDATIONPOSE_DIR"]) /
        "weights/2024-01-11-20-02-45/model_best.pth")
    return env


def dump_new(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def sequence_inventory(gt_root):
    if not gt_root.is_dir():
        raise FileNotFoundError(f"GT/original-sequence directory missing: {gt_root}")
    rows = []
    for name, cad in SEQUENCE_OBJECTS.items():
        training = name in TRAIN_BASES
        rows.append({"sequence": name, "cad_object": cad,
                     "directory_present": (gt_root / name).is_dir(),
                     "role": "development_training" if training else "reserved_not_trained",
                     "quality_training_object_seen": cad in {SEQUENCE_OBJECTS[b] for b in TRAIN_BASES}})
    return {"expected_sequences": 9, "known_object_identities": 5,
            "train_bases": list(TRAIN_BASES), "training_conditions": list(CONDITIONS),
            "training_instances": len(TRAIN_BASES) * len(CONDITIONS),
            "sequences": rows,
            "unlisted_directories": sorted(p.name for p in gt_root.iterdir()
                                           if p.is_dir() and p.name not in SEQUENCE_OBJECTS),
            "note": "Reserved means excluded here, not certification of an untouched test set. "}


def preflight_frames(dataset_root, gt_root, result_root):
    """Require complete condition inputs with matching clean-frame names and GT counts."""
    issues = []
    for base in SEQUENCE_OBJECTS:
        reference = {}
        for kind in ("rgb", "depth"):
            folder = dataset_root / (base + "_clean") / kind
            reference[kind] = sorted(p.name for p in folder.glob("*.png"))
            if not reference[kind]:
                issues.append(f"Missing clean {kind} images: {folder}")
            # The trainer needs GT, K and init_mask from gt_root, not duplicate images.
            # If original images ARE present, also check their names for consistency.
            original = sorted(p.name for p in (gt_root / base / kind).glob("*.png"))
            if original and original != reference[kind]:
                issues.append(f"Original/clean {kind} filename mismatch: {base}")
        if reference["rgb"] != reference["depth"]:
            issues.append(f"Clean RGB/depth filename mismatch: {base}")
        gt_files = list((gt_root / base / "annotated_poses").glob("*.txt"))
        if len(gt_files) < 2 or len(gt_files) != len(reference["rgb"]):
            issues.append(f"Clean RGB/GT count mismatch: {base}: {len(reference['rgb'])}/{len(gt_files)}")
        for suffix in CONDITIONS:
            seq = base + suffix
            for kind in ("rgb", "depth"):
                folder = dataset_root / seq / kind
                actual = sorted(p.name for p in folder.glob("*.png"))
                if not actual or actual != reference[kind]:
                    issues.append(f"Missing or mismatched {kind}: {folder} "
                                  f"(actual={len(actual)}, clean={len(reference[kind])})")
            predictions = result_root / base / seq
            count = len(list(predictions.glob("*.txt")))
            if count != len(gt_files) or count < 2:
                issues.append(f"Missing/mismatched condition-specific predictions: {predictions} "
                              f"(actual={count}, GT={len(gt_files)})")
    if issues:
        raise ValueError("Cannot create a complete nine-sequence reference manifest:\n" + "\n".join(issues) +
                         "\nGenerate/fix those corruption inputs and their own backbone predictions first. "
                         "No clean-prediction substitution or frame truncation is performed.")


def run(command, env, cwd, log=None):
    print("Running:", " ".join(str(x) for x in command), flush=True)
    if log is None:
        subprocess.run(command, env=env, cwd=str(cwd), check=True)
        return
    # Open exclusively: never turn a previous training log into the current run.
    with log.open("x", encoding="utf-8") as stream:
        with subprocess.Popen(command, env=env, cwd=str(cwd), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = proc.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)


def training_command(env, release):
    args = [env["TRAIN_PYTHON"], "-u", "-B", str(release / "source/2-risk_label.py"),
            "--observer_config", str(release / "observer_config.json"),
            "--final_fit", "--seed", "42", "--manifest_path", env["REFERENCE_MANIFEST"],
            "--ycb_dir", env["GT_ROOT"], "--data_dir", env["DATASET_ROOT"],
            "--res_dir", env["RESULT_ROOT"], "--mesh_path_root", env["CAD_MODEL_ROOT"],
            "--target_seqs", *TRAIN_BASES,
            "--cad_models_seq", *(SEQUENCE_OBJECTS[b] for b in TRAIN_BASES),
            "--corruption_lists", *CONDITIONS,
            "--train_fraction", "0.7", "--on_policy_refine_rounds", "1",
            "--risk_threshold", "1.0", "--prior_advantage_margin_cm", "0.1",
            "--blackout_min_frames", "10", "--foundationpose_refine_iter", "5"]
    for option, key in (
        ("foundationpose_python", "FOUNDATIONPOSE_PYTHON"), ("foundationpose_dir", "FOUNDATIONPOSE_DIR"),
        ("foundationpose_refiner_weight", "FOUNDATIONPOSE_REFINER_WEIGHT"),
        ("sam2_python", "SAM2_PYTHON"), ("sam2_dir", "SAM2_DIR"),
        ("sam2_config", "SAM2_CONFIG"), ("sam2_checkpoint", "SAM2_CHECKPOINT"),
    ):
        args += ["--" + option, env[key]]
    args += ["--sam2_cache_root", env["SAM2_CACHE_ROOT"]]
    for option, name in (
        ("shared_model_out", "shared_pose_quality_model.joblib"),
        ("shared_scaler_out", "shared_pose_quality_scaler.joblib"),
        ("shared_calibrator_out", "shared_risk_calibrator.joblib"),
        ("shared_config_out", "shared_quality_config.json"),
    ):
        args += ["--" + option, str(release / "artifacts" / name)]
    return args


def main(argv=None):
    args = arguments(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    release = absolute(args.release_dir) if args.release_dir else (
        REPO / "final_training_releases" / f"train_{stamp}_{uuid.uuid4().hex[:8]}")
    env = environment(args, release)
    # Inventory/hash all nine; the trainer filters out the other six before fitting.
    inventory = sequence_inventory(Path(env["GT_ROOT"]))
    preflight_frames(Path(env["DATASET_ROOT"]), Path(env["GT_ROOT"]), Path(env["RESULT_ROOT"]))
    release.mkdir(parents=True, exist_ok=False)
    print("New training release:", release, flush=True)
    print("Reference: 9 sequences x 9 conditions = 81; training: 4 development sequences x 9 = 36.", flush=True)
    print("Other five sequences: no quality fitting, rollout, GT labels or evaluation.", flush=True)
    dump_new(release / "sequence_inventory.json", inventory)
    dump_new(release / "manifest_config.json", {
        "base_sequences": list(SEQUENCE_OBJECTS), "common_conditions": list(CONDITIONS), "extra_conditions": {}})
    run([env["TRAIN_PYTHON"], "-u", "-B", str(REPO / "1-build_dataset_manifest_all.py"),
         "--mode", "build-reference", "--dataset_root", env["DATASET_ROOT"],
         "--gt_root", env["GT_ROOT"], "--result_root", env["RESULT_ROOT"],
         "--config", str(release / "manifest_config.json"),
         "--output", env["REFERENCE_MANIFEST"]], env, REPO)
    # Independently confirm exact coverage; the builder also checks IDs and hashes.
    with Path(env["REFERENCE_MANIFEST"]).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    expected = {b + c for b in SEQUENCE_OBJECTS for c in CONDITIONS}
    if {r["sequence"] for r in rows} != expected or len({(r["sequence"], r["frame_id"]) for r in rows}) != len(rows):
        raise ValueError("Generated reference manifest does not exactly match the 81-instance inventory")
    print("REFERENCE_MANIFEST=" + env["REFERENCE_MANIFEST"], flush=True)
    print("Reference built from current inputs; hashes freeze bytes, not proof of an independently trusted source.", flush=True)
    if args.manifest_only:
        print("Manifest only: no model fitting, environment preflight or FROZEN marker.", flush=True)
        return release
    run([env["TRAIN_PYTHON"], "-B", str(REPO / "train_release.py"), "prepare-generated"], env, REPO)
    if args.check_only:
        print("Checks complete: no training and no FROZEN marker.", flush=True)
        return release
    run([env["TRAIN_PYTHON"], "-B", "-c",
         "import torch; assert torch.cuda.is_available(), 'GPU unavailable: enable GPU mode before training'"], env, release)
    run([env["SE3_PYTHON"], "-B", "-c",
         "import torch; assert torch.cuda.is_available(), 'SE3 observer Python has no usable CUDA'"], env, release)
    (release / "artifacts").mkdir()
    run([env["TRAIN_PYTHON"], "-u", "-B", str(release / "source/prepare_sam_cache.py"),
         "--release", str(release)], env, release, release / "sam2_precompute.log")
    run(training_command(env, release), env, release, release / "training.log")
    run([env["TRAIN_PYTHON"], "-B", str(release / "source/train_release.py"), "seal"], env, release)
    print("FROZEN MODEL:", release / "artifacts", flush=True)
    print("New-object testing has NOT been run. Use this release's model/config/source for the test runner.", flush=True)
    return release


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print("TRAINING RELEASE FAILED:", exc, file=sys.stderr)
        print("No successful freeze is claimed. Inspect the failed run; rerun in a NEW directory.", file=sys.stderr)
        sys.exit(1)
