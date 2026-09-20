"""Prepare/seal a development-only release. No inference or fitting in this helper."""
import runtime_settings
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

BASES = ["mustard_easy_00_02","mustard0", "bleach0", "bleach_hard_00_03_chaitanya"]
CONDITIONS = ["_clean", "_black10", "_occ40", "_occ60", "_drop60"]
EXTENDED_CONDITIONS = CONDITIONS + ["_black10_2", "_black10_3", "_black10_4", "_black10_5"]
CAD = {"mustard_easy_00_02": "006_mustard_bottle", "mustard0": "006_mustard_bottle", "bleach0": "021_bleach_cleanser",
       "bleach_hard_00_03_chaitanya": "021_bleach_cleanser"}
FIELDS = ("rgb", "depth", "gt", "pred")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def configured_conditions():
    """Keep the old shell entry point compatible; the new Python runner uses nine."""
    selected = json.loads(os.environ.get("TRAIN_CONDITIONS_JSON", json.dumps(CONDITIONS)))
    if (not isinstance(selected, list) or not all(isinstance(c, str) for c in selected)
            or len(selected) != len(set(selected))
            or set(selected) not in (set(CONDITIONS), set(EXTENDED_CONDITIONS))):
        raise ValueError("Only the five-condition or nine-condition development protocol is supported")
    return selected


def select_rows(rows):
    conditions = configured_conditions()
    expected = {b + c for b in BASES for c in conditions}
    rows = [r.copy() for r in rows if r["sequence"] in expected]
    if {r["sequence"] for r in rows} != expected:
        raise ValueError(f"Manifest must contain all {len(expected)} development instances")
    frame_maps = {}
    for base in BASES:
        reference = None
        for suffix in conditions:
            seq = base + suffix
            group = sorted((r for r in rows if r["sequence"] == seq),
                           key=lambda r: int(r["sequence_index"]))
            if any(r["base_sequence"] != base for r in group):
                raise ValueError("Incorrect base_sequence: " + seq)
            if [int(r["sequence_index"]) for r in group] != list(range(len(group))):
                raise ValueError("Missing/duplicate sequence_index: " + seq)
            ids = [int(r["frame_id"]) for r in group]
            if len(ids) < 2 or len(set(ids)) != len(ids):
                raise ValueError("Too few frames or duplicate frame IDs: " + seq)
            identity = [(r["frame_id"], r["gt_sha256"]) for r in group]
            if reference is not None and identity != reference:
                raise ValueError("Conditions do not share the same original frame/GT mapping: " + base)
            reference = identity
            cut = max(1, min(len(group) - 1, int(len(group) * 0.7)))
            for i, r in enumerate(group):
                r["split"] = "train" if i < cut else "cal"
            frame_maps[seq] = group
    return [r for b in BASES for c in conditions for r in frame_maps[b + c]]

def prepare(generated_manifest=False):
    env = os.environ
    repo = Path(env["TRAIN_REPO"]).resolve()
    release = Path(env["RELEASE_DIR"]).resolve()
    conditions = configured_conditions()
    # A new runner creates only these three files before handing off preparation.
    # Never adopt a previous prepared/training/frozen directory.
    if generated_manifest:
        expected_files = {"reference_manifest.csv", "manifest_config.json", "sequence_inventory.json"}
        if (not release.is_dir() or {p.name for p in release.iterdir()} != expected_files
                or any(not (release / name).is_file() for name in expected_files)
                or Path(env["REFERENCE_MANIFEST"]).resolve() != release / "reference_manifest.csv"):
            raise ValueError("Generated-manifest preparation requires a new manifest-only run directory")
    else:
        release.mkdir(parents=True, exist_ok=False)
    source = release / "source"
    source.mkdir()
    for p in repo.glob("*.py"):
        shutil.copy2(p, source / p.name)
    shutil.copy2(repo / "run_train.sh", source / "run_train.sh")
    if (repo / "scripts").is_dir():
        shutil.copytree(repo / "scripts", source / "scripts", ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    manifest = Path(env["REFERENCE_MANIFEST"]).resolve()
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = select_rows(list(reader))
        columns = list(reader.fieldnames)
    roots = {"rgb": Path(env["DATASET_ROOT"]), "depth": Path(env["DATASET_ROOT"]),
             "gt": Path(env["GT_ROOT"]), "pred": Path(env["RESULT_ROOT"])}
    inventory = {}
    def record(p, expected=None):
        p = Path(p).resolve()
        if not p.is_file():
            raise FileNotFoundError(str(p))
        key = str(p)
        if key not in inventory:
            inventory[key] = sha(p)
        if expected is not None and inventory[key] != str(expected).lower():
            raise ValueError("Manifest hash mismatch; do not silently rebuild: " + key)
    from online_observer import build_config, asset_paths, validate_config
    from run_train import SEQUENCE_OBJECTS
    observer_path = release / "observer_config.json"
    observer_cfg = build_config(env, SEQUENCE_OBJECTS)
    dump(observer_path, observer_cfg)
    validate_config(observer_path)
    record(observer_path)
    for p in asset_paths(observer_cfg):
        record(p)
    record(manifest)
    if generated_manifest:
        record(release / "manifest_config.json")
        record(release / "sequence_inventory.json")
    for r in rows:
        for kind in FIELDS:
            raw = Path(r[kind + "_path"].replace("\\", "/"))
            p = raw if raw.is_absolute() else repo / raw
            if not p.is_file():
                p = roots[kind] / raw
            record(p, r[kind + "_sha256"])
            r[kind + "_path"] = str(p.resolve())
    # Detect omitted trailing frames as well as gaps in the manifest.
    for seq in {r["sequence"] for r in rows}:
        group = [r for r in rows if r["sequence"] == seq]
        for kind, suffix in (("rgb", "*.png"), ("depth", "*.png"),
                             ("gt", "*.txt"), ("pred", "*.txt")):
            folder = Path(group[0][kind + "_path"]).parent
            if len(list(folder.glob(suffix))) != len(group):
                raise ValueError("Manifest/directory frame count mismatch: " + str(folder))
    for base in BASES:
        record(Path(env["GT_ROOT"]) / base / "init_mask.png")
        cad = Path(env["CAD_MODEL_ROOT"]) / CAD[base]
        for required in ("points.xyz", "textured.obj"):
            record(cad / required)
        for p in cad.rglob("*"):
            if p.is_file():
                record(p)
    for key in ("TRAIN_PYTHON", "FOUNDATIONPOSE_PYTHON", "SAM2_PYTHON",
                "FOUNDATIONPOSE_REFINER_WEIGHT", "FOUNDATIONPOSE_SCORER_WEIGHT",
                "SAM2_CHECKPOINT"):
        record(env[key])
    record(Path(env["SAM2_DIR"]) / "sam2" / env["SAM2_CONFIG"])
    # Record external component source/config bytes, not just their Git labels.
    for key in ("FOUNDATIONPOSE_DIR", "SAM2_DIR"):
        root = Path(env[key])
        for pattern in ("*.py", "*.yaml", "*.yml"):
            for p in root.rglob(pattern):
                if not any(x in p.parts for x in (".git", "__pycache__", ".venv", "datasets")):
                    record(p)
    for p in source.rglob("*"):
        if p.is_file():
            record(p)
    training_manifest = manifest
    if not generated_manifest:
        # Backward compatibility for the old run_train.sh entry point only.
        training_manifest = release / "train_manifest.csv"
        with training_manifest.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows({k: r[k] for k in columns} for r in rows)
        with (release / "train_cal_split.csv").open("w", encoding="utf-8", newline="") as f:
            fields = ["base_sequence", "sequence", "sequence_index", "frame_id", "split"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: r[k] for k in fields} for r in rows)
        record(training_manifest)
        record(release / "train_cal_split.csv")
    # Persist selection/split boundaries as configuration, not another manifest.
    split_summary = []
    for seq in sorted({r["sequence"] for r in rows}):
        group = [r for r in rows if r["sequence"] == seq]
        cut = sum(r["split"] == "train" for r in group)
        split_summary.append({"sequence": seq, "frames": len(group),
                              "train_index_start": 0, "train_index_stop_exclusive": cut,
                              "cal_index_start": cut, "cal_index_stop_exclusive": len(group)})
    keys = ("TRAIN_PYTHON", "DATASET_ROOT", "GT_ROOT", "RESULT_ROOT", "CAD_MODEL_ROOT",
            "FOUNDATIONPOSE_PYTHON", "FOUNDATIONPOSE_DIR", "FOUNDATIONPOSE_REFINER_WEIGHT",
            "FOUNDATIONPOSE_SCORER_WEIGHT", "SAM2_PYTHON", "SAM2_DIR", "SAM2_CONFIG",
            "SAM2_CHECKPOINT", "PYOPENGL_PLATFORM", "PYTHONHASHSEED", "TRAIN_CONDITIONS_JSON")
    keys += ("SE3_PYTHON", "SE3_WEIGHT_ROOT", "SE3_DATA_ROOT")
    keys += ("B5_NUM_THREADS", "B5_IO_WORKERS", "SAM2_CACHE_ROOT") + runtime_settings.THREAD_KEYS
    config = {"status": "prepared_not_frozen", "created_utc": datetime.now(timezone.utc).isoformat(),
              "training_manifest": str(training_manifest.resolve()),
              "manifest_mode": "single_reference" if generated_manifest else "legacy_subset",
              "training_selection": split_summary,
              "train_bases": BASES, "conditions": conditions, "held_out_base": None,
              "train_fraction": 0.7, "on_policy_refine_rounds": 0, "seed": 42,
              "risk_threshold_cm": 1.0, "prior_advantage_margin_cm": 0.1,
              "blackout_min_frames": 10, "foundationpose_refine_iter": 5,
              "test_data_used": False, "test_data_used_for_fitting": False,
              "test_data_used_field_scope": "fitting only; not a claim of untouched method development",
              "method_revision_informed_by_previous_test_inspection": True,
              "manifest_contains_reserved_sequences": generated_manifest,
              "backbone_retrained": False,
              "observer_config_sha256": sha(observer_path),
              "perception_runtime_config": __import__("perception_runtime").CONFIG.copy(),
              "execution_settings": runtime_settings.execution_config(),
              "backbone_observations": "frozen predictions until correction; then restarted SE3 own-observation recursion",
              "bitwise_determinism_guaranteed": False,
              "paths": {k: env.get(k) for k in keys}}
    dump(release / "effective_config.json", config)
    record(release / "effective_config.json")
    dump(release / "input_hashes.json", inventory)
    for kind, python_key, repo_key in (("sam2", "SAM2_PYTHON", "SAM2_DIR"),
                                      ("foundationpose", "FOUNDATIONPOSE_PYTHON", "FOUNDATIONPOSE_DIR")):
        subprocess.run([env[python_key], '-B', str(source/'perception_workers.py'),
                        '--check', kind, env[repo_key]], check=True)
    for label, key in (("training", "TRAIN_PYTHON"), ("sam2", "SAM2_PYTHON"),
                       ("foundationpose", "FOUNDATIONPOSE_PYTHON"), ("se3", "SE3_PYTHON")):
        result = subprocess.run([env[key], "-m", "pip", "freeze"],
                                check=True, capture_output=True, text=True)
        (release / (label + "_pip_freeze.txt")).write_text(result.stdout, encoding="utf-8")
    # Import/config audit only; no model inference. All object cameras must match.
    for obj in observer_cfg['objects']:
        base = next(b for b, o in observer_cfg['sequence_objects'].items() if o == obj)
        subprocess.run([env['SE3_PYTHON'], '-B', str(source/'online_observer.py'),
                        '--check', str(observer_path), base], check=True)
    print("Preflight passed:", len(rows), "frames across", len(BASES) * len(conditions),
          "instances;", len(inventory), "verified files")

def seal():
    release = Path(os.environ["RELEASE_DIR"]).resolve()
    if (release / "FROZEN.json").exists():
        raise FileExistsError("Release is already sealed")
    inventory = json.loads((release / "input_hashes.json").read_text(encoding="utf-8"))
    for path, expected in inventory.items():
        if sha(path) != expected:
            raise ValueError("Input/source changed during training: " + path)
    artifacts = release / "artifacts"
    cfg = json.loads((artifacts / "shared_quality_config.json").read_text(encoding="utf-8"))
    from b5_revision import CONFIG
    if cfg.get('b5_policy_config') != CONFIG:
        raise ValueError('Cannot seal incompatible policy/model configuration')
    from perception_runtime import CONFIG as runtime_config
    effective_runtime = json.loads((release / 'effective_config.json').read_text(encoding='utf-8'))
    if cfg.get('execution_settings') != effective_runtime.get('execution_settings') or not cfg.get('execution_settings'):
        raise ValueError('Cannot seal mismatched CPU/I/O budgets')
    if (cfg.get('perception_runtime_config') != runtime_config
            or effective_runtime.get('perception_runtime_config') != runtime_config):
        raise ValueError('Cannot seal mismatched persistent perception configuration')
    if cfg.get('observer_config_sha256') != sha(release / 'observer_config.json'):
        raise ValueError('Wrong observer configuration')
    if cfg.get("training_mode") != "final_development_fit" or cfg["held_out_base"] is not None:
        raise ValueError("Refusing to seal a leave-one-out model as the final model")
    if set(cfg["train_bases"]) != set(BASES):
        raise ValueError("Unexpected training bases")
    if set(cfg.get("fit_conditions", [])) != set(configured_conditions()):
        raise ValueError("Unexpected development conditions")
    for key, expected in (("seed", 42), ("train_fraction", 0.7),
                          ("on_policy_refine_rounds", 0), ("risk_threshold_cm", 1.0),
                          ("prior_advantage_margin_cm", 0.1)):
        if cfg.get(key) != expected:
            raise ValueError("Unexpected final-fit parameter: " + key)
    effective_path = release / "effective_config.json"
    if effective_path.is_file():
        effective = json.loads(effective_path.read_text(encoding="utf-8"))
        training_manifest = Path(effective.get("training_manifest", release / "train_manifest.csv"))
    else:
        # Older prepared releases did not record the manifest path separately.
        training_manifest = release / "train_manifest.csv"
    if cfg["manifest_sha256"] != sha(training_manifest):
        raise ValueError("Wrong training manifest")
    for key, name in (("model", "shared_pose_quality_model.joblib"),
                      ("scaler", "shared_pose_quality_scaler.joblib"),
                      ("calibrator", "shared_risk_calibrator.joblib")):
        if sha(artifacts / name) != cfg[key + "_sha256"]:
            raise ValueError("Artifact/config mismatch: " + name)
    if not 0 < float(cfg["p_risk_threshold"]) < 1:
        raise ValueError("Invalid probability threshold")
    from sam2_episode_cache import verify_index
    cache_index = verify_index(release/'sam2_cache_index.json')
    if cache_index['manifest_sha256'] != sha(training_manifest):
        raise ValueError('SAM2 cache manifest differs from release')
    receipt = {
        "status": "final_development_model_frozen",
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "p_risk_threshold": cfg["p_risk_threshold"],
        "training_mode": cfg["training_mode"],
        "recovery_gate_config": cfg.get("recovery_gate_config"),
        "b5_policy_config": cfg.get("b5_policy_config"),
        "observer_config_sha256": cfg.get("observer_config_sha256"),
        "perception_runtime_config": cfg.get("perception_runtime_config"),
        "execution_settings": cfg.get("execution_settings"),
        "test_evaluation_completed": False,
        "note": "Model/source/input release only; new-test runner still requires adaptation."}
    files = []
    for p in release.rglob("*"):
        if p.is_file() and p.name != "freeze.sha256" and not any(
                x in p.relative_to(release).parts for x in ("sam2_cache", "recovery_debug", "__pycache__")):
            files.append(p)
    checksum = release / "freeze.sha256"
    receipt_bytes = (json.dumps(receipt, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    checksum.write_text("".join(sha(p) + "  " + p.relative_to(release).as_posix() + "\n"
                               for p in sorted(files))
                        + hashlib.sha256(receipt_bytes).hexdigest() + "  FROZEN.json\n",
                        encoding="utf-8")
    # The success marker is created only after validation and checksum generation.
    (release / "FROZEN.json").write_bytes(receipt_bytes)
    for p in list(artifacts.iterdir()) + [release / "FROZEN.json", checksum]:
        p.chmod(0o444)
    print("Frozen final-development release:", release)

if __name__ == "__main__":
    {"prepare": prepare, "prepare-generated": lambda: prepare(True), "seal": seal}[sys.argv[1]]()
