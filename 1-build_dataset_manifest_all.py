import argparse
import glob
import hashlib
import json
import os
import re
from collections import defaultdict

import pandas as pd

ASSOCIATION_METHOD = "official_ycbineoat_reference_sorted_index"
ASSOCIATION_REFERENCE = (
    "https://github.com/wenbowen123/iros20-6d-pose-tracking/blob/master/predict.py"
    "#L630-L650"
)
ASSOCIATION_DESCRIPTION = (
    "Replicates the YCBInEOAT reference loader: independently sort RGB, depth, "
    "and annotated-pose paths, then consume the same sequence index. The release "
    "provides no independent timestamp-to-GT mapping artifact. This is a frozen "
    "reference-loader protocol, not a claimed official timestamp mapping."
)
ARTIFACT_TYPES = ("rgb", "depth", "gt", "pred")
PATTERNS = {"rgb": "*.png", "depth": "*.png", "gt": "*.txt", "pred": "*.txt"}


def extract_frame_id(path):
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[-1]) if nums else -1


def compute_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _require_directory(path, label):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} directory does not exist: {path}")


def _official_sorted_files(directory, pattern, label):
    """Reproduce the reference loader's sorted(path) order."""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        raise FileNotFoundError(f"{label} directory contains no {pattern} files: {directory}")
    return sorted(files)  # Glob/directory enumeration order must not affect mapping.


def _extract_and_validate_pose_ids(files, label):
    """Detect duplicate IDs before an ID-keyed dictionary can overwrite them."""
    grouped = defaultdict(list)
    frame_ids = []
    for path in files:
        frame_id = extract_frame_id(path)
        if frame_id < 0:
            raise ValueError(f"{label} filename has no frame ID: {path}")
        grouped[frame_id].append(path)
        frame_ids.append(frame_id)
    duplicates = {fid: paths for fid, paths in grouped.items() if len(paths) > 1}
    if duplicates:
        details = {fid: [os.path.basename(p) for p in paths] for fid, paths in duplicates.items()}
        raise ValueError(f"{label} contains duplicate frame IDs: {details}")
    return frame_ids


def _expected_episodes(config):
    episodes = []
    for base_seq in config["base_sequences"]:
        conditions = list(config["common_conditions"])
        conditions.extend(config["extra_conditions"].get(base_seq, []))
        for condition in conditions:
            episodes.append((base_seq, condition, base_seq + condition))
    return episodes


def _episode_directories(dataset_root, gt_root, result_root, base_seq, sequence):
    return {
        "rgb": os.path.join(dataset_root, sequence, "rgb"),
        "depth": os.path.join(dataset_root, sequence, "depth"),
        "gt": os.path.join(gt_root, base_seq, "annotated_poses"),
        "pred": os.path.join(result_root, base_seq, sequence),
    }


def build_one_episode(dataset_root, gt_root, result_root, base_seq, condition):
    """Strictly build one episode from a trusted copy for one-time freezing."""
    sequence = base_seq + condition
    directories = _episode_directories(dataset_root, gt_root, result_root, base_seq, sequence)
    files = {}
    for artifact_type, directory in directories.items():
        _require_directory(directory, f"[{sequence}] {artifact_type}")
        files[artifact_type] = _official_sorted_files(
            directory, PATTERNS[artifact_type], f"[{sequence}] {artifact_type}"
        )
    gt_ids = _extract_and_validate_pose_ids(files["gt"], f"[{sequence}] GT")
    pred_ids = _extract_and_validate_pose_ids(files["pred"], f"[{sequence}] prediction")
    counts = {name: len(paths) for name, paths in files.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"[{sequence}] missing/extra input count mismatch: {counts}")
    if set(gt_ids) != set(pred_ids):
        raise ValueError(
            f"[{sequence}] GT/prediction frame-ID mismatch: "
            f"GT-only={sorted(set(gt_ids) - set(pred_ids))[:10]}, "
            f"prediction-only={sorted(set(pred_ids) - set(gt_ids))[:10]}"
        )

    rows = []
    roots = {"rgb": dataset_root, "depth": dataset_root, "gt": gt_root, "pred": result_root}
    for sequence_index, paths in enumerate(zip(*(files[name] for name in ARTIFACT_TYPES))):
        gt_frame_id = gt_ids[sequence_index]
        pred_frame_id = pred_ids[sequence_index]
        if gt_frame_id != pred_frame_id:
            raise ValueError(
                f"[{sequence}] sorted-index GT/prediction mismatch at index {sequence_index}: "
                f"GT={os.path.basename(paths[2])} ({gt_frame_id}), "
                f"prediction={os.path.basename(paths[3])} ({pred_frame_id})"
            )
        row = {
            "base_sequence": base_seq, "condition": condition.strip("_"),
            "sequence": sequence, "sequence_index": sequence_index, "frame_id": gt_frame_id,
            "association_method": ASSOCIATION_METHOD,
            "association_reference": ASSOCIATION_REFERENCE,
            "association_description": ASSOCIATION_DESCRIPTION,
        }
        for artifact_type, path in zip(ARTIFACT_TYPES, paths):
            row[f"{artifact_type}_name"] = os.path.basename(path)
            row[f"{artifact_type}_path"] = os.path.relpath(path, roots[artifact_type])
            row[f"{artifact_type}_sha256"] = compute_sha256(path)
        rows.append(row)
    return rows


def build_all_manifest(dataset_root, gt_root, result_root, config_path):
    """Build a reference once from a separately trusted dataset copy."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    rows = []
    for base_seq, condition, _ in _expected_episodes(config):
        rows.extend(build_one_episode(dataset_root, gt_root, result_root, base_seq, condition))
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("No manifest rows were generated")
    if manifest.duplicated(["sequence", "frame_id"]).any():
        raise RuntimeError("Generated manifest has duplicate (sequence, frame_id)")
    if manifest.duplicated(["sequence", "sequence_index"]).any():
        raise RuntimeError("Generated manifest has duplicate (sequence, sequence_index)")
    return manifest.sort_values(["sequence", "sequence_index"], kind="stable").reset_index(drop=True)


def _discover_runtime_episodes(dataset_root, result_root):
    episodes = set()
    if os.path.isdir(dataset_root):
        for name in os.listdir(dataset_root):
            path = os.path.join(dataset_root, name)
            if os.path.isdir(os.path.join(path, "rgb")) or os.path.isdir(os.path.join(path, "depth")):
                episodes.add(name)
    if os.path.isdir(result_root):
        for base_name in os.listdir(result_root):
            base_dir = os.path.join(result_root, base_name)
            if not os.path.isdir(base_dir):
                continue
            for name in os.listdir(base_dir):
                if os.path.isdir(os.path.join(base_dir, name)):
                    episodes.add(name)
    return episodes


def scan_runtime_inventory(reference_manifest, dataset_root, gt_root, result_root):
    """Independently scan runtime roots; do not derive actual paths from manifest rows."""
    inventory, scans = [], {}
    episode_info = reference_manifest[["sequence", "base_sequence"]].drop_duplicates()
    roots = {"rgb": dataset_root, "depth": dataset_root, "gt": gt_root, "pred": result_root}
    for episode in episode_info.itertuples(index=False):
        sequence, base_sequence = str(episode.sequence), str(episode.base_sequence)
        directories = _episode_directories(dataset_root, gt_root, result_root, base_sequence, sequence)
        scans[sequence] = {}
        for artifact_type, directory in directories.items():
            paths = sorted(glob.glob(os.path.join(directory, PATTERNS[artifact_type]))) \
                if os.path.isdir(directory) else []
            scans[sequence][artifact_type] = paths
            for actual_index, path in enumerate(paths):
                inventory.append({
                    "sequence": sequence, "base_sequence": base_sequence,
                    "artifact_type": artifact_type, "actual_sequence_index": actual_index,
                    "frame_id": extract_frame_id(path) if artifact_type in ("gt", "pred") else "",
                    "name": os.path.basename(path), "path": os.path.relpath(path, roots[artifact_type]),
                    "sha256": compute_sha256(path),
                })
    columns = ["sequence", "base_sequence", "artifact_type", "actual_sequence_index",
               "frame_id", "name", "path", "sha256"]
    return pd.DataFrame(inventory, columns=columns), scans


def _duplicates_by_frame_id(paths):
    grouped = defaultdict(list)
    for path in paths:
        grouped[extract_frame_id(path)].append(os.path.basename(path))
    return {str(fid): names for fid, names in grouped.items() if fid < 0 or len(names) > 1}


def _new_report(reference_manifest):
    return {
        "passed": False, "reference_rows": int(len(reference_manifest)),
        "reference_episodes": int(reference_manifest["sequence"].nunique()),
        "association_method": ASSOCIATION_METHOD, "association_reference": ASSOCIATION_REFERENCE,
        "limitation": ASSOCIATION_DESCRIPTION, "missing_episodes": [], "extra_episodes": [],
        "count_mismatches": [], "missing": [], "extra": [], "duplicates": [],
        "mismatched": [], "reordered": [], "sequence_index_errors": [], "hash_mismatches": [],
    }


def verify_runtime(reference_manifest, dataset_root, gt_root, result_root):
    """Compare frozen expected rows with an independent runtime scan."""
    required = {"base_sequence", "sequence", "sequence_index", "frame_id",
                "association_method", "association_reference", "association_description"}
    required.update(f"{kind}_{suffix}" for kind in ARTIFACT_TYPES
                    for suffix in ("name", "path", "sha256"))
    missing_columns = required - set(reference_manifest.columns)
    if missing_columns:
        raise ValueError(f"Reference manifest is missing columns: {sorted(missing_columns)}")
    report = _new_report(reference_manifest)
    if reference_manifest.duplicated(["sequence", "frame_id"]).any():
        report["duplicates"].append({"scope": "reference", "key": "sequence+frame_id"})
    if reference_manifest.duplicated(["sequence", "sequence_index"]).any():
        report["duplicates"].append({"scope": "reference", "key": "sequence+sequence_index"})
    methods = reference_manifest["association_method"].dropna().unique().tolist()
    if methods != [ASSOCIATION_METHOD]:
        report["mismatched"].append({"field": "association_method", "actual": methods})

    inventory, scans = scan_runtime_inventory(reference_manifest, dataset_root, gt_root, result_root)
    runtime_roots = {"rgb": dataset_root, "depth": dataset_root,
                     "gt": gt_root, "pred": result_root}
    expected_episodes = set(reference_manifest["sequence"].astype(str))
    actual_episodes = _discover_runtime_episodes(dataset_root, result_root)
    report["missing_episodes"] = sorted(expected_episodes - actual_episodes)
    report["extra_episodes"] = sorted(actual_episodes - expected_episodes)

    for sequence in sorted(expected_episodes):
        expected = reference_manifest[reference_manifest["sequence"].astype(str) == sequence].copy()
        expected["sequence_index"] = expected["sequence_index"].astype(int)
        expected = expected.sort_values("sequence_index", kind="stable").reset_index(drop=True)
        expected_indices = expected["sequence_index"].tolist()
        if expected_indices != list(range(len(expected))):
            report["sequence_index_errors"].append(
                {"sequence": sequence, "expected": list(range(len(expected))), "actual": expected_indices}
            )

        for artifact_type in ARTIFACT_TYPES:
            paths = scans[sequence][artifact_type]
            actual_names = [os.path.basename(path) for path in paths]
            expected_names = expected[f"{artifact_type}_name"].astype(str).tolist()
            if len(actual_names) != len(expected_names):
                report["count_mismatches"].append({"sequence": sequence,
                    "artifact_type": artifact_type, "expected": len(expected_names),
                    "actual": len(actual_names)})
            report["missing"].extend(f"{sequence}:{artifact_type}:{name}"
                for name in sorted(set(expected_names) - set(actual_names)))
            report["extra"].extend(f"{sequence}:{artifact_type}:{name}"
                for name in sorted(set(actual_names) - set(expected_names)))
            if set(actual_names) == set(expected_names) and actual_names != expected_names:
                report["reordered"].append({"sequence": sequence,
                    "artifact_type": artifact_type, "expected": expected_names, "actual": actual_names})
            if artifact_type in ("gt", "pred"):
                duplicates = _duplicates_by_frame_id(paths)
                if duplicates:
                    report["duplicates"].append({"sequence": sequence,
                        "artifact_type": artifact_type, "frame_ids": duplicates})
            actual_by_name = {os.path.basename(path): path for path in paths}
            for row in expected.itertuples(index=False):
                expected_name = str(getattr(row, f"{artifact_type}_name"))
                actual_path = actual_by_name.get(expected_name)
                if actual_path is None:
                    continue
                expected_path = os.path.normpath(str(getattr(row, f"{artifact_type}_path")))
                actual_relative_path = os.path.normpath(
                    os.path.relpath(actual_path, runtime_roots[artifact_type])
                )
                if actual_relative_path != expected_path:
                    report["mismatched"].append({
                        "sequence": sequence, "sequence_index": int(row.sequence_index),
                        "field": f"{artifact_type}_path", "expected": expected_path,
                        "actual": actual_relative_path,
                    })
                expected_hash = str(getattr(row, f"{artifact_type}_sha256")).lower()
                actual_hash = compute_sha256(actual_path)
                if actual_hash != expected_hash:
                    report["hash_mismatches"].append({"sequence": sequence,
                        "sequence_index": int(row.sequence_index), "artifact_type": artifact_type,
                        "name": expected_name, "expected": expected_hash, "actual": actual_hash})

        gt_ids = [extract_frame_id(path) for path in scans[sequence]["gt"]]
        pred_ids = [extract_frame_id(path) for path in scans[sequence]["pred"]]
        expected_ids = expected["frame_id"].astype(int).tolist()
        for label, actual_ids in (("gt", gt_ids), ("pred", pred_ids)):
            if set(actual_ids) != set(expected_ids):
                report["mismatched"].append({"sequence": sequence,
                    "field": f"{label}_frame_ids",
                    "missing": sorted(set(expected_ids) - set(actual_ids)),
                    "extra": sorted(set(actual_ids) - set(expected_ids))})
        limit = min([len(expected)] + [len(scans[sequence][kind]) for kind in ARTIFACT_TYPES])
        for index in range(limit):
            row = expected.iloc[index]
            actual_mapping = {f"{kind}_name": os.path.basename(scans[sequence][kind][index])
                              for kind in ARTIFACT_TYPES}
            actual_mapping.update({"frame_id": extract_frame_id(scans[sequence]["gt"][index]),
                                   "pred_frame_id": extract_frame_id(scans[sequence]["pred"][index])})
            expected_mapping = {f"{kind}_name": str(row[f"{kind}_name"])
                                for kind in ARTIFACT_TYPES}
            expected_mapping.update({"frame_id": int(row["frame_id"]),
                                     "pred_frame_id": int(row["frame_id"])})
            if actual_mapping != expected_mapping:
                report["mismatched"].append({"sequence": sequence, "sequence_index": index,
                    "expected_mapping": expected_mapping, "actual_mapping": actual_mapping})

    failure_fields = ["missing_episodes", "extra_episodes", "count_mismatches", "missing", "extra",
                      "duplicates", "mismatched", "reordered", "sequence_index_errors", "hash_mismatches"]
    report["runtime_inventory_rows"] = int(len(inventory))
    report["passed"] = not any(report[field] for field in failure_fields)
    return inventory, report


def write_hash_inventories_from_runtime(inventory, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    inventory[inventory["artifact_type"].isin(["rgb", "depth", "gt"])].to_csv(
        os.path.join(output_dir, "dataset_artifact_sha256.csv"), index=False)
    inventory[inventory["artifact_type"] == "pred"].to_csv(
        os.path.join(output_dir, "prediction_artifact_sha256.csv"), index=False)


def _write_parent(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["build-reference", "verify-runtime"])
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--gt_root", required=True)
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="dataset_manifest_all.csv")
    parser.add_argument("--reference_manifest", default=None)
    parser.add_argument("--runtime_inventory", default="runtime_inventory.csv")
    parser.add_argument("--verification_report", default="input_verification.json")
    parser.add_argument("--hash_inventory_dir", default=None)
    args = parser.parse_args(argv)

    if args.mode == "build-reference":
        if not args.config:
            parser.error("--config is required in build-reference mode")
        df = build_all_manifest(args.dataset_root, args.gt_root, args.result_root, args.config)
        _write_parent(args.output)
        df.to_csv(args.output, index=False)
        print(f"Frozen reference saved: {args.output}")
        print(f"Frames: {len(df)}, Episodes: {df.sequence.nunique()}")
    else:
        if not args.reference_manifest:
            parser.error("--reference_manifest is required in verify-runtime mode")
        reference = pd.read_csv(args.reference_manifest)
        inventory, report = verify_runtime(reference, args.dataset_root, args.gt_root, args.result_root)
        _write_parent(args.runtime_inventory)
        _write_parent(args.verification_report)
        inventory.to_csv(args.runtime_inventory, index=False)
        with open(args.verification_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")
        if args.hash_inventory_dir:
            write_hash_inventories_from_runtime(inventory, args.hash_inventory_dir)
        print(f"Runtime inventory saved: {args.runtime_inventory}")
        print(f"Verification report saved: {args.verification_report}")
        if not report["passed"]:
            print(json.dumps(report, indent=2, ensure_ascii=False))
            raise SystemExit(2)
        print("PASS: frozen reference and runtime inputs match exactly")
    print(f"Association method: {ASSOCIATION_METHOD}")
    print(f"Reference loader: {ASSOCIATION_REFERENCE}")
    print("[LIMITATION] No independent YCBInEOAT timestamp-to-GT mapping is claimed.")


if __name__ == "__main__":
    main()
