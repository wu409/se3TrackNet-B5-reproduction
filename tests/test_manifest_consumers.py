import argparse
import ast
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_selected_functions(filename, names, namespace):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(ROOT / filename), "exec"), namespace)
    return namespace


class TestManifestConsumers(unittest.TestCase):
    def test_risk_consumer_checks_missing_and_sha256(self):
        ns = load_selected_functions(
            "2-risk_label.py", {"resolve_path", "compute_sha256", "verify_manifest_artifacts"},
            {"os": os, "hashlib": hashlib},
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("rgb", "depth", "gt", "pred"):
                (root / name).mkdir()
                (root / name / "frame.bin").write_bytes(name.encode())
            row = {"sequence": "s", "frame_id": 1}
            for name in ("rgb", "depth", "gt", "pred"):
                row[f"{name}_path"] = "frame.bin"
                row[f"{name}_sha256"] = hashlib.sha256(name.encode()).hexdigest()
            args = argparse.Namespace(data_dir=str(root / "depth"), ycb_dir=str(root / "gt"),
                                      res_dir=str(root / "pred"))
            # RGB and depth normally share data_dir; use absolute paths in this isolated test.
            row["rgb_path"] = str(root / "rgb" / "frame.bin")
            manifest = pd.DataFrame([row])
            ns["verify_manifest_artifacts"](manifest, args)
            (root / "pred" / "frame.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                ns["verify_manifest_artifacts"](manifest, args)

    def test_evaluation_consumer_checks_paths_and_sha256(self):
        ns = load_selected_functions(
            "3-train_evaluation.py", {"compute_full_sha256", "load_episode_manifest"},
            {"os": os, "hashlib": hashlib, "pd": pd},
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, gt, pred = root / "data", root / "gt", root / "pred"
            (data / "s" / "depth").mkdir(parents=True)
            gt.mkdir(); pred.mkdir()
            files = {
                "depth": data / "s" / "depth" / "000001.png",
                "gt": gt / "pose_000001.txt", "pred": pred / "pose_000001.txt",
            }
            for name, path in files.items():
                path.write_bytes(name.encode())
            manifest = root / "reference_manifest.csv"
            pd.DataFrame([{
                "sequence": "s", "sequence_index": 0, "frame_id": 1,
                "depth_path": "s/depth/000001.png", "gt_path": "base/annotated_poses/pose_000001.txt",
                "pred_path": "base/s/pose_000001.txt",
                "depth_sha256": hashlib.sha256(b"depth").hexdigest(),
                "gt_sha256": hashlib.sha256(b"gt").hexdigest(),
                "pred_sha256": hashlib.sha256(b"pred").hexdigest(),
                "association_method": "official_ycbineoat_reference_sorted_index",
                "association_reference": "reference-loader", "association_description": "frozen limitation",
            }]).to_csv(manifest, index=False)
            episode = ns["load_episode_manifest"](
                str(manifest), "s", str(data), str(gt), str(pred)
            )
            self.assertEqual([1], episode["frame_id"].tolist())
            files["gt"].unlink()
            with self.assertRaises(FileNotFoundError):
                ns["load_episode_manifest"](str(manifest), "s", str(data), str(gt), str(pred))

    def test_consumers_default_to_frozen_reference(self):
        for filename in ("2-risk_label.py", "3-train_evaluation.py"):
            text = (ROOT / filename).read_text(encoding="utf-8")
            self.assertIn("./reference_manifest.csv", text)
            self.assertNotIn("build_all_manifest(", text)


if __name__ == "__main__":
    unittest.main()
