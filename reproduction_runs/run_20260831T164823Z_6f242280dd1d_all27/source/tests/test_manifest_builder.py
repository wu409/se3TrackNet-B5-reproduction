import glob as glob_module
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "manifest_builder", ROOT / "1-build_dataset_manifest_all.py"
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class ManifestFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.dataset = root / "dataset"
        self.gt = root / "gt"
        self.result = root / "result"
        self.sequence = "obj_clean"
        (self.dataset / self.sequence / "rgb").mkdir(parents=True)
        (self.dataset / self.sequence / "depth").mkdir(parents=True)
        (self.gt / "obj" / "annotated_poses").mkdir(parents=True)
        (self.result / "obj" / self.sequence).mkdir(parents=True)
        for frame in (1, 2, 3):
            (self.dataset / self.sequence / "rgb" / f"{frame:06d}.png").write_bytes(
                f"rgb-{frame}".encode()
            )
            (self.dataset / self.sequence / "depth" / f"{frame:06d}.png").write_bytes(
                f"depth-{frame}".encode()
            )
            pose = ("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n").encode()
            (self.gt / "obj" / "annotated_poses" / f"pose_{frame:06d}.txt").write_bytes(pose)
            (self.result / "obj" / self.sequence / f"pose_{frame:06d}.txt").write_bytes(pose)
        self.config = root / "config.json"
        self.config.write_text(json.dumps({
            "base_sequences": ["obj"], "common_conditions": ["_clean"],
            "extra_conditions": {"obj": []},
        }), encoding="utf-8")
        self.reference = builder.build_all_manifest(
            str(self.dataset), str(self.gt), str(self.result), str(self.config)
        )

    def tearDown(self):
        self.tmp.cleanup()

    def verify(self):
        return builder.verify_runtime(
            self.reference, str(self.dataset), str(self.gt), str(self.result)
        )[1]


class TestManifestBuilder(ManifestFixture):
    def test_matching_runtime_passes(self):
        report = self.verify()
        self.assertTrue(report["passed"], report)

    def test_missing_file_is_reported(self):
        (self.dataset / self.sequence / "depth" / "000002.png").unlink()
        report = self.verify()
        self.assertFalse(report["passed"])
        self.assertTrue(report["missing"])
        self.assertTrue(report["count_mismatches"])

    def test_duplicate_pose_frame_id_is_reported_before_overwrite(self):
        original = self.result / "obj" / self.sequence / "pose_000001.txt"
        (original.parent / "alternate_000001.txt").write_bytes(original.read_bytes())
        report = self.verify()
        self.assertFalse(report["passed"])
        self.assertTrue(report["duplicates"])

    def test_mismatched_frame_ids_are_reported(self):
        old = self.result / "obj" / self.sequence / "pose_000003.txt"
        old.rename(old.parent / "pose_000004.txt")
        report = self.verify()
        self.assertFalse(report["passed"])
        self.assertTrue(report["mismatched"])

    def test_glob_enumeration_reordering_does_not_change_mapping(self):
        real_glob = glob_module.glob
        with mock.patch.object(
            builder.glob, "glob", side_effect=lambda pattern: list(reversed(real_glob(pattern)))
        ):
            rebuilt = builder.build_all_manifest(
                str(self.dataset), str(self.gt), str(self.result), str(self.config)
            )
            _, report = builder.verify_runtime(
                self.reference, str(self.dataset), str(self.gt), str(self.result)
            )
        self.assertEqual(
            self.reference[["sequence_index", "frame_id", "rgb_name", "depth_name",
                            "gt_name", "pred_name"]].to_dict("records"),
            rebuilt[["sequence_index", "frame_id", "rgb_name", "depth_name",
                     "gt_name", "pred_name"]].to_dict("records"),
        )
        self.assertTrue(report["passed"], report)
        self.assertEqual([], report["reordered"])

    def test_changed_content_is_a_hash_mismatch(self):
        path = self.dataset / self.sequence / "depth" / "000002.png"
        path.write_bytes(b"changed")
        report = self.verify()
        self.assertFalse(report["passed"])
        self.assertTrue(report["hash_mismatches"])


if __name__ == "__main__":
    unittest.main()
