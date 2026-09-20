import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_test", ROOT / "train_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
tree = ast.parse((ROOT / "2-risk_label.py").read_text(encoding="utf-8"))
scope = {}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name == "select_training_bases"], type_ignores=[]),
             "select_training_bases", "exec"), scope)
select_bases = scope["select_training_bases"]

def manifest_rows():
    return [dict(base_sequence=b, sequence=b+c, sequence_index=str(i),
                 frame_id=str(i), gt_sha256="gt"+str(i))
            for b in release.BASES for c in release.CONDITIONS for i in range(10)]

class TestFinalTraining(unittest.TestCase):
    def test_final_fit_uses_all_three_bases(self):
        args = types.SimpleNamespace(final_fit=True, target_seqs=release.BASES, ci_object="bleach0")
        held, bases = select_bases(args)
        self.assertIsNone(held)
        self.assertEqual(bases, release.BASES)

    def test_fold_mode_still_excludes_one(self):
        args = types.SimpleNamespace(final_fit=False, target_seqs=release.BASES, ci_object="bleach0")
        held, bases = select_bases(args)
        self.assertEqual(held, "bleach0")
        self.assertNotIn(held, bases)
        self.assertEqual(len(bases), len(release.BASES)-1)

    def test_new_test_base_cannot_enter_final_fit(self):
        args = types.SimpleNamespace(final_fit=True, target_seqs=release.BASES+["sugar_box1"])
        with self.assertRaises(ValueError):
            select_bases(args)

    def test_manifest_selects_only_development_and_aligns_splits(self):
        rows = manifest_rows() + [dict(sequence="sugar_box1_clean")]
        selected = release.select_rows(rows)
        self.assertEqual(len(selected), len(release.BASES)*50)
        self.assertEqual(sum(r["split"] == "train" for r in selected), len(release.BASES)*35)
        for r in selected:
            self.assertEqual(r["split"], "train" if int(r["sequence_index"]) < 7 else "cal")

    def test_missing_and_duplicate_frames_fail(self):
        rows = manifest_rows()
        for invalid in (rows[1:], rows + [rows[0].copy()]):
            with self.assertRaises(ValueError):
                release.select_rows(invalid)

    def test_cross_condition_gt_mismatch_fails(self):
        rows = manifest_rows()
        rows[10]["gt_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            release.select_rows(rows)

    def fixture(self, path, mode="final_development_fit"):
        artifacts = path / "artifacts"
        artifacts.mkdir()
        manifest = path / "train_manifest.csv"
        manifest.write_text("synthetic test only", encoding="utf-8")
        release.dump(path / "input_hashes.json", {str(manifest): release.sha(manifest)})
        from b5_revision import CONFIG
        from perception_runtime import CONFIG as runtime_config
        release.dump(path / "effective_config.json", {"perception_runtime_config": runtime_config,
                                                      "execution_settings": __import__('runtime_settings').execution_config(),
                                                      "training_manifest": str(manifest)})
        release.dump(path / "observer_config.json", {"fixture": True})
        cfg = dict(b5_policy_config=dict(CONFIG), perception_runtime_config=runtime_config,
                   execution_settings=__import__('runtime_settings').execution_config(), observer_config_sha256=release.sha(path/"observer_config.json"), training_mode=mode, held_out_base=None, train_bases=release.BASES,
                   fit_conditions=release.CONDITIONS, seed=42, train_fraction=0.7,
                   on_policy_refine_rounds=1, risk_threshold_cm=1.0,
                   prior_advantage_margin_cm=0.1, p_risk_threshold=0.35,
                   manifest_sha256=release.sha(manifest))
        for key, filename in (("model", "shared_pose_quality_model.joblib"),
                              ("scaler", "shared_pose_quality_scaler.joblib"),
                              ("calibrator", "shared_risk_calibrator.joblib")):
            p = artifacts / filename
            p.write_bytes(b"synthetic artifact; not a real model")
            cfg[key+"_sha256"] = release.sha(p)
        release.dump(artifacts / "shared_quality_config.json", cfg)
        from cache_fixture import make_index
        make_index(path, 'train_manifest.csv')

    def test_seal_checks_artifacts_and_writes_verifiable_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.fixture(path)
            with mock.patch.dict(os.environ, RELEASE_DIR=str(path)), mock.patch.object(Path, "chmod"):
                release.seal()
            self.assertTrue((path / "FROZEN.json").is_file())
            for line in (path / "freeze.sha256").read_text().splitlines():
                expected, filename = line.split("  ", 1)
                self.assertEqual(release.sha(path / filename), expected)

    def test_failed_checks_leave_no_frozen_marker(self):
        for failure in ("changed_input", "wrong_mode"):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                self.fixture(path, "leave_one_sequence_out" if failure == "wrong_mode" else "final_development_fit")
                if failure == "changed_input":
                    (path / "train_manifest.csv").write_text("changed", encoding="utf-8")
                with mock.patch.dict(os.environ, RELEASE_DIR=str(path)):
                    with self.assertRaises(ValueError):
                        release.seal()
                self.assertFalse((path / "FROZEN.json").exists())

if __name__ == "__main__":
    unittest.main()

