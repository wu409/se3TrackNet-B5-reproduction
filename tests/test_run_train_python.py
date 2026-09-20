import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_train_python_test", ROOT / "run_train.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
spec = importlib.util.spec_from_file_location("release_python_test", ROOT / "train_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class TestPythonTrainingRunner(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data, self.gt, self.results = [self.root / x for x in ("corrupted", "original", "predictions")]
        for base in runner.SEQUENCE_OBJECTS:
            (self.gt / base).mkdir(parents=True)
        for base in runner.SEQUENCE_OBJECTS:
            for kind in ("rgb", "depth", "annotated_poses"):
                folder = self.gt / base / kind
                folder.mkdir()
                for i in range(3):
                    name = f"{i:07d}.txt" if kind == "annotated_poses" else f"timestamp_{i:04d}.png"
                    (folder / name).write_bytes(f"fixture-{kind}-{i}".encode())
            for suffix in runner.CONDITIONS:
                seq = base + suffix
                for kind in ("rgb", "depth"):
                    folder = self.data / seq / kind
                    folder.mkdir(parents=True)
                    for i in range(3):
                        (folder / f"timestamp_{i:04d}.png").write_bytes(f"fixture-{kind}-{suffix}-{i}".encode())
                folder = self.results / base / seq
                folder.mkdir(parents=True)
                for i in range(3):
                    (folder / f"{i:07d}.txt").write_bytes(f"fixture-pose-{i}".encode())

    def argv(self, *more):
        return ["--dataset-root", str(self.data), "--gt-root", str(self.gt),
                "--result-root", str(self.results), "--release-dir", str(self.root / "output"), *more]

    def test_nine_sequences_are_not_nine_training_objects(self):
        inv = runner.sequence_inventory(self.gt)
        self.assertEqual(len(inv["sequences"]), 9)
        self.assertEqual(inv["known_object_identities"], 5)
        train = [r for r in inv["sequences"] if r["role"] == "development_training"]
        self.assertEqual(len(train), 4)
        easy = next(r for r in inv["sequences"] if r["sequence"] == "mustard_easy_00_02")
        self.assertEqual(easy["role"], "development_training")
        self.assertTrue(easy["quality_training_object_seen"])

    def test_missing_extra_blackout_prediction_stops(self):
        p = self.results / "mustard0/mustard0_black10_5/0000001.txt"
        p.unlink()
        with self.assertRaisesRegex(ValueError, "condition-specific predictions"):
            runner.preflight_frames(self.data, self.gt, self.results)

    def test_equal_counts_with_wrong_rgb_names_stop(self):
        folder = self.data / "bleach0_black10_3/rgb"
        (folder / "timestamp_0001.png").rename(folder / "wrong.png")
        with self.assertRaisesRegex(ValueError, "mismatched rgb"):
            runner.preflight_frames(self.data, self.gt, self.results)

    def test_original_image_copies_are_not_required(self):
        for base in runner.TRAIN_BASES:
            for kind in ("rgb", "depth"):
                for p in (self.gt / base / kind).glob("*.png"):
                    p.unlink()
        runner.preflight_frames(self.data, self.gt, self.results)

    def test_present_original_images_must_match_clean(self):
        folder = self.gt / "mustard0/rgb"
        (folder / "timestamp_0001.png").rename(folder / "wrong.png")
        with self.assertRaisesRegex(ValueError, "Original/clean rgb"):
            runner.preflight_frames(self.data, self.gt, self.results)

    def test_manifest_only_builds_all_nine_sequences_without_training(self):
        with mock.patch.dict(os.environ, {"TRAIN_PYTHON": os.sys.executable}):
            output = runner.main(self.argv("--manifest-only"))
        with (output / "reference_manifest.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 243)
        self.assertEqual(len({r["sequence"] for r in rows}), 81)
        self.assertEqual({r["base_sequence"] for r in rows}, set(runner.SEQUENCE_OBJECTS))
        self.assertTrue(all(r["pred_sha256"] and r["association_description"] for r in rows))
        self.assertEqual({p.name for p in output.iterdir()}, {
            "reference_manifest.csv", "manifest_config.json", "sequence_inventory.json"})
        content = (output / "reference_manifest.csv").read_bytes()
        with self.assertRaises(FileExistsError):
            runner.main(self.argv("--manifest-only"))
        self.assertEqual(content, (output / "reference_manifest.csv").read_bytes())

    def test_training_args_use_all_conditions_and_three_bases(self):
        args = runner.arguments(self.argv())
        env = runner.environment(args, self.root / "output")
        self.assertEqual(env['SAM2_CONFIG'], 'configs/sam2.1/sam2.1_hiera_s.yaml')
        self.assertTrue(env['SAM2_CHECKPOINT'].endswith('sam2.1_hiera_small.pt'))
        cmd = runner.training_command(env, self.root / "output")
        start = cmd.index("--corruption_lists") + 1
        self.assertEqual(cmd[start:start + 9], list(runner.CONDITIONS))
        self.assertEqual(json.loads(env["TRAIN_CONDITIONS_JSON"]), list(runner.CONDITIONS))
        for base in set(runner.SEQUENCE_OBJECTS) - set(runner.TRAIN_BASES):
            self.assertNotIn(base, cmd)
        self.assertIn("--final_fit", cmd)
        self.assertNotIn("--ci_episode", cmd)
        self.assertEqual(cmd[cmd.index("--manifest_path") + 1], env["REFERENCE_MANIFEST"])
        self.assertNotIn(str(self.root / "output/train_manifest.csv"), cmd)

    def test_shell_entry_builds_all_nine_and_propagates_failure(self):
        bash = shutil.which("bash")
        if not bash and Path("C:/Program Files/Git/bin/bash.exe").is_file():
            bash = "C:/Program Files/Git/bin/bash.exe"
        if not bash:
            self.skipTest("Bash is not installed")
        env = dict(os.environ, TRAIN_PYTHON=os.sys.executable,
                   PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        command = [bash, str(ROOT / "run_train.sh"), *self.argv("--manifest-only")]
        result = subprocess.run(command, cwd=self.root, env=env, capture_output=True,
                                text=True, encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = self.root / "output"
        self.assertEqual({p.name for p in output.glob("*.csv")}, {"reference_manifest.csv"})
        with (output / "reference_manifest.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual({r["base_sequence"] for r in rows}, set(runner.SEQUENCE_OBJECTS))
        self.assertEqual(len({r["sequence"] for r in rows}), 81)
        self.assertFalse((output / "FROZEN.json").exists())
        (self.results / "sugar_box1/sugar_box1_clean/0000001.txt").unlink()
        command += ["--release-dir", str(self.root / "failed_output")]
        result = subprocess.run(command, cwd=self.root, env=env, capture_output=True,
                                text=True, encoding="utf-8", errors="replace")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sugar_box1_clean", result.stdout + result.stderr)
        self.assertFalse((self.root / "failed_output").exists())

    def test_missing_reserved_sequence_prediction_stops_full_inventory(self):
        (self.results / "sugar_box1/sugar_box1_clean/0000001.txt").unlink()
        with self.assertRaisesRegex(ValueError, "sugar_box1_clean"):
            runner.preflight_frames(self.data, self.gt, self.results)

    def test_prepare_and_seal_use_only_reference_csv(self):
        # Exercise the real builder and release helper with synthetic CPU-only assets.
        with mock.patch.dict(os.environ, {"TRAIN_PYTHON": os.sys.executable}):
            output = runner.main(self.argv("--manifest-only"))
        args = runner.arguments(self.argv())
        env = runner.environment(args, output)
        for key in ("CAD_MODEL_ROOT", "FOUNDATIONPOSE_DIR", "SAM2_DIR"):
            env[key] = str(self.root / key.lower())
        for key in ("TRAIN_PYTHON", "FOUNDATIONPOSE_PYTHON", "SAM2_PYTHON",
                    "FOUNDATIONPOSE_REFINER_WEIGHT", "FOUNDATIONPOSE_SCORER_WEIGHT", "SAM2_CHECKPOINT"):
            p = self.root / "synthetic_assets" / key
            p.parent.mkdir(exist_ok=True)
            p.write_bytes(b"unit-test fixture only")
            env[key] = str(p)
        for base in runner.TRAIN_BASES:
            for name in ("init_mask.png", "cam_K.txt"):
                (self.gt / base / name).write_bytes(b"unit-test fixture only")
            cad = Path(env["CAD_MODEL_ROOT"]) / release.CAD[base]
            cad.mkdir(parents=True, exist_ok=True)
            for name in ("points.xyz", "textured.obj"):
                (cad / name).write_bytes(b"unit-test fixture only")
        from online_observer import build_config, asset_paths
        env.update(SE3_PYTHON=os.sys.executable, SE3_WEIGHT_ROOT=str(self.root/'se3weights'),
                   SE3_DATA_ROOT=str(self.root/'se3data'))
        for asset in asset_paths(build_config(env, runner.SEQUENCE_OBJECTS)):
            if asset.resolve() != Path(os.sys.executable).resolve():
                asset.parent.mkdir(parents=True,exist_ok=True)
                asset.write_bytes(b'unit-test asset')
        sam_config = Path(env["SAM2_DIR"]) / "sam2" / env["SAM2_CONFIG"]
        sam_config.parent.mkdir(parents=True)
        sam_config.write_text("unit-test fixture only")
        with mock.patch.dict(os.environ, env), mock.patch.object(
                release.subprocess, "run", return_value=mock.Mock(stdout="fixture-package==0\n")):
            release.prepare(generated_manifest=True)
        self.assertEqual({p.name for p in output.glob("*.csv")}, {"reference_manifest.csv"})
        effective = json.loads((output / "effective_config.json").read_text())
        self.assertEqual(effective["manifest_mode"], "single_reference")
        self.assertEqual(effective["training_manifest"], str(output / "reference_manifest.csv"))
        self.assertEqual({r["sequence"] for r in effective["training_selection"]},
                         {b+c for b in runner.TRAIN_BASES for c in runner.CONDITIONS})
        self.assertTrue(all(r["train_index_stop_exclusive"] == 2
                            and r["cal_index_stop_exclusive"] == 3
                            for r in effective["training_selection"]))
        from b5_revision import CONFIG
        cfg = dict(b5_policy_config=dict(CONFIG), observer_config_sha256=release.sha(output/'observer_config.json'),
                   perception_runtime_config=effective['perception_runtime_config'],
                   execution_settings=effective['execution_settings'],
                   training_mode="final_development_fit", held_out_base=None,
                   train_bases=list(runner.TRAIN_BASES), fit_conditions=list(runner.CONDITIONS),
                   seed=42, train_fraction=0.7, on_policy_refine_rounds=1,
                   risk_threshold_cm=1.0, prior_advantage_margin_cm=0.1, p_risk_threshold=0.35,
                   manifest_sha256=release.sha(output / "reference_manifest.csv"))
        artifacts = output / "artifacts"
        artifacts.mkdir()
        for key, name in (("model", "shared_pose_quality_model.joblib"),
                          ("scaler", "shared_pose_quality_scaler.joblib"),
                          ("calibrator", "shared_risk_calibrator.joblib")):
            p = artifacts / name
            p.write_bytes(b"synthetic artifact, not a real model")
            cfg[key + "_sha256"] = release.sha(p)
        release.dump(artifacts / "shared_quality_config.json", cfg)
        from cache_fixture import make_index
        make_index(output)
        with mock.patch.dict(os.environ, env), mock.patch.object(Path, "chmod"):
            release.seal()
        self.assertTrue((output / "FROZEN.json").is_file())
        self.assertEqual({p.name for p in output.glob("*.csv")}, {"reference_manifest.csv"})
        for line in (output / "freeze.sha256").read_text().splitlines():
            digest, name = line.split("  ", 1)
            self.assertEqual(release.sha(output / name), digest)

    def test_release_extended_selection_and_no_test_leakage(self):
        rows = [dict(base_sequence=b, sequence=b+c, sequence_index=str(i), frame_id=str(i),
                     gt_sha256="gt"+str(i))
                for b in runner.TRAIN_BASES for c in runner.CONDITIONS for i in range(10)]
        rows += [dict(sequence="sugar_box1_clean")]
        with mock.patch.dict(os.environ, TRAIN_CONDITIONS_JSON=json.dumps(runner.CONDITIONS)):
            selected = release.select_rows(rows)
        self.assertEqual(len(selected), 360)
        self.assertEqual(sum(r["split"] == "train" for r in selected), 252)

    def test_release_rejects_unexpected_or_duplicate_conditions(self):
        for bad in (["_clean"], list(runner.CONDITIONS) + ["_clean"], [1], {"_clean": 1}):
            with mock.patch.dict(os.environ, TRAIN_CONDITIONS_JSON=json.dumps(bad)):
                with self.assertRaises(ValueError):
                    release.configured_conditions()

    def test_generated_prepare_cannot_adopt_old_run(self):
        out = self.root / "output"
        out.mkdir()
        (out / "FROZEN.json").write_text("old")
        with mock.patch.dict(os.environ, TRAIN_REPO=str(ROOT), RELEASE_DIR=str(out),
                             REFERENCE_MANIFEST=str(out / "reference_manifest.csv")):
            with self.assertRaisesRegex(ValueError, "new manifest-only"):
                release.prepare(generated_manifest=True)
        self.assertEqual((out / "FROZEN.json").read_text(), "old")


if __name__ == "__main__":
    unittest.main()
