import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("test_release_driver", ROOT / "test_release.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
TREE = ast.parse((ROOT / "3-train_evaluation.py").read_text(encoding="utf-8"))


def functions(names, **namespace):
    ns = dict(np=np, os=os, json=json, hashlib=hashlib, close_episode_observers=lambda f: f,
              close_episode_perception=lambda f: f)
    ns.update(namespace)
    nodes = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "evaluation_test", "exec"), ns)
    return ns


class TestFrozenEvaluation(unittest.TestCase):
    def test_controls_are_gt_free_and_predeclared(self):
        f = functions({"decision_inputs"})["decision_inputs"]
        self.assertEqual(f("full", 2, 1, .8, .3), (2, 1, .8, .3))
        self.assertEqual(f("simple", 2, 1, .8, .3), (2, 1, .8, .3))
        for retired in ("no_absolute_gate", "no_relative_advantage"):
            with self.assertRaises(ValueError): f(retired, 2, 1, .8, .3)

    def test_ece_includes_zero_probability(self):
        f = functions({"compute_ece"})["compute_ece"]
        self.assertEqual(f(np.array([0.0]), np.array([1])), 1.0)

    def test_latency_requires_five_good_frames_and_censors_failure(self):
        ns = functions({"build_recovery_decomposition"}, calc_auc=lambda x: 0.0)
        f = ns["build_recovery_decomposition"]
        rows = f("s", [{"recovery_index": 0, "recovery_frame": 0}], [],
                 {"B5": np.zeros(60), "B1": np.ones(60)*2})
        self.assertEqual(rows[0]["B5_latency_confirmed_frames"], 4)
        self.assertEqual(rows[0]["B5_latency_success"], 1)
        self.assertEqual(rows[0]["B1_latency_censored"], 1)
        self.assertTrue(np.isnan(rows[0]["B1_latency_confirmed_frames"]))
        short = f("s", [{"recovery_index": 0, "recovery_frame": 0}], [], {"B5": np.zeros(4)})[0]
        self.assertEqual(short["B5_latency_censored"], 1)
        self.assertEqual(short["B5_window_complete"], 0)

    def test_frozen_preflight_never_reads_label_csv(self):
        pd = mock.Mock()
        pd.read_csv.side_effect = AssertionError("Test labels must not be read")
        cfg = dict(held_out_base=None, train_bases=sorted(driver.TRAIN), risk_threshold_cm=1,
                   prior_advantage_margin_cm=.1)
        load = mock.Mock(return_value=(cfg, None, None, None, .74))
        bind = mock.Mock()
        ns = functions({"main"}, pd=pd, o3d=mock.Mock(), load_shared_artifacts=load,
                       bind_frozen_quality_implementation=bind)
        ns["main"](argparse.Namespace(frozen_test=True, preflight_only=True, seed=42, csv_path="MUST_NOT_READ"))
        pd.read_csv.assert_not_called()
        bind.assert_called_once()

    def test_frozen_episode_runs_without_labels_or_gt_policy_inputs(self):
        frames = pd.DataFrame([dict(seq_idx=i, frame_idx=i, frame_id=i, rgb_path="rgb",
                                    gt_path="gt", pred_path="pred", depth_path="depth",
                                    base_sequence="sugar_box1") for i in range(6)])
        transition = mock.Mock(return_value=(np.eye(4), "MODE_1_ACCEPT", {}, None))
        ns = functions({"evaluate_episode", "decision_inputs", "calc_auc", "_latency_string", "build_recovery_decomposition"},
            RestartableObserver=lambda *a: mock.Mock(observe=lambda pose,*x: pose, source="precomputed", wall_ms=0.),
            PerceptionSession=lambda *a: mock.Mock(advance=lambda *a: None, wall_ms=0., index=0,
                                                   prefetch=lambda f, items: map(f, items)),
            pd=pd, time=time, cv2=mock.Mock(IMREAD_UNCHANGED=-1, imread=mock.Mock(return_value=np.ones((4,4))*1000)),
            load_episode_manifest=mock.Mock(return_value=frames), init_b5_state=lambda: {},
            load_foundationpose_recovery_rgb=lambda _: (np.zeros((4,4,3)), "rgb"),
            extract_pose_conditioned_features=lambda *a: {"x4": 0.0},
            predict_shared_quality=lambda *a: (0.01, 0.1, 0.2),
            compute_se3_prior=lambda *a: np.eye(4), se3_log_map=lambda *a: np.zeros(6),
            se3_exp_map=lambda *a: np.eye(4), b5_transition=transition, U=mock.Mock(adi=lambda *a: 0.001), K=np.eye(3))
        args = argparse.Namespace(frozen_test=True, manifest_path="reference", data_dir="data", gt_dir="gt",
            test_base_seq="sugar_box1", alpha=.5, blackout_min_frames=10, ycbineoat_root="gt",
            foundationpose_mesh_file="mesh", foundationpose_python="fp", foundationpose_dir="fp",
            foundationpose_refiner_weight="weight", foundationpose_refine_iter=5,
            sam2_python="sam2", sam2_dir="sam2", sam2_config="cfg", sam2_checkpoint="ckpt",
            sam2_cache_root="cache", observer_config="mock", prior_advantage_margin_cm=.1, risk_threshold=1.0, policy_variant="full")
        with mock.patch.object(np, "loadtxt", return_value=np.eye(4)), mock.patch.object(pd.DataFrame, "to_csv"):
            result = ns["evaluate_episode"](args, "sugar_box1_clean", None, np.zeros((5,3)), None, 10,
                                            None, None, None, None, None, None, .74)
        self.assertEqual(len(result[6]), 6)
        self.assertEqual(transition.call_count, 6)
        self.assertFalse(any("gt" in key.lower() for call in transition.call_args_list for key in call.kwargs))
        self.assertTrue(all(call.kwargs["p_risk_threshold"] == .74 for call in transition.call_args_list))


class TestTestRelease(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.release = self.root / "release"
        self.release.mkdir()
        self.data, self.gt, self.pred, self.cad = [self.root / k for k in ("data", "gt", "pred", "cad")]
        rows = []
        def put(p, data=b"synthetic unit test only"):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            return driver.sha(p)
        for base, obj in driver.TEST.items():
            put(self.gt / base / "init_mask.png")
            for name in ("points.xyz", "textured.obj"):
                put(self.cad / obj / name)
            for c in driver.CONDITIONS:
                for i in range(2):
                    row = dict(sequence=base+c, base_sequence=base, frame_id=str(i), sequence_index=str(i),
                               condition=c.strip("_"), association_method="official_ycbineoat_reference_sorted_index",
                               association_reference="fixture", association_description="fixture")
                    for kind, root, rel in (
                        ("rgb", self.data, Path(base+c) / "rgb" / (str(i)+".png")),
                        ("depth", self.data, Path(base+c) / "depth" / (str(i)+".png")),
                        ("gt", self.gt, Path(base) / "annotated_poses" / (str(i)+".txt")),
                        ("pred", self.pred, Path(base) / (base+c) / (str(i)+".txt"))):
                        row[kind+"_sha256"] = put(root / rel)
                        row[kind+"_name"] = rel.name
                        row[kind+"_path"] = rel.as_posix()
                    rows.append(row)
        with (self.release / "reference_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.paths = dict(DATASET_ROOT=str(self.data), GT_ROOT=str(self.gt), RESULT_ROOT=str(self.pred),
                          CAD_MODEL_ROOT=str(self.cad), TRAIN_PYTHON=sys.executable,
                          FOUNDATIONPOSE_PYTHON=sys.executable, SAM2_PYTHON=sys.executable, SE3_PYTHON=sys.executable,
                          FOUNDATIONPOSE_DIR=str(self.root / "fp"), FOUNDATIONPOSE_REFINER_WEIGHT="fixture-weight",
                          SAM2_DIR=str(self.root / "sam2"), SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_s.yaml",
                          SAM2_CHECKPOINT="fixture-checkpoint", SAM2_CACHE_ROOT=str(self.root/'cache'))
        self.cfg = dict(version="shared_pose_quality_v1", training_mode="final_development_fit",
                        perception_runtime_config=__import__('perception_runtime').CONFIG,
                        execution_settings=__import__('runtime_settings').execution_config(),
                        train_bases=sorted(driver.TRAIN), held_out_base=None, fit_conditions=list(driver.CONDITIONS),
                        feature_columns=["fixture"], risk_threshold_cm=1.0, prior_advantage_margin_cm=.1,
                        seed=42, p_risk_threshold=.74,
                        manifest_sha256=driver.sha(self.release / "reference_manifest.csv"))
        for key, name in (("model", "shared_pose_quality_model.joblib"), ("scaler", "shared_pose_quality_scaler.joblib"),
                          ("calibrator", "shared_risk_calibrator.joblib")):
            self.cfg[key+"_sha256"] = put(self.release / "artifacts" / name)
        from b5_revision import CONFIG
        from online_observer import VERSION
        driver.dump(self.release/"observer_config.json", dict(version=VERSION, python=sys.executable, objects={}, sequence_objects=driver.TEST))
        self.cfg.update(b5_policy_config=dict(CONFIG), observer_config_sha256=driver.sha(self.release/"observer_config.json"))
        driver.dump(self.release / "artifacts/shared_quality_config.json", self.cfg)
        self.effective = dict(paths=self.paths, observer_config_sha256=self.cfg["observer_config_sha256"], train_bases=sorted(driver.TRAIN), seed=42,
                              perception_runtime_config=self.cfg["perception_runtime_config"],
                              execution_settings=self.cfg['execution_settings'],
                              risk_threshold_cm=1.0, prior_advantage_margin_cm=.1,
                              blackout_min_frames=10, foundationpose_refine_iter=5)
        driver.dump(self.release / "effective_config.json", self.effective)
        driver.dump(self.release / "input_hashes.json", {})
        driver.dump(self.release / "FROZEN.json", {"status": "final_development_model_frozen"})
        for name in ("b5_policy.py", "2-risk_label.py", "3-train_evaluation.py", "b5_revision.py", "pose_safety.py", "online_observer.py", "predict.py", "matched_schedule.py"):
            put(self.release / "source" / name)
        for name in ("perception_runtime.py", "perception_workers.py", "runtime_settings.py", "ordered_prefetch.py", "sam2_episode_cache.py", "prepare_sam_cache.py"):
            put(self.release / "source" / name, (ROOT/name).read_bytes())
        from cache_fixture import make_index
        make_index(self.release)
        self.checksums()

    def checksums(self):
        files = sorted(p for p in self.release.rglob("*") if p.is_file() and p.name != "freeze.sha256")
        (self.release / "freeze.sha256").write_text("".join(
            driver.sha(p)+"  "+p.relative_to(self.release).as_posix()+"\n" for p in files), encoding="utf-8")

    def test_release_verification_and_input_audit(self):
        cfg, _ = driver.verify_release(self.release)
        inventory, n = driver.audit_test_inputs(self.release, self.paths)
        self.assertEqual(n, 90)
        self.assertEqual(cfg["p_risk_threshold"], .74)
        self.assertIn(str(self.gt / "sugar_box1/init_mask.png"), inventory)

    def test_changed_frozen_artifact_stops(self):
        (self.release / "artifacts/shared_pose_quality_model.joblib").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            driver.verify_release(self.release)

    def test_changed_test_input_stops(self):
        (self.data / "sugar_box1_clean/depth/0.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            driver.audit_test_inputs(self.release, self.paths)

    def test_missing_condition_stops(self):
        (self.pred / "sugar_box1/sugar_box1_black10_5/1.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            driver.audit_test_inputs(self.release, self.paths)

    def test_command_uses_frozen_paths_and_no_labels(self):
        cmd, work = driver.command_for(self.release, self.root / "out", self.cfg, self.effective, "sugar_box1", "full")
        self.assertIn("--frozen_test", cmd)
        self.assertNotIn("--csv_path", cmd)
        self.assertNotIn("--final_fit", cmd)
        self.assertEqual(cmd[cmd.index("--test_base_seq")+1], "sugar_box1")
        self.assertEqual(cmd[cmd.index("--sam2_config")+1], self.paths["SAM2_CONFIG"])
        self.assertEqual(cmd[cmd.index("--manifest_path")+1], str(self.release / "reference_manifest.csv"))
        self.assertEqual(cmd[cmd.index("--point_path")+1], str(self.cad / "004_sugar_box/points.xyz"))

    def test_retired_gate_transplant_is_rejected(self):
        before=driver.sha(self.release/'freeze.sha256')
        with self.assertRaises(SystemExit):
            driver.main(['--release',str(self.release),'--recovery-gate-revision','occlusion-aware-dev'])
        self.assertEqual(before,driver.sha(self.release/'freeze.sha256'))

    def test_old_model_policy_is_rejected_without_retrofit(self):
        self.cfg['b5_policy_config']={'version':'legacy'}
        (self.release/'artifacts/shared_quality_config.json').write_text(json.dumps(self.cfg))
        self.checksums()
        with self.assertRaisesRegex(ValueError,'newly trained four-mode'):
            driver.verify_release(self.release)

    def test_check_only_copies_frozen_policy_and_never_runs_inference(self):
        output = self.root / "out"
        before = driver.sha(self.release / "freeze.sha256")
        with mock.patch.object(driver.subprocess, "run", return_value=mock.Mock(returncode=0, stdout="fixture")), \
                mock.patch.object(driver, "scan_test_images", create=True, return_value={"passed": True, "errors": []}), \
                mock.patch.object(driver, "run_logged") as run:
            driver.main(["--release", str(self.release), "--output", str(output), "--check-only"])
        self.assertTrue((output / "CHECKED.json").is_file())
        self.assertFalse((output / "COMPLETE.json").exists())
        self.assertFalse((output / "test_manifest.csv").exists())
        self.assertEqual(driver.sha(output / "source/b5_policy.py"), driver.sha(self.release / "source/b5_policy.py"))
        self.assertEqual(driver.sha(self.release / "freeze.sha256"), before)
        self.assertEqual(run.call_count, 1)
        self.assertIn("--preflight_only", run.call_args[0][0])

    def test_frozen_model_loader_accepts_final_and_rejects_training_base(self):
        args = argparse.Namespace(frozen_test=True, test_base_seq="sugar_box1", train_seqs=sorted(driver.TRAIN),
                                  manifest_path=str(self.release / "reference_manifest.csv"),
                                  observer_config=str(self.release/"observer_config.json"), risk_threshold=1.0, prior_advantage_margin_cm=.1)
        for kind, name in (("model", "shared_pose_quality_model.joblib"), ("scaler", "shared_pose_quality_scaler.joblib"),
                           ("calibrator", "shared_risk_calibrator.joblib"), ("config", "shared_quality_config.json")):
            setattr(args, "shared_"+kind+"_path", str(self.release / "artifacts" / name))
        ns = functions({"load_shared_artifacts", "compute_full_sha256"}, joblib=mock.Mock(), SHARED_FEATURE_COLUMNS=["fixture"])
        result = ns["load_shared_artifacts"](args)
        self.assertEqual(result[-1], .74)
        args.test_base_seq = "mustard0"
        with self.assertRaisesRegex(ValueError, "Development sequence"):
            ns["load_shared_artifacts"](args)

    def test_aggregation_clusters_objects_not_frames(self):
        output = self.root / "aggregate"
        for variant in ("full", "simple"):
            for base in driver.TEST:
                work = output / variant / base
                work.mkdir(parents=True)
                for c in driver.CONDITIONS:
                    pd.DataFrame(dict(error_b5_ours_cm=[.1,.2] if variant == "full" else [1.,2.],
                        error_b1_obs_cm=[2.,2.], recovery_attempted=[0,1], both_candidates_bad=[0,1],
                        policy_wall_ms=[1.,20.], quality_wall_ms=[.5,.5], transition_wall_ms=[.5,19.5],
                        relocalization_attempted=[0,0], observer_wall_ms=[0.,0.], observer_restart=[0,0],
                        relocalization_generated=[0,0], relocalization_used=[0,0], output_uncertain=[0,0],
                        relocalization_wall_ms=[0.,0.], sam2_cache_hit=[0,0])).to_csv(work / ("checkpoint2_per_frame_"+base+c+"_log_threshold1.0.csv"), index=False)
                    pd.DataFrame(columns=["episode","frame_id"]).to_csv(work/("checkpoint2_relocalization_"+base+c+".csv"),index=False)
                pd.DataFrame([dict(row_type="event", recovery_index=2), dict(row_type="summary")]).to_csv(
                    work / ("checkpoint2_recovery_decomposition_"+base+"_threshold1.0.csv"), index=False)
                pd.DataFrame([dict(Metric="ECE", Value=.1)]).to_csv(
                    work / "checkpoint2_probability_calibration_metrics_threshold1.0.csv", index=False)
        driver.summarize(output, ["full", "simple"], 42)
        paired = pd.read_csv(output / "paired_comparisons.csv")
        clusters = paired[paired.base_sequence == "OBJECT_CLUSTER_MEAN"]
        self.assertEqual(len(clusters), 2)
        self.assertTrue((clusters.independent_objects == 3).all())
        self.assertEqual(len(pd.read_csv(output / "sequence_metrics.csv")), 10)
        self.assertEqual(len(pd.read_csv(output / "episode_metrics.csv")), 90)


if __name__ == "__main__":
    unittest.main()
