import argparse
import ast
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ablation_policy as policy
import ablation_release as driver
import ablation_summary as summary
import run_train
import train_release


class PolicyControls(unittest.TestCase):
    def test_no_quality_is_exact_simple_alias(self):
        tree = ast.parse((ROOT / "3-train_evaluation.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decision_inputs")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "controls", "exec"), namespace)
        f = namespace["decision_inputs"]
        for inputs in ((2, 1, .9, .2), (100, 0, 1, 0), (0, 0, 0, 0)):
            self.assertEqual(f("simple", *inputs), f("no_quality", *inputs))
            self.assertEqual(f("no_rollout", *inputs), inputs)
            self.assertEqual(f("no_recovery_admission", *inputs), inputs)

    def test_no_admission_ignores_depth_mask_and_large_translation(self):
        pose = np.eye(4)
        pose[0, 3] = 100.
        result = policy.ungated_recovery_validity(pose, visible_mask=None, depth_real=None)
        self.assertTrue(result["accepted_recovery"])
        self.assertEqual(result["recovery_rejection_reasons"], [])

    def test_no_admission_keeps_se3_safety(self):
        reflection = np.diag([-1., 1., 1., 1.])
        for bad in (None, np.zeros((3,3)), np.full((4,4), np.nan), np.zeros((4,4)), reflection):
            self.assertFalse(policy.ungated_recovery_validity(bad)["accepted_recovery"])

    def test_scoped_hook_restores_even_on_exception(self):
        original = mock.Mock(return_value={"accepted_recovery": False})
        ns = {"evaluate_recovery_pose_validity": original, "np": np}
        exec("def transition(fail=False):\n    result=evaluate_recovery_pose_validity(np.eye(4))\n    if fail: raise RuntimeError('fixture')\n    return result\n", ns)
        controlled = policy.without_recovery_admission(ns["transition"])
        self.assertTrue(controlled()["accepted_recovery"])
        self.assertIs(ns["evaluate_recovery_pose_validity"], original)
        with self.assertRaises(RuntimeError):
            controlled(True)
        self.assertIs(ns["evaluate_recovery_pose_validity"], original)
        original.assert_not_called()

    def test_actual_transition_uses_recovery_and_resets_history(self):
        from test_b5_revision import b5, pose
        def raw(**kwargs):
            result = b5.evaluate_recovery_pose_validity(pose(1))
            accepted = result["accepted_recovery"]
            return pose(1), accepted, dict(result, raw_recovery_generated=True)
        state = b5.init_b5_state()
        state["exited_blackout"] = True
        original = b5.evaluate_recovery_pose_validity
        with mock.patch.object(b5, "_resolve_init_mask_path", return_value=None), \
             mock.patch.object(b5, "_ensure_template1_cached", side_effect=lambda **kw: kw["state"]), \
             mock.patch.object(b5, "_execute_sam2_recovery", side_effect=raw) as recovery:
            result, _, state, _ = policy.without_recovery_admission(b5.b5_transition)(
                T_obs=pose(10), T_prior=pose(20), support=0., depth_real=np.ones((2,2)),
                model_pts=np.zeros((1,3)), K=np.eye(3), frame_index=11, frame_id=11, state=state,
                E_obs_hat_cm=10., E_prior_hat_cm=1., p_obs_risk=1., p_prior_risk=.1,
                p_risk_threshold=.74)
        self.assertEqual(recovery.call_count, 1)
        self.assertEqual(result[0,3], 1.)
        self.assertTrue(state["reset_motion_history"])
        self.assertIs(b5.evaluate_recovery_pose_validity, original)


class NoRolloutControls(unittest.TestCase):
    def sources(self):
        return ((ROOT / "2-risk_label.py").read_text(encoding="utf-8"),
                (ROOT / "train_release.py").read_text(encoding="utf-8"))

    def test_patches_only_validation_and_frozen_round_count(self):
        label, release = self.sources()
        revised_label, revised_release = driver.patch_zero_refit(label, release)
        self.assertEqual(label, revised_label.replace("if args.on_policy_refine_rounds < 0:",
            "if args.on_policy_refine_rounds < 1:").replace("on_policy_refine_rounds must be >= 0",
            "on_policy_refine_rounds must be >= 1"))
        self.assertEqual(release, revised_release.replace('"on_policy_refine_rounds": 0, "seed": 42',
            '"on_policy_refine_rounds": 1, "seed": 42').replace('("on_policy_refine_rounds", 0)',
            '("on_policy_refine_rounds", 1)'))
        ast.parse(revised_label, feature_version=(3,8))
        ast.parse(revised_release, feature_version=(3,8))

    def test_unknown_source_fails_closed(self):
        label, release = self.sources()
        with self.assertRaises(ValueError):
            driver.patch_zero_refit(label.replace("< 1:", "< 2:"), release)

    def test_real_refit_loop_zero_skips_all_refits_one_uses_only_three_bases(self):
        main = next(n for n in ast.parse(self.sources()[0]).body if isinstance(n, ast.FunctionDef) and n.name == "main")
        loop = next(n for n in main.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "round_idx")
        for rounds in (0,1):
            bases = ("mustard0", "bleach0", "bleach_hard_00_03_chaitanya")
            rollout, fit = mock.Mock(return_value=[{}]), mock.Mock(return_value=(None, None, None, .7, {}))
            ns = dict(args=argparse.Namespace(on_policy_refine_rounds=rounds,
                      corruption_lists=run_train.CONDITIONS, risk_threshold=1., train_fraction=.7),
                      train_bases=bases, base_to_idx=dict(zip(bases, range(3))),
                      get_episode_df=lambda manifest, seq: seq, manifest=None, rollout_episode=rollout,
                      models_pts=None, scenes=None, renders_obj=None, mesh_nodes=None, d_objs=None,
                      open3d_models=None, scaler=None, regressor=None, calibrator=None,
                      p_risk_threshold=.74, pd=pd, _hypothesis_samples_from_rollout=lambda x: x,
                      fit_shared_quality_model=fit, print=lambda *a: None)
            exec(compile(ast.Module(body=[loop], type_ignores=[]), "refit_loop", "exec"), ns)
            self.assertEqual(fit.call_count, rounds)
            self.assertEqual(rollout.call_count, rounds * 27)
            self.assertEqual({c.args[1] for c in rollout.call_args_list},
                {b+c for b in bases for c in run_train.CONDITIONS} if rounds else set())

    def configs(self):
        cfg = dict(on_policy_refine_rounds=1, seed=42, train_fraction=.7,
            risk_threshold_cm=1., prior_advantage_margin_cm=.1, b5_policy_config={"version":"new"},
            recovery_gate_config={"version":"new"}, manifest_sha256="same")
        effective = dict(cfg, paths={"TRAIN_PYTHON":sys.executable}, training_selection=["same"],
                         blackout_min_frames=10, foundationpose_refine_iter=5)
        return cfg, effective

    def test_q0_allows_own_calibration_but_not_changed_policy(self):
        full = self.configs()
        q0 = copy.deepcopy(full)
        q0[0].update(on_policy_refine_rounds=0, p_risk_threshold=.6)
        q0[1]["on_policy_refine_rounds"] = 0
        driver.assert_matched(full, q0)
        q0[0]["b5_policy_config"]["version"] = "changed"
        with self.assertRaisesRegex(ValueError, "b5_policy_config"):
            driver.assert_matched(full, q0)

    def test_new_output_cannot_overwrite_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            with self.assertRaises(ValueError):
                driver.new_output(str(parent / "child"), "fixture", parent)


class AggregationTests(unittest.TestCase):
    def test_pairing_uses_three_objects_not_frames_or_conditions(self):
        rows = []
        for variant in ("full", "no_rollout"):
            for base, obj in driver.evaluation.TEST.items():
                rows.append(dict(variant=variant, base_sequence=base, object_id=obj,
                    auc_percent=60. if variant == "full" else 50., b1_auc_percent=40.,
                    failure_1cm_percent=10., failure_2cm_percent=5.))
        paired = summary.paired_tracking(pd.DataFrame(rows), 42)
        macro = paired[paired.aggregation == "object_cluster_bootstrap_exploratory_n3"]
        self.assertTrue((macro.independent_objects == 3).all())
        row = macro[(macro.comparison == "full-minus-no_rollout") & (macro.metric == "auc_percent")].iloc[0]
        self.assertEqual(row.estimate, 10.)
        self.assertEqual(row.ci_low, 10.)

    def test_recovery_denominators_and_incomplete_window(self):
        row = dict(variant="full", base_sequence="s", generated_count=1, accepted_count=1,
            used_count=1, rejected_count=0, not_generated_count=0, raw_good_count=1,
            accepted_good_count=1, B5_latency_success=1, B5_latency_censored=0,
            B5_window_complete=1, B5_window_auc_percent=80.)
        bad = dict(row, generated_count=0, accepted_count=0, raw_good_count=0,
                   accepted_good_count=0, B5_latency_success=0, B5_latency_censored=1,
                   B5_window_complete=0, B5_window_auc_percent=0.)
        result = summary.recovery_summary(pd.DataFrame([row,bad])).iloc[0]
        self.assertEqual(result.raw_good_percent, 100.)
        self.assertEqual(result.successful_recovery_percent, 50.)
        self.assertEqual(result.B5_window_auc_percent, 80.)
        self.assertEqual(result.incomplete_window_events, 1)

    def test_missing_sequence_fails(self):
        with self.assertRaises(ValueError):
            summary.validate_sequences(pd.DataFrame([dict(base_sequence="s")]), "full")

    def test_combines_runs_aliases_simple_and_writes_raw_event_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runs = {name: root / name for name in ("full_run", "controls", "q0")}
            for name, variants in (("full_run", ["full", "simple"]),
                                   ("controls", ["no_recovery_admission"]), ("q0", ["no_rollout"])):
                run = runs[name]
                run.mkdir()
                driver.evaluation.dump(run / "COMPLETE.json", {"variants": variants})
                seqs, episodes, events, cal = [], [], [], []
                for variant in variants:
                    for base, obj in driver.evaluation.TEST.items():
                        seqs.append(dict(variant=variant, base_sequence=base, object_id=obj,
                            auc_percent=50., b1_auc_percent=45., failure_1cm_percent=30., failure_2cm_percent=20.))
                        for condition in driver.evaluation.CONDITIONS:
                            episodes.append(dict(variant=variant, episode=base+condition))
                        events.append(dict(variant=variant, base_sequence=base, episode=base+"_black10",
                            recovery_index=20, raw_recovery_error_cm=.1, generated_count=1))
                        cal.append(dict(variant=variant, base_sequence=base, metric="fixture"))
                for filename, rows in (("sequence_metrics",seqs), ("episode_metrics",episodes),
                                       ("recovery_events",events), ("relocalization_events",events), ("calibration_metrics",cal)):
                    pd.DataFrame(rows).to_csv(run / (filename+".csv"), index=False)
            out=root / "summary"
            out.mkdir()
            summary.summarize(out, runs["full_run"], runs["controls"], runs["q0"], True, 42)
            result=pd.read_csv(out / "ablation_sequence_metrics.csv")
            self.assertEqual(len(result),20)
            self.assertEqual(set(result.variant), {"full","no_quality","no_recovery_admission","no_rollout"})
            self.assertEqual(set(result[result.variant=="no_quality"].source_variant), {"simple"})
            self.assertEqual(len(pd.read_csv(out / "ablation_raw_recovery_pair_audit.csv")),15)


class FrozenAdapterIntegration(unittest.TestCase):
    def fixture(self):
        import test_frozen_test_runner as fixtures
        item=fixtures.TestTestRelease()
        item.setUp()
        self.addCleanup(item.doCleanups)
        return item

    def test_refuse_q1_as_no_rollout_before_inference(self):
        item=self.fixture()
        with mock.patch.object(driver.evaluation,"run_logged") as inference:
            with self.assertRaisesRegex(ValueError,"separately trained"):
                driver.evaluation.main(["--release",str(item.release),"--variants","no_rollout",
                                        "--output",str(item.root / "not_started"),"--check-only"])
        inference.assert_not_called()
        self.assertFalse((item.root / "not_started").exists())

    def test_no_admission_helper_is_snapshotted_and_hashed(self):
        item=self.fixture()
        output=item.root / "gate_ablation"
        with mock.patch.object(driver.evaluation.subprocess,"run",return_value=mock.Mock(stdout="fixture",returncode=0)), \
             mock.patch.object(driver.evaluation,"scan_test_images",return_value={"passed":True,"errors":[]}), \
             mock.patch.object(driver.evaluation,"run_logged"):
            driver.evaluation.main(["--release",str(item.release),"--variants","no_recovery_admission",
                                    "--output",str(output),"--check-only"])
        helper=output / "source/ablation_policy.py"
        self.assertEqual(driver.evaluation.sha(helper),driver.evaluation.sha(ROOT / "ablation_policy.py"))
        self.assertIn(str(helper.resolve()),driver.evaluation.read_json(output / "test_source_hashes.json"))
        self.assertEqual(driver.evaluation.read_json(output / "test_protocol.json")["evaluation_status"],
                         "post_test_diagnostic_not_untouched")

    def test_original_three_sequence_release_keeps_its_own_population(self):
        item=self.fixture()
        bases=["mustard0","bleach0","bleach_hard_00_03_chaitanya"]
        item.cfg["train_bases"]=bases
        item.effective["train_bases"]=bases
        (item.release / "artifacts/shared_quality_config.json").write_text(json.dumps(item.cfg),encoding="utf-8")
        (item.release / "effective_config.json").write_text(json.dumps(item.effective),encoding="utf-8")
        item.checksums()
        cfg,effective=driver.evaluation.verify_release(item.release)
        cmd,_=driver.evaluation.command_for(item.release,item.root / "unused",cfg,effective,"sugar_box1","full")
        start=cmd.index("--train_seqs")+1
        self.assertEqual(cmd[start:cmd.index("--test_base_seq")],bases)
        self.assertNotIn("mustard_easy_00_02",cmd)


class TrainingIntegration(unittest.TestCase):
    def test_prepare_train_command_and_seal_q0_preserve_parent(self):
        # Real manifest/prepare/seal + dummy fitting outputs. No model/GPU execution.
        import test_run_train_python as fixtures
        fixture = fixtures.TestPythonTrainingRunner()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        args = run_train.arguments(fixture.argv())
        parent = fixture.root / "output"
        env = run_train.environment(args, parent)
        env.update(TRAIN_PYTHON=sys.executable, FOUNDATIONPOSE_PYTHON=sys.executable,
                   SAM2_PYTHON=sys.executable)
        for key in ("CAD_MODEL_ROOT", "FOUNDATIONPOSE_DIR", "SAM2_DIR"):
            env[key] = str(fixture.root / key.lower())
        for key in ("FOUNDATIONPOSE_REFINER_WEIGHT", "FOUNDATIONPOSE_SCORER_WEIGHT", "SAM2_CHECKPOINT"):
            p = fixture.root / key
            p.write_bytes(b"fixture")
            env[key] = str(p)
        for base in run_train.TRAIN_BASES:
            (fixture.gt / base / "init_mask.png").write_bytes(b"fixture")
            cad = Path(env["CAD_MODEL_ROOT"]) / train_release.CAD[base]
            cad.mkdir(parents=True, exist_ok=True)
            for name in ("points.xyz", "textured.obj"):
                (cad / name).write_bytes(b"fixture")
        from online_observer import build_config, asset_paths
        env.update(SE3_PYTHON=sys.executable, SE3_WEIGHT_ROOT=str(fixture.root/"se3weights"), SE3_DATA_ROOT=str(fixture.root/"se3data"))
        for asset in asset_paths(build_config(env, run_train.SEQUENCE_OBJECTS)):
            if asset.resolve() != Path(sys.executable).resolve():
                asset.parent.mkdir(parents=True,exist_ok=True)
                asset.write_bytes(b"fixture")
        sam_cfg=Path(env["SAM2_DIR"]) / "sam2" / env["SAM2_CONFIG"]
        sam_cfg.parent.mkdir(parents=True)
        sam_cfg.write_text("fixture")
        parent.mkdir()
        driver.evaluation.dump(parent / "manifest_config.json", dict(base_sequences=list(run_train.SEQUENCE_OBJECTS),
                               common_conditions=list(run_train.CONDITIONS), extra_conditions={}))
        driver.evaluation.dump(parent / "sequence_inventory.json", {})
        builder=driver.load_module(ROOT / "1-build_dataset_manifest_all.py", "ablation_fixture_manifest")
        builder.build_all_manifest(str(fixture.data), str(fixture.gt), str(fixture.results),
            str(parent / "manifest_config.json")).to_csv(parent / "reference_manifest.csv", index=False)
        with mock.patch.dict(os.environ, env), mock.patch.object(train_release.subprocess,"run",return_value=mock.Mock(stdout="fixture\n")):
            train_release.prepare(True)

        def artifacts(release, rounds):
            from b5_revision import CONFIG
            from recovery_gate import CONFIG as GATE
            cfg=dict(version="shared_pose_quality_v1", training_mode="final_development_fit",
                perception_runtime_config=__import__('perception_runtime').CONFIG,
                execution_settings=__import__('runtime_settings').execution_config(),
                held_out_base=None, train_bases=list(run_train.TRAIN_BASES), fit_conditions=list(run_train.CONDITIONS),
                seed=42, train_fraction=.7, on_policy_refine_rounds=rounds, risk_threshold_cm=1.,
                prior_advantage_margin_cm=.1, p_risk_threshold=.74 if rounds else .6,
                manifest_sha256=driver.evaluation.sha(release / "reference_manifest.csv"),
                observer_config_sha256=driver.evaluation.sha(release/"observer_config.json"),
                b5_policy_config=dict(CONFIG), recovery_gate_config=dict(GATE), feature_columns=["fixture"],
                target="normalized_ADD-S_error_E_over_D_obj")
            (release / "artifacts").mkdir(exist_ok=True)
            for key,name in (("model","shared_pose_quality_model.joblib"),("scaler","shared_pose_quality_scaler.joblib"),
                             ("calibrator","shared_risk_calibrator.joblib")):
                path=release / "artifacts" / name
                path.write_bytes(b"fixture, not a trained model")
                cfg[key+"_sha256"]=driver.evaluation.sha(path)
            driver.evaluation.dump(release / "artifacts/shared_quality_config.json",cfg)
        artifacts(parent,1)
        from cache_fixture import make_index
        make_index(parent)
        with mock.patch.dict(os.environ,env), mock.patch.object(Path,"chmod"):
            train_release.seal()
        before=driver.evaluation.sha(parent / "freeze.sha256")
        real_loader=driver.load_module
        commands=[]
        def loader(path,name):
            module=real_loader(path,name)
            if name=="ablation_frozen_training_runner":
                def fake_fit(command, environment, cwd, log):
                    if 'prepare_sam_cache.py' in str(command[3]):
                        make_index(Path(cwd))
                        log.write_text('Synthetic cache, no inference')
                        return
                    commands.append(command)
                    self.assertEqual(command[command.index("--on_policy_refine_rounds")+1],"0")
                    artifacts(Path(cwd),0)
                    log.write_text("CPU fixture only; no actual fit")
                module.run=fake_fit
            return module
        def process(command, **kwargs):
            if command[-1] in ("prepare-generated","seal"):
                helper=real_loader(Path(command[2]),"ablation_fixture_release")
                with mock.patch.dict(os.environ,kwargs["env"]), mock.patch.object(Path,"chmod"):
                    helper.prepare(True) if command[-1]=="prepare-generated" else helper.seal()
            return mock.Mock(stdout="fixture\n",returncode=0)
        output=fixture.root / "q0_training"
        with mock.patch.object(driver,"load_module",side_effect=loader), \
             mock.patch.object(driver.subprocess,"run",side_effect=process):
            driver.train_no_rollout(argparse.Namespace(release=str(parent),output=str(output),check_only=False))
        self.assertEqual(len(commands),1)
        q0=output / "no_rollout_release"
        self.assertTrue((q0 / "FROZEN.json").is_file())
        self.assertEqual(driver.evaluation.sha(parent / "freeze.sha256"),before)
        self.assertEqual(driver.evaluation.sha(parent / "source/b5_policy.py"), driver.evaluation.sha(q0 / "source/b5_policy.py"))
        self.assertEqual(driver.evaluation.sha(parent / "reference_manifest.csv"),driver.evaluation.sha(q0 / "reference_manifest.csv"))


if __name__ == "__main__":
    unittest.main()
