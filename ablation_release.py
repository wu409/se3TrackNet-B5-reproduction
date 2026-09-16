"""Matched ablation orchestration; Python 3.8; never edits a parent release.

train-no-rollout replays the parent's frozen training implementation with only
the refinement count changed to zero. evaluate uses separately frozen releases.
All modifications after previous test inspection remain post-test diagnostics.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import test_release as evaluation

ROOT = Path(__file__).resolve().parent


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("Unsupported frozen source; expected exactly one: " + before)
    return text.replace(before, after, 1)


def patch_zero_refit(label_source, release_source):
    # No changes to features, fitting, train/cal split, rollout or B5 decisions.
    label_source = replace_once(label_source,
        "if args.on_policy_refine_rounds < 1:", "if args.on_policy_refine_rounds < 0:")
    label_source = replace_once(label_source,
        '"on_policy_refine_rounds must be >= 1"', '"on_policy_refine_rounds must be >= 0"')
    release_source = replace_once(release_source,
        '"on_policy_refine_rounds": 1, "seed": 42', '"on_policy_refine_rounds": 0, "seed": 42')
    release_source = replace_once(release_source,
        '("on_policy_refine_rounds", 1)', '("on_policy_refine_rounds", 0)')
    return label_source, release_source


def assert_full(cfg, effective):
    expected = dict(on_policy_refine_rounds=1, seed=42, train_fraction=0.7,
                    risk_threshold_cm=1.0, prior_advantage_margin_cm=0.1)
    for key, value in expected.items():
        if cfg.get(key) != value or effective.get(key) != value:
            raise ValueError("Unsupported full training setting: " + key)
    if not cfg.get("b5_policy_config") or not cfg.get("recovery_gate_config"):
        raise ValueError("Use the NEW full release containing revised fusion/history and gate")


def assert_matched(full, q0):
    fc, fe = full
    qc, qe = q0
    assert_full(fc, fe)
    if qc.get("on_policy_refine_rounds") != 0 or qe.get("on_policy_refine_rounds") != 0:
        raise ValueError("No-rollout release must have zero refits")
    for key in ("seed", "train_fraction", "risk_threshold_cm", "prior_advantage_margin_cm",
                "train_bases", "fit_conditions", "feature_columns", "target", "manifest_sha256",
                "b5_policy_config", "recovery_gate_config"):
        if fc.get(key) != qc.get(key):
            raise ValueError("Full/q0 settings differ: " + key)
    for key in ("paths", "training_selection", "blackout_min_frames", "foundationpose_refine_iter"):
        if fe.get(key) != qe.get(key):
            raise ValueError("Full/q0 environment or split differs: " + key)
    # Probability thresholds/calibrators may differ: each is fitted on development only.


def new_output(value, kind, parent):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(value).expanduser().resolve() if value else ROOT / "ablation_runs" / (
        kind + "_" + stamp + "_" + uuid.uuid4().hex[:8])
    if output.exists() or output == parent or parent in output.parents:
        raise ValueError("Output must be NEW and outside the parent release")
    return output


def frozen_env(effective):
    paths = effective["paths"]
    if Path(sys.executable).resolve() != Path(paths["TRAIN_PYTHON"]).resolve():
        raise ValueError("Use the parent's TRAIN_PYTHON: " + paths["TRAIN_PYTHON"])
    env = dict(os.environ)
    env.update({k: str(v) for k, v in paths.items() if v is not None})
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONHASHSEED=str(effective["seed"]))
    return env


def validate_training_command(command, cfg, effective):
    start = command.index("--target_seqs") + 1
    stop = command.index("--cad_models_seq")
    if command[start:stop] != cfg["train_bases"]:
        raise ValueError("Frozen training runner's population/order differs from the full model")
    start = command.index("--corruption_lists") + 1
    stop = next(i for i in range(start, len(command)) if command[i].startswith("--"))
    if command[start:stop] != cfg["fit_conditions"]:
        raise ValueError("Frozen training runner's conditions/order differs from the full model")
    for option, value in (("--seed", cfg["seed"]), ("--train_fraction", cfg["train_fraction"]),
        ("--risk_threshold", cfg["risk_threshold_cm"]),
        ("--prior_advantage_margin_cm", cfg["prior_advantage_margin_cm"]),
        ("--blackout_min_frames", effective["blackout_min_frames"]),
        ("--foundationpose_refine_iter", effective["foundationpose_refine_iter"])):
        if float(command[command.index(option)+1]) != float(value):
            raise ValueError("Frozen training runner's setting differs from full: " + option)


def train_no_rollout(args):
    parent = Path(args.release).expanduser().resolve()
    cfg, effective = evaluation.verify_release(parent)
    assert_full(cfg, effective)
    env = frozen_env(effective)
    output = new_output(args.output, "q0_training", parent)
    output.mkdir(parents=True)
    stage = output / "training_source"
    shutil.copytree(parent / "source", stage)
    labels, seal = patch_zero_refit((stage / "2-risk_label.py").read_text(encoding="utf-8-sig"),
                                  (stage / "train_release.py").read_text(encoding="utf-8-sig"))
    (stage / "2-risk_label.py").write_text(labels, encoding="utf-8")
    (stage / "train_release.py").write_text(seal, encoding="utf-8")
    release = output / "no_rollout_release"
    release.mkdir()
    for name in ("reference_manifest.csv", "manifest_config.json", "sequence_inventory.json"):
        shutil.copy2(parent / name, release / name)
    # Use the parent's exact population, never the current workspace TRAIN_BASES.
    runner = load_module(stage / "run_train.py", "ablation_frozen_training_runner")
    env.update(TRAIN_REPO=str(stage), RELEASE_DIR=str(release),
               REFERENCE_MANIFEST=str(release / "reference_manifest.csv"))
    command = runner.training_command(env, release)
    validate_training_command(command, cfg, effective)
    index = command.index("--on_policy_refine_rounds") + 1
    if command[index] != "1":
        raise ValueError("Unexpected parent training command")
    command[index] = "0"
    evaluation.dump(output / "ablation_plan.json", dict(
        created_utc=datetime.now(timezone.utc).isoformat(), parent_release=str(parent),
        parent_freeze_sha256=evaluation.sha(parent / "freeze.sha256"),
        train_bases=cfg["train_bases"], conditions=cfg["fit_conditions"],
        ablation="no_rollout_refitting", model="q0_observation_only_bootstrap",
        changed="refit count 1 -> 0; permit zero in CLI and seal validation only",
        final_diagnostic_rollout="still runs; no fit after it",
        calibration="same development split/procedure; own q0 calibrator and threshold",
        evaluation_status="post_test_diagnostic_not_untouched", command=command))
    subprocess.run([env["TRAIN_PYTHON"], "-B", str(stage / "train_release.py"),
                    "prepare-generated"], env=env, cwd=stage, check=True)
    for name in ("training_pip_freeze.txt", "sam2_pip_freeze.txt", "foundationpose_pip_freeze.txt"):
        if (parent / name).read_text().splitlines() != (release / name).read_text().splitlines():
            raise ValueError("Environment differs from full training: " + name)
    # Bind provenance into the child freeze, without touching the original release.
    shutil.copy2(output / "ablation_plan.json", release / "ablation_parent.json")
    if args.check_only:
        evaluation.dump(output / "CHECKED.json", {"status": "prepared_no_training_no_freeze"})
        print("Checks passed; use a NEW output directory for actual q0 training:", output)
        return output
    subprocess.run([env["TRAIN_PYTHON"], "-B", "-c",
                    "import torch; assert torch.cuda.is_available(), 'GPU unavailable'"], env=env, check=True)
    (release / "artifacts").mkdir()
    runner.run(command, env, release, release / "training.log")
    subprocess.run([env["TRAIN_PYTHON"], "-B", str(release / "source/train_release.py"), "seal"],
                   cwd=release, env=env, check=True)
    evaluation.verify_release(parent)
    assert_matched((cfg, effective), evaluation.verify_release(release))
    print("NO_ROLLOUT_RELEASE=" + str(release))
    return output


def verify_result(run, parent, required):
    run = Path(run).expanduser().resolve()
    done = evaluation.read_json(run / "COMPLETE.json")
    protocol = evaluation.read_json(run / "test_protocol.json")
    if Path(protocol["training_release"]).resolve() != parent or not set(required) <= set(done["variants"]):
        raise ValueError("Existing full results do not belong to this release/variants")
    if protocol.get("b5_policy_revision", "frozen") != "frozen" or protocol.get("recovery_gate_revision", "frozen") != "frozen":
        raise ValueError("Existing results override the parent policy; not a matched control")
    if protocol["test_sequences"] != evaluation.TEST or tuple(protocol["conditions"]) != evaluation.CONDITIONS:
        raise ValueError("Existing result population mismatch")
    for name in ("test_source_hashes.json", "test_input_hashes.json"):
        for path, digest in evaluation.read_json(run / name).items():
            if evaluation.sha(path) != digest:
                raise ValueError("Existing run source/input changed: " + path)
    return run


def evaluate(args):
    parent = Path(args.release).expanduser().resolve()
    full = evaluation.verify_release(parent)
    assert_full(*full)
    q0 = Path(args.no_rollout_release).expanduser().resolve()
    assert_matched(full, evaluation.verify_release(q0))
    provenance = evaluation.read_json(q0 / "ablation_parent.json")
    if (Path(provenance["parent_release"]).resolve() != parent
            or provenance["parent_freeze_sha256"] != evaluation.sha(parent / "freeze.sha256")):
        raise ValueError("q0 was not derived from this exact full release")
    for name in ("b5_policy.py", "b5_revision.py", "recovery_gate.py"):
        if evaluation.sha(parent / "source" / name) != evaluation.sha(q0 / "source" / name):
            raise ValueError("Full/q0 policy source mismatch: " + name)
    existing = verify_result(args.full_results, parent, ["full"]) if args.full_results else None
    variants = ["no_recovery_admission"]
    if args.with_decision_subablations:
        variants += ["no_absolute_gate", "no_relative_advantage"]
    reuse_simple = bool(existing and "simple" in evaluation.read_json(existing / "COMPLETE.json")["variants"])
    if not reuse_simple:
        variants.append("no_quality")
    if existing is None:
        variants.insert(0, "full")
    output = new_output(args.output, "evaluation", parent)
    for protected in (q0, existing):
        if protected and (output == protected or protected in output.parents):
            raise ValueError("Output must be outside all input releases/results")
    output.mkdir(parents=True)
    evaluation.dump(output / "ablation_plan.json", dict(
        created_utc=datetime.now(timezone.utc).isoformat(), parent_release=str(parent),
        parent_freeze_sha256=evaluation.sha(parent / "freeze.sha256"),
        no_rollout_release=str(q0), no_rollout_freeze_sha256=evaluation.sha(q0 / "freeze.sha256"),
        existing_full_results=str(existing) if existing else None, reuse_simple_as_no_quality=reuse_simple,
        variants=variants, evaluation_status="post_test_diagnostic_not_untouched"))
    extra = ["--check-only"] if args.check_only else []
    controls = evaluation.main(["--release", str(parent), "--output", str(output / "controls"),
                                "--variants", *variants, *extra])
    q0_run = evaluation.main(["--release", str(q0), "--output", str(output / "no_rollout"),
                              "--variants", "no_rollout", *extra])
    if args.check_only:
        print("Checks passed, no inference. Actual run requires a NEW output:", output)
        return output
    from ablation_summary import summarize
    summarize(output, existing or controls, controls, q0_run, reuse_simple, full[0]["seed"])
    evaluation.dump(output / "COMPLETE.json", {"status": "post_test_ablation_complete"})
    print("ABLATION_RESULTS=" + str(output))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train-no-rollout", "evaluate"])
    parser.add_argument("--release", required=True, help="Completed NEW full training release")
    parser.add_argument("--output", help="NEW output directory; never reuse a failed/check-only run")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-rollout-release")
    parser.add_argument("--full-results", help="Optional completed full/simple run from this exact release")
    parser.add_argument("--with-decision-subablations", action="store_true",
                        help="Also run no_absolute_gate/no_relative_advantage; optional, not default")
    args = parser.parse_args(argv)
    if args.action == "evaluate" and not args.no_rollout_release:
        parser.error("evaluate requires --no-rollout-release (train it first)")
    return train_no_rollout(args) if args.action == "train-no-rollout" else evaluate(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print("ABLATION FAILED:", exc, file=sys.stderr)
        print("No complete result claimed; preserve logs and use a new output directory.", file=sys.stderr)
        raise SystemExit(1)
