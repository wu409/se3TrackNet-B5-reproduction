import unittest
import os
from pathlib import Path
import tempfile
from unittest.mock import Mock
import numpy as np
import pandas as pd
from fixed_policy_control import FEATURES, fixed_prior, validate_samples, frozen_release, load_frozen_module, fit_and_report


class FixedPolicyTests(unittest.TestCase):
    def setUp(self):
        self.manifest = pd.DataFrame(dict(sequence=["dev_clean"] * 3, sequence_index=[0, 1, 2], frame_id=[10, 20, 30]))
        chunks = []
        for source in ("obs", "prior"):
            chunk = self.manifest.assign(hypothesis=source, D_obj_cm=10., target_E_cm=2., target_e_norm=.2)
            for feature in FEATURES:
                chunk[feature] = .5
            chunks.append(chunk)
        self.samples = pd.concat(chunks, ignore_index=True)

    def test_first_two_frames_do_not_extrapolate(self):
        observation = np.eye(4)
        extrapolate = Mock(side_effect=AssertionError("unexpected extrapolation"))
        for history in ([], [np.eye(4)]):
            result = fixed_prior(history, observation, extrapolate)
            np.testing.assert_equal(result, observation)
            self.assertIsNot(result, observation)

    def test_prior_uses_only_last_two_observations(self):
        a, b, now = np.eye(4), np.eye(4), np.eye(4)
        a[0, 3], b[0, 3], now[0, 3] = 1., 3., 100.
        extrapolate = Mock(return_value=b.copy())
        fixed_prior([np.zeros((4, 4)), a, b], now, extrapolate)
        self.assertIs(extrapolate.call_args.args[0], b)
        self.assertIs(extrapolate.call_args.args[1], a)

    def test_matched_candidates_are_accepted(self):
        validate_samples(self.samples.sample(frac=1, random_state=42), self.manifest)

    def test_missing_prior_frame_rejected(self):
        with self.assertRaises(ValueError):
            validate_samples(self.samples.iloc[:-1], self.manifest)

    def test_duplicate_and_test_candidate_rejected(self):
        for bad in (pd.concat([self.samples, self.samples.iloc[[0]]]), self.samples.assign(sequence="test_clean")):
            with self.assertRaises(ValueError):
                validate_samples(bad, self.manifest)

    def test_nonfinite_and_wrong_units_rejected(self):
        for column, value in ((FEATURES[0], np.nan), ("target_e_norm", 20.), ("D_obj_cm", 0.)):
            bad = self.samples.copy()
            bad.loc[0, column] = value
            with self.assertRaises(ValueError):
                validate_samples(bad, self.manifest)


@unittest.skipUnless(os.environ.get("REFIT_TEST_Q0") and os.environ.get("REFIT_TEST_Q1"), "Frozen-server integration test")
class FrozenFitIntegrationTest(unittest.TestCase):
    def test_all_control_models_and_reports_on_synthetic_fixture(self):
        q0 = frozen_release(os.environ["REFIT_TEST_Q0"], 0)
        full = frozen_release(os.environ["REFIT_TEST_Q1"], 1)
        frozen = load_frozen_module(full)
        raw = []
        for sequence in ("synthetic_a", "synthetic_b"):
            for i in range(30):
                row = dict(sequence=sequence, sequence_index=i, frame_id=10 + i * 3, D_obj_cm=10.)
                for kind in ("obs", "prior"):
                    error = .2 if i % 2 == 0 else 1.8
                    row["E_%s_cm" % kind] = error
                    row["e_%s_norm" % kind] = error / 10.
                    for feature in ("x1_%s_norm", "x2_%s_inlier_error", "x4_%s_support_ratio", "x5_%s_geometry_inconsistency"):
                        row[feature % kind] = .01 * i + (i % 2) * .5
                raw.append(row)
        labels = pd.DataFrame(raw)
        manifest = labels[["sequence", "sequence_index", "frame_id"]]
        fixed = frozen._hypothesis_samples_from_rollout(labels)
        with tempfile.TemporaryDirectory(prefix="refit_fit_test_") as folder:
            path = Path(folder)
            labels.to_csv(path / "synthetic_labels.csv", index=False)
            q0["labels"] = full["labels"] = path / "synthetic_labels.csv"
            fit_and_report(frozen, q0, full, fixed, manifest, path)
            counts = pd.read_csv(path / "training_counts.csv").set_index("model")
            self.assertEqual(counts.loc["fixed_obs_only", "candidates"], 60)
            for name in ("fixed_obs_duplicated_count_control", "fixed_obs_prior", "q0_policy_obs_prior_refit"):
                self.assertEqual(counts.loc[name, "candidates"], 120)
                self.assertTrue((path / "models" / name / "model.joblib").is_file())
            report = pd.read_csv(path / "metrics.csv")
            self.assertEqual(set(report.model), {"frozen_q0", "frozen_q1", *counts.index})
            self.assertEqual(set(report.candidate_population), {"fixed_policy", "q0_policy", "q1_policy"})
            self.assertEqual(len(report), 54)
            self.assertTrue(np.isfinite(report.mae_cm).all())


if __name__ == "__main__":
    unittest.main()
