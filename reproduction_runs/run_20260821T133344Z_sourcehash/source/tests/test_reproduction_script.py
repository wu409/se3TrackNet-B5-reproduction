import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestReproductionScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (ROOT / "run.sh").read_text(encoding="utf-8")

    def test_requires_frozen_manifest_and_object_weight_root(self):
        for token in ("REFERENCE_MANIFEST", "SE3TRACKNET_WEIGHTS_ROOT",
                      "mustard_bottle/model_best_val.pth.tar",
                      "bleach_cleanser/model_best_val.pth.tar",
                      "mustard_bottle/mean.npy", "bleach_cleanser/std.npy"):
            self.assertIn(token, self.script)

    def test_git_is_strict_when_available_and_source_hashes_are_fallback(self):
        for token in ("GIT_AVAILABLE=true", "GIT_AVAILABLE=false", "git status --porcelain",
                      "git_commit=", "source_file_sha256.txt", "source_bundle_sha256.txt"):
            self.assertIn(token, self.script)

    def test_verification_precedes_consumers_and_reference_is_not_rebuilt(self):
        verify = self.script.index("--mode verify-runtime")
        risk = self.script.index("python 2-risk_label.py")
        evaluate = self.script.index("python 3-train_evaluation.py")
        self.assertLess(verify, risk)
        self.assertLess(verify, evaluate)
        self.assertNotIn("--mode build-reference", self.script)

    def test_script_directory_works_when_invoked_as_bash_run_sh(self):
        self.assertIn('dirname -- "${BASH_SOURCE[0]}"', self.script)
        self.assertNotIn('${BASH_SOURCE[0]%/*}', self.script)

    def test_required_evidence_is_recorded(self):
        for token in ("runtime_inventory.csv", "input_verification.json",
                      "dataset_artifact_sha256.csv", "prediction_artifact_sha256.csv",
                      "checkpoint_identifier.txt", "se3tracknet_weight_artifact_sha256.txt",
                      "prediction_sequence_checkpoint_mapping.csv", "cad_artifact_sha256.txt",
                      "checkpoint2_blackout_frame_intervals", "environment.yml",
                      "pip_freeze.txt", "full_run.log", "sha256sum -c sha256.txt"):
            self.assertIn(token, self.script)

    def test_full_log_is_closed_before_portable_hash_manifest(self):
        close = self.script.index("exec 1>&3 2>&4")
        portable_hash = self.script.index("find . -type f ! -name \"sha256.txt\"")
        self.assertLess(close, portable_hash)


if __name__ == "__main__":
    unittest.main()
