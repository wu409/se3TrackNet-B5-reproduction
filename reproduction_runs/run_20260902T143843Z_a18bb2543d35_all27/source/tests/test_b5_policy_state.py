import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

sys.modules.setdefault(
    "open3d",
    types.ModuleType("open3d")
)

if "scipy.spatial.transform" not in sys.modules:
    scipy = types.ModuleType("scipy")
    spatial = types.ModuleType("scipy.spatial")
    transform = types.ModuleType("scipy.spatial.transform")

    transform.Rotation = object

    scipy.spatial = spatial
    spatial.transform = transform

    sys.modules["scipy"] = scipy
    sys.modules["scipy.spatial"] = spatial
    sys.modules["scipy.spatial.transform"] = transform


SPEC = importlib.util.spec_from_file_location(
    "b5_policy_test",
    ROOT / "b5_policy.py"
)

b5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b5)


class TestB5PolicyState(unittest.TestCase):

    def transition(
        self,
        state,
        is_blackout,
        index,
        frame_id,
    ):
        identity = np.eye(4)

        # 当前 B5 blackout 只由 full-frame depth 决定
        if is_blackout:
            # 0 depth -> 0% valid -> blackout
            depth_real = np.zeros(
                (10, 10),
                dtype=np.float32,
            )
        else:
            # 1 m -> 100% valid -> non-blackout
            depth_real = np.ones(
                (10, 10),
                dtype=np.float32,
            )

        return b5.b5_transition(
            T_obs=identity,
            T_prior=identity,

            p_obs_bad=0.0,
            p_prior_bad=0.0,

            # support 当前已经不参与 blackout 判断
            support=0.0,

            depth_real=depth_real,

            model_pts=np.zeros(
                (1, 3),
                dtype=np.float64,
            ),

            K=np.eye(3),

            # 新接口：obs / prior 阈值分离
            p_obs_threshold=0.8,
            p_prior_threshold=0.8,

            frame_index=index,
            frame_id=frame_id,
            state=state,

            blackout_min_frames=2,
            use_prior_predictor=True,
        )


    def test_exact_blackout_and_recovery_interval(self):

        state = b5.init_b5_state()

        # actual_recovery_action 当前返回三个值：
        # T_recovery, recovery_ok, diagnostics
        with mock.patch.object(
            b5,
            "actual_recovery_action",
            return_value=(
                np.eye(4),
                True,
                {},
            ),
        ):
            _, mode0, state, _ = self.transition(
                state,
                True,
                0,
                100,
            )

            _, mode1, state, _ = self.transition(
                state,
                True,
                1,
                101,
            )

            _, mode2, state, recovery = self.transition(
                state,
                False,
                2,
                102,
            )

        self.assertEqual(
            "MODE_3_BLACKOUT_WAITING",
            mode0,
        )

        self.assertEqual(
            "MODE_3_BLACKOUT_WAITING",
            mode1,
        )

        self.assertEqual(
            "MODE_3_RECOVERY_EXECUTE",
            mode2,
        )

        expected = {
            "blackout_start_index": 0,
            "blackout_end_index": 1,
            "recovery_index": 2,

            "blackout_start_frame": 100,
            "blackout_end_frame": 101,
            "recovery_frame": 102,
        }

        self.assertEqual(
            expected,
            state["blackout_intervals"][0],
        )

        self.assertEqual(
            state["blackout_intervals"][0],
            recovery["blackout_interval"],
        )


    def test_short_gap_does_not_create_blackout_interval(self):

        state = b5.init_b5_state()

        _, _, state, _ = self.transition(
            state,
            True,
            0,
            10,
        )

        _, mode, state, recovery = self.transition(
            state,
            False,
            1,
            11,
        )

        self.assertEqual(
            "MODE_1_ACCEPT",
            mode,
        )

        self.assertIsNone(
            recovery
        )

        self.assertEqual(
            [],
            state["blackout_intervals"],
        )


if __name__ == "__main__":
    unittest.main()
