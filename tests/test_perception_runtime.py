import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import perception_runtime as runtime
from perception_workers import LazyFrames, SamWorker, FoundationWorker


class FakeWorker:
    def __init__(self, python, kind, cfg, scratch, log_path):
        self.kind, self.cfg, self.scratch = kind, cfg, Path(scratch)
        self.log_path = Path(log_path)
        self.calls, self.closed = [], False

    def call(self, **request):
        self.calls.append(request)
        if self.kind == "sam2":
            np.save(self.scratch / "current_mask.npy", np.ones((8, 8), dtype=bool))
            return dict(frame_index=request["frame_index"], peak_allocated_mb=1)
        return dict(frame_index=request["frame_index"], pose=np.eye(4).tolist(),
                    diagnostics={"raw_valid": True})

    def close(self):
        self.closed = True


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = types.SimpleNamespace(sam2_live_diagnostic=True, sam2_python=sys.executable, sam2_dir=str(self.root),
            sam2_checkpoint=str(self.root/'checkpoint'), sam2_config='cfg',
            foundationpose_python=sys.executable, foundationpose_dir=str(self.root),
            foundationpose_refiner_weight=str(self.root/'weight'))
        self.paths = [str(self.root / (str(i)+'.png')) for i in range(3)]
        self.patch = mock.patch.object(runtime, 'JsonWorker', FakeWorker)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.session = runtime.PerceptionSession(self.args, self.paths, self.root/'mask.png',
                                                self.root/'mesh.obj', self.root/'episode')
        self.addCleanup(self.session.close)

    def test_advances_once_and_recovery_uses_exact_current_frame(self):
        for i, path in enumerate(self.paths):
            self.session.advance(i, path)
            mask, ok, diag = self.session.current_mask(self.paths[:i+1], self.root/'mask.png')
            self.assertTrue(ok)
            self.assertEqual(diag['sam2_target_index'], i)
            self.assertEqual(mask.shape, (8, 8))
        self.assertEqual(len(self.session.sam.calls), 3)
        self.session.current_mask(self.paths, self.root/'mask.png')
        self.assertEqual(len(self.session.sam.calls), 3)  # no new inference on get
        with self.assertRaises(ValueError):
            self.session.advance(2, self.paths[2])
        with self.assertRaises(RuntimeError):
            self.session.current_mask(self.paths[:1], self.root/'mask.png')

    def test_wrong_path_and_prompt_are_rejected(self):
        with self.assertRaises(ValueError):
            self.session.advance(0, self.paths[1])
        self.session.advance(0, self.paths[0])
        with self.assertRaises(RuntimeError):
            self.session.current_mask(self.paths[:1], self.root/'other.png')

    def test_repeated_registration_reuses_worker(self):
        worker = self.session.fp
        for i in range(2):
            self.session.advance(i, self.paths[i])
            pose, valid, _ = self.session.register(np.zeros((8,8,3)), np.ones((8,8)),
                np.ones((8,8)), np.eye(3), self.root/'mesh.obj', 5)
            self.assertTrue(valid)
            np.testing.assert_array_equal(pose, np.eye(4))
        self.assertIs(self.session.fp, worker)
        self.assertEqual(len(worker.calls), 2)
        self.assertTrue(all(c['operation'] == 'register' for c in worker.calls))

    def test_close_resets_session_and_records_cost(self):
        self.session.advance(0, self.paths[0])
        self.session.close(complete=False)
        self.assertIsNone(runtime.active_session())
        self.assertTrue(self.session.sam.closed and self.session.fp.closed)
        receipt = json.loads(self.session.summary_path.read_text())
        self.assertEqual(receipt['frames_advanced'], 1)
        self.assertGreater(receipt['sam2_wall_ms'], 0)
        self.assertFalse(receipt['complete'])
        self.assertFalse(self.session.scratch.exists())

    def test_cleanup_refuses_non_uuid_or_non_tmp_path(self):
        (self.session.scratch/'owned_mesh.json').write_text(json.dumps({'path': str(self.root/'mesh.obj')}))
        with self.assertRaises(RuntimeError):
            runtime.cleanup_owned_mesh(self.session.scratch)
        (self.session.scratch/'owned_mesh.json').unlink()


class WorkerProtocolTests(unittest.TestCase):
    def run_worker(self, code, request, timeout=2):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        worker = runtime.JsonWorker(sys.executable, 'sam2', {}, temp.name, Path(temp.name)/'worker.log', timeout)
        self.addCleanup(worker.close)
        real_popen = subprocess.Popen
        def start(command, **kwargs):
            return real_popen([sys.executable, '-u', '-c', code], **kwargs)
        with mock.patch.object(runtime.subprocess, 'Popen', side_effect=start):
            return worker.call(**request)

    def test_protocol_roundtrip(self):
        result = self.run_worker("import sys,json\nfor line in sys.stdin:\n print(line.strip(),flush=True)", {'frame_index': 0})
        self.assertEqual(result['request_id'], 1)

    def test_wrong_id_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'ID mismatch'):
            self.run_worker("import sys\nsys.stdin.readline()\nprint('{\"request_id\":99}',flush=True)", {})

    def test_native_stdout_does_not_corrupt_private_json_channel(self):
        code = ("import sys,os,json\n"
                "channel=os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1,encoding='utf-8')\n"
                "os.dup2(sys.stderr.fileno(),sys.stdout.fileno())\n"
                "request=json.loads(sys.stdin.readline())\n"
                "os.write(1,b'native library log\\n')\n"
                "channel.write(json.dumps(request)+'\\n'); channel.flush()\n")
        self.assertEqual(self.run_worker(code, {})['request_id'], 1)

    def test_worker_exit_fails(self):
        with self.assertRaises(RuntimeError):
            self.run_worker("import sys\nsys.stdin.readline()\nsys.exit(3)", {})

    def test_timeout_fails_and_stops_worker(self):
        with self.assertRaises(RuntimeError):
            self.run_worker("import sys,time\nsys.stdin.readline()\ntime.sleep(30)", {}, timeout=.1)

    def test_lazy_frames_rejects_future_before_image_loading(self):
        frames = LazyFrames(['zero','one'], 1024, 480, 640)
        torch_stub = types.SimpleNamespace()
        misc = types.ModuleType('sam2.utils.misc')
        misc._load_img_as_tensor = mock.Mock()
        with mock.patch.dict(sys.modules, {'torch': torch_stub, 'sam2.utils.misc': misc}):
            with self.assertRaises(RuntimeError):
                frames[1]
        misc._load_img_as_tensor.assert_not_called()

    def test_fp_clears_old_scores_before_early_return(self):
        worker = FoundationWorker.__new__(FoundationWorker)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        worker.scratch = Path(temp.name)
        np.savez(worker.scratch/'register_input.npz', rgb=np.zeros((2,2,3)), depth=np.ones((2,2)),
                 mask=np.ones((2,2)), K=np.eye(3))
        worker.est = types.SimpleNamespace(scores=np.array([999.]), pose_last=np.eye(4),
                                           register=lambda **kwargs: np.eye(4))
        with mock.patch.dict(sys.modules, {'torch': types.SimpleNamespace()}):
            result = worker.handle({'operation':'register', 'frame_index':0, 'iteration':5})
        self.assertFalse(result['diagnostics']['raw_valid'])
        self.assertFalse(hasattr(worker.est, 'scores'))


if __name__ == '__main__':
    unittest.main()
