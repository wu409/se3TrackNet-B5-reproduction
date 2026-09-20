import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import sam2_episode_cache as cache
import perception_runtime as runtime


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rgb = self.root/'frame.png'
        self.rgb.write_bytes(b'rgb input')
        self.prompt = self.root/'initial.png'
        self.prompt.write_bytes(b'initial prompt')
        self.args = types.SimpleNamespace(sam2_cache_root=str(self.root/'cache'))
        self.patcher = mock.patch.object(cache, 'profile', return_value={'fixed': 'model config'})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.ident = cache.identity(self.args, [self.rgb], self.prompt)
        self.directory = cache.folder(self.args.sam2_cache_root, self.ident)
        self.directory.mkdir(parents=True)
        self.mask = np.arange(35).reshape(5, 7) % 2 == 0
        self.path = self.directory/'frame.npz'
        with self.path.open('wb') as f:
            np.savez_compressed(f, bits=np.packbits(self.mask), shape=np.asarray(self.mask.shape))
        self.receipt = dict(status='complete', identity=self.ident, generation_wall_ms=25,
            masks=[dict(frame_index=0, rgb_path=str(self.rgb), file='frame.npz',
                        sha256=cache.sha(self.path), shape=[5, 7])])
        cache.atomic_json(self.directory/'COMPLETE.json', self.receipt)

    def test_lossless_and_input_fingerprint(self):
        reader = cache.CachedEpisode(self.args.sam2_cache_root, self.ident)
        np.testing.assert_array_equal(reader.read(0), self.mask)
        self.rgb.write_bytes(b'changed RGB')
        ident = cache.identity(self.args, [self.rgb], self.prompt)
        self.assertNotEqual(cache.digest(ident), cache.digest(self.ident))
        with self.assertRaises(FileNotFoundError):
            cache.CachedEpisode(self.args.sam2_cache_root, ident)

    def test_incomplete_never_consumed(self):
        (self.directory/'COMPLETE.json').unlink()
        with self.assertRaises(FileNotFoundError):
            cache.CachedEpisode(self.args.sam2_cache_root, self.ident)

    def test_corruption_detected_on_open_and_after_open(self):
        reader = cache.CachedEpisode(self.args.sam2_cache_root, self.ident)
        self.path.write_bytes(b'truncated')
        with self.assertRaises(ValueError): reader.read(0)
        with self.assertRaises(ValueError): cache.CachedEpisode(self.args.sam2_cache_root, self.ident)

    def test_complete_hit_does_not_create_sam_worker(self):
        with mock.patch.object(runtime, 'JsonWorker', side_effect=AssertionError('No inference allowed')):
            reader = cache.ensure_episode(self.args, [self.rgb], self.prompt)
        np.testing.assert_array_equal(reader.read(0), self.mask)

    def test_frame_mapping_and_path_escape_rejected(self):
        for change in ({'frame_index': 1}, {'rgb_path': 'wrong'}, {'file': '../escape.npz'}):
            receipt = json.loads(json.dumps(self.receipt))
            receipt['masks'][0].update(change)
            cache.atomic_json(self.directory/'COMPLETE.json', receipt)
            with self.assertRaises(ValueError):
                cache.CachedEpisode(self.args.sam2_cache_root, self.ident)

    def test_session_is_cache_only_and_reports_timing_scope(self):
        self.args.__dict__.update(sam2_python='unused', sam2_dir='unused', sam2_checkpoint='unused',
            sam2_config='unused', foundationpose_python='unused', foundationpose_dir='unused',
            foundationpose_refiner_weight='unused')
        with mock.patch.object(runtime, 'JsonWorker') as factory:
            session = runtime.PerceptionSession(self.args, [self.rgb], self.prompt, self.root/'mesh', self.root/'log')
            try:
                self.assertIsNone(session.sam)
                self.assertEqual(factory.call_count, 1)  # only lazy FP wrapper
                self.assertEqual(factory.call_args.args[1], 'foundationpose')
                session.advance(0, self.rgb)
                np.testing.assert_array_equal(session.mask, self.mask)
                _, _, diag = session.current_mask([self.rgb], self.prompt)
                self.assertTrue(diag['sam2_cache_hit'])
                self.assertFalse(diag['sam2_called'])
            finally:
                session.close(complete=True)
        summary = json.loads((self.root/'log_runtime.json').read_text())
        self.assertEqual(summary['sam2_timing_scope'], 'cache_read_only_not_online_fps')


if __name__ == '__main__':
    unittest.main()
