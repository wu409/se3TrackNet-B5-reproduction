import os
import time
import unittest
from unittest import mock

import runtime_settings as settings
from ordered_prefetch import OrderedPrefetch


class SettingsTests(unittest.TestCase):
    def test_default_sixteen_cpu_eight_io(self):
        env = {}
        self.assertEqual(settings.configure_environment(env), 16)
        self.assertTrue(all(env[k] == '16' for k in settings.THREAD_KEYS))
        self.assertEqual(env['B5_IO_WORKERS'], '8')

    def test_budget_is_configurable_not_locked(self):
        env = {'B5_NUM_THREADS': '6', 'B5_IO_WORKERS': '2', 'OPENBLAS_NUM_THREADS': '32'}
        self.assertEqual(settings.configure_environment(env), 6)
        self.assertEqual(env['OPENBLAS_NUM_THREADS'], '6')
        self.assertEqual(env['B5_IO_WORKERS'], '2')

    def test_invalid_budget_rejected(self):
        for value in ('0', '-1', 'auto', ''):
            with self.assertRaises(ValueError):
                settings.configure_environment({'B5_NUM_THREADS': value})

    def test_small_pair_and_mismatch(self):
        settings.validate_sam_pair(settings.SAM2_DEFAULT_CONFIG, settings.SAM2_DEFAULT_CHECKPOINT)
        with self.assertRaises(ValueError):
            settings.validate_sam_pair(settings.SAM2_DEFAULT_CONFIG, 'sam2.1_hiera_large.pt')

    def test_frozen_budget_is_restored(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(settings, 'configure_libraries', side_effect=settings.configure_environment):
            settings.apply_frozen(dict(version='cpu_io_budget_v1', cpu_threads=4, io_workers=4))
            self.assertEqual(os.environ['B5_NUM_THREADS'], '4')


class PrefetchTests(unittest.TestCase):
    def test_order_and_bounded_submission(self):
        submitted = []
        def source():
            for value in range(20):
                submitted.append(value)
                yield value
        def load(value):
            time.sleep(.002 * (3-value % 4))
            return value
        loader = OrderedPrefetch(load, source(), 4)
        try:
            self.assertEqual(len(submitted), 4)
            self.assertEqual(next(loader), 0)
            self.assertEqual(len(submitted), 5)
            self.assertEqual(list(loader), list(range(1, 20)))
        finally:
            loader.close()
        self.assertTrue(loader.closed)

    def test_error_propagates_in_order_and_closes(self):
        def load(value):
            if value == 2:
                raise ValueError('bad frame')
            return value
        loader = OrderedPrefetch(load, range(6), 4)
        self.assertEqual(next(loader), 0)
        self.assertEqual(next(loader), 1)
        with self.assertRaisesRegex(ValueError, 'bad frame'):
            next(loader)
        self.assertTrue(loader.closed)


if __name__ == '__main__':
    unittest.main()
