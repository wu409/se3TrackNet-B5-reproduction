"""Restartable observation stream. No GT access; legacy predictions before reset.

The worker runs in the explicitly frozen SE3 environment. After a correction it
tracks from its own last observation, never from every B5 output. RGB is RGB;
depth passed to Tracker is the original uint16 millimetre image (no hole filling).
"""
import runtime_settings
import atexit
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from pose_safety import is_se3

VERSION = "restartable_se3_observer_v2"
_LIVE = set()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build_config(env, sequence_objects):
    weights = Path(env['SE3_WEIGHT_ROOT'])
    data = Path(env['SE3_DATA_ROOT'])
    cad = Path(env['CAD_MODEL_ROOT'])
    objects = {}
    for obj in sorted(set(sequence_objects.values())):
        name = obj.split('_', 1)[1]
        objects[obj] = dict(checkpoint=str(weights/name/'model_best_val.pth.tar'),
            mean=str(weights/name/'mean.npy'), std=str(weights/name/'std.npy'),
            dataset_info=str(data/name/'dataset_info.yml'), mesh=str(cad/obj/'textured.obj'))
    return dict(version=VERSION, python=env['SE3_PYTHON'], objects=objects,
                sequence_objects=sequence_objects, seed=42, samples=1,
                trans_normalizer=0.03, rot_normalizer_degrees=30,
                depth_input='manifest_uint16_mm_no_filling',
                initial_stream='frozen_precomputed_predictions_until_first_correction',
                after_restart='previous_tracker_observation_not_previous_policy_output')


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    if cfg.get('version') != VERSION:
        raise ValueError('Expected four-mode v2 observer configuration; retrain a new release')
    if not isinstance(cfg.get('sequence_objects'), dict) or not isinstance(cfg.get('objects'), dict):
        raise ValueError('Invalid frozen observer asset mappings')
    return cfg


def asset_paths(cfg):
    yield Path(cfg['python'])
    for item in cfg['objects'].values():
        for value in item.values():
            yield Path(value)
        # Textures/materials affect render-and-compare, not just the OBJ bytes.
        for p in Path(item['mesh']).parent.rglob('*'):
            if p.is_file():
                yield p


def validate_config(path):
    cfg = read_config(path)
    for asset in asset_paths(cfg):
        if not asset.is_file():
            raise FileNotFoundError('SE3 observation restart asset missing: ' + str(asset))
    return cfg


def close_episode_observers(function):
    """Close even when an episode fails; functions are single-process sequential."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        before = set(_LIVE)
        try:
            return function(*args, **kwargs)
        finally:
            for observer in set(_LIVE) - before:
                observer.close()
    return wrapped


class RestartableObserver:
    def __init__(self, config_path, base, log_path):
        self.path = str(Path(config_path).resolve())
        self.config = read_config(self.path)
        if base not in self.config['sequence_objects']:
            raise ValueError('No frozen SE3 object mapping for ' + base)
        self.base, self.log_path = base, Path(log_path)
        self.previous = None
        self.process = self.log = None
        self.wall_ms = 0.
        self.source = 'precomputed'
        self.pending_reset = False
        _LIVE.add(self)

    def restart(self, pose):
        if not is_se3(pose):
            raise ValueError('Cannot restart tracker with an invalid SE3 pose')
        self.previous = np.asarray(pose).copy()
        self.pending_reset = True

    def observe(self, legacy_pose, rgb_path, depth_path):
        self.wall_ms = 0.
        if self.previous is None:
            self.source = 'precomputed'
            return np.asarray(legacy_pose).copy()
        start = time.perf_counter()
        self.source = 'restarted_se3'
        if self.process is None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log = self.log_path.open('a', encoding='utf-8')
            self.process = subprocess.Popen([self.config['python'], '-u', '-B',
                str(Path(__file__).resolve()), '--worker', self.path, self.base],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                text=True, encoding='utf-8', env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        message = dict(pose=self.previous.tolist(), rgb=str(rgb_path), depth=str(depth_path),
                       reset=self.pending_reset)
        self.process.stdin.write(json.dumps(message) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('SE3 worker exited; inspect ' + str(self.log_path))
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError('SE3 worker failed: ' + result['error'] + '; log=' + str(self.log_path))
        pose = np.asarray(result['pose'], dtype=float)
        if not is_se3(pose):
            raise ValueError('SE3 worker returned invalid pose; no silent legacy fallback')
        self.previous, self.pending_reset = pose.copy(), False
        self.wall_ms = (time.perf_counter() - start) * 1000.
        return pose

    def close(self):
        if self.process is not None:
            try:
                try:
                    self.process.stdin.close()
                except (OSError, ValueError):
                    pass
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.process.stdout.close()
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
        _LIVE.discard(self)


@atexit.register
def _cleanup():
    for observer in list(_LIVE):
        observer.close()


def worker(config_path, base, check_only=False):
    # Keep imported tracker prints away from the JSON IPC channel.
    channel = sys.stdout
    sys.stdout = sys.stderr
    import random
    import cv2
    import yaml
    import torch
    from PIL import Image
    from predict import Tracker
    runtime_settings.configure_libraries()
    cfg = validate_config(config_path)
    assets = cfg['objects'][cfg['sequence_objects'][base]]
    info = yaml.safe_load(Path(assets['dataset_info']).read_text())
    expected = dict(focalX=319.582000732421875, focalY=417.118682861328125,
                    centerX=320.21498476769557, centerY=244.34866808710467,
                    width=640, height=480)
    if any(not np.isclose(info['camera'][k], v) for k, v in expected.items()):
        raise ValueError('SE3 camera differs from frozen B5 camera')
    if info.get('renderer') != 'pyrenderer':
        raise ValueError('Expected pyrenderer with frozen textured.obj')
    if check_only:
        return  # imports/config only; no model load or GPU allocation
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    tracker = Tracker(info, np.load(assets['mean']), np.load(assets['std']), assets['checkpoint'],
        model_path=assets['mesh'], trans_normalizer=cfg['trans_normalizer'],
        rot_normalizer=cfg['rot_normalizer_degrees'] * np.pi / 180.)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            rgb = np.asarray(Image.open(request['rgb']).convert('RGB'))
            depth = cv2.imread(request['depth'], cv2.IMREAD_UNCHANGED)
            if (depth is None or depth.dtype != np.uint16 or depth.shape != rgb.shape[:2]
                    or depth.shape != (info['camera']['height'], info['camera']['width'])):
                raise ValueError('Expected matching RGB and uint16-mm depth images')
            if request['reset']:
                tracker.prev_rgb = tracker.prev_depth = None
                tracker.frame_cnt = 0
            with torch.no_grad():
                pose = tracker.on_track(np.asarray(request['pose'], dtype=float), rgb, depth,
                    gt_A_in_cam=np.eye(4), gt_B_in_cam=np.eye(4), debug=False, samples=cfg['samples'])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            channel.write(json.dumps(dict(pose=np.asarray(pose).tolist()), allow_nan=False) + '\n')
            channel.flush()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            channel.write(json.dumps(dict(error=str(exc))) + '\n')
            channel.flush()
            raise


if __name__ == '__main__':
    worker(sys.argv[2], sys.argv[3], check_only=sys.argv[1] == '--check')
