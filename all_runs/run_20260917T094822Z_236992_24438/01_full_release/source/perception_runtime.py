"""Episode-scoped persistent perception IPC. Python 3.8; no torch in parent.

Production consumes precomputed causal SAM2 masks; live SAM2 is diagnostic only.
FoundationPose stays resident within an episode. Worker failure is a
run failure, never a silent policy fallback or an unreported prefix replay.
"""
import runtime_settings
import atexit
import functools
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import tempfile
import threading
import time

import numpy as np

CONFIG = dict(version="cached_perception_v2", sam2="precomputed_causal_episode_masks",
              scheduling="serialized_gpu_resident_workers", frame_input="causal_lazy_rgb_float32",
              sam2_video_storage="cpu_one_image_cache", sam2_state_storage="cpu_no_memory_pruning",
              foundationpose="resident_models_independent_register_per_request",
              cache="complete_content_addressed_episode_read_only", timeout_seconds=1200,
              failure="abort_no_silent_fallback", seed=42)
_ACTIVE = None


def active_session():
    return _ACTIVE


def close_episode_perception(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        if active_session() is not None:
            raise RuntimeError("Nested perception episodes are not supported")
        ok = False
        try:
            result = function(*args, **kwargs)
            ok = True
            return result
        finally:
            if active_session() is not None:
                active_session().close(complete=ok)
    return wrapped


class JsonWorker:
    """Single outstanding request; stdout is JSON only, library logs go to file."""
    def __init__(self, python, kind, config, scratch, log_path, timeout=None):
        self.timeout = CONFIG["timeout_seconds"] if timeout is None else timeout
        self.kind, self.config = kind, config
        self.scratch, self.log_path = Path(scratch), Path(log_path)
        self.python = str(python)
        self.process = self.log = self.reader = None
        self.responses = queue.Queue()
        self.counter = 0
        self.broken = False

    def _start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        config_path = self.scratch / (self.kind + "_config.json")
        config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.log = self.log_path.open("a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                [self.python, "-u", "-B", str(Path(__file__).with_name("perception_workers.py")),
                 self.kind, str(config_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, text=True, encoding="utf-8",
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED=str(CONFIG["seed"])))
        except BaseException:
            self.log.close()
            self.log = None
            raise
        def read():
            try:
                for line in self.process.stdout:
                    self.responses.put(line)
            finally:
                self.responses.put(None)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def call(self, **request):
        if self.broken:
            raise RuntimeError("Worker is broken; start a NEW run")
        try:
            if self.process is None:
                self._start()
            self.counter += 1
            request["request_id"] = self.counter
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            line = self.responses.get(timeout=self.timeout)
            if line is None:
                raise RuntimeError("worker exited without a response")
            result = json.loads(line)
            if result.get("request_id") != self.counter:
                raise RuntimeError("worker response ID mismatch")
            if result.get("error"):
                raise RuntimeError(result["error"])
            return result
        except (Exception, KeyboardInterrupt) as exc:
            self.broken = True
            self.close()
            raise RuntimeError("%s worker failed: %s; inspect %s" % (self.kind, exc, self.log_path)) from exc

    def close(self):
        process = self.process
        if process is not None:
            try:
                process.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if self.reader:
                self.reader.join(timeout=5)
            process.stdout.close()
            self.process = None
        if self.log:
            self.log.close()
            self.log = None


def cleanup_owned_mesh(scratch):
    """Only an exact UUID export recorded by this worker, after it has stopped."""
    receipt = Path(scratch) / "owned_mesh.json"
    if not receipt.is_file():
        return
    path = Path(json.loads(receipt.read_text(encoding="utf-8"))["path"])
    if (path.parent != Path("/tmp") or path.is_symlink()
            or not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\.obj", path.name)):
        raise RuntimeError("Refusing to clean unexpected FoundationPose export: " + str(path))
    if path.is_file():
        path.unlink()


class PerceptionSession:
    def __init__(self, args, rgb_paths, initial_mask_path, mesh_file, log_prefix):
        global _ACTIVE
        runtime_settings.configure_libraries()
        runtime_settings.validate_sam_pair(args.sam2_config, args.sam2_checkpoint)
        if _ACTIVE is not None:
            raise RuntimeError("Previous perception episode was not closed")
        self.paths = [str(Path(p).resolve()) for p in rgb_paths]
        if not self.paths or len(set(self.paths)) != len(self.paths):
            raise ValueError("Perception requires a non-empty, unique ordered RGB manifest")
        self.initial_mask_path = str(Path(initial_mask_path).resolve())
        self.mesh_file = str(Path(mesh_file).resolve())
        self.cached = None
        # Live mode is restricted to explicit component-equivalence diagnostics.
        if not getattr(args, 'sam2_live_diagnostic', False):
            from sam2_episode_cache import CachedEpisode, identity
            self.cached = CachedEpisode(args.sam2_cache_root,
                identity(args, self.paths, self.initial_mask_path))
        prefix = Path(log_prefix).resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="b5_perception_", dir=str(prefix.parent))
        self.scratch = Path(self.temp.name)
        self.summary_path = prefix.with_name(prefix.name + "_runtime.json")
        self.index, self.mask, self.wall_ms = -1, None, 0.
        self.sam_diag = {}
        self.sam_total_ms = self.fp_total_ms = 0.
        self.fp_calls = 0
        self.loaders = []
        self.started = time.perf_counter()
        self.closed = False
        shared = dict(scratch=str(self.scratch), seed=CONFIG["seed"])
        sam_cfg = dict(shared, repo=str(Path(args.sam2_dir).resolve()),
                       checkpoint=str(Path(args.sam2_checkpoint).resolve()), model_cfg=args.sam2_config,
                       rgb_paths=self.paths, initial_mask=self.initial_mask_path)
        fp_cfg = dict(shared, repo=str(Path(args.foundationpose_dir).resolve()), mesh=self.mesh_file,
                      refiner_weight=str(Path(args.foundationpose_refiner_weight).resolve()))
        self.sam = (None if self.cached else JsonWorker(args.sam2_python, "sam2", sam_cfg,
                    self.scratch, str(prefix) + "_sam2.log"))
        self.fp = JsonWorker(args.foundationpose_python, "foundationpose", fp_cfg, self.scratch,
                             str(prefix) + "_foundationpose.log")
        _ACTIVE = self

    def prefetch(self, function, iterable):
        from ordered_prefetch import OrderedPrefetch
        loader = OrderedPrefetch(function, iterable, runtime_settings.execution_config()['io_workers'])
        self.loaders.append(loader)
        return loader

    def advance(self, index, rgb_path):
        index = int(index)
        if index != self.index + 1 or index >= len(self.paths):
            raise ValueError("SAM2 frames must advance exactly once in manifest order")
        if str(Path(rgb_path).resolve()) != self.paths[index]:
            raise ValueError("SAM2 frame path mismatch")
        start = time.perf_counter()
        if self.cached:
            self.mask = self.cached.read(index)
            result = dict(frame_index=index, cache_hit=True)
        else:
            result = self.sam.call(operation="advance", frame_index=index, rgb_path=self.paths[index])
            self.mask = np.load(self.scratch / "current_mask.npy", allow_pickle=False).astype(bool)
        if result.get("frame_index") != index:
            raise RuntimeError("SAM2 returned a stale/future frame")
        if self.mask.ndim != 2:
            raise RuntimeError("SAM2 mask must be two dimensional")
        self.index = index
        self.wall_ms = (time.perf_counter() - start) * 1000.
        self.sam_total_ms += self.wall_ms
        self.sam_diag = result

    def current_mask(self, rgb_paths, initial_mask_path):
        if ([str(Path(p).resolve()) for p in rgb_paths] != self.paths[:self.index + 1]
                or self.index < 0 or str(Path(initial_mask_path).resolve()) != self.initial_mask_path):
            raise RuntimeError("Recovery request does not match the active SAM2 frame/initial prompt")
        pixels = int(self.mask.sum())
        diagnostics = dict(sam2_called=self.cached is None, sam2_subprocess_started=False,
            sam2_cache_hit=self.cached is not None, sam2_persistent=self.cached is None, sam2_current_frame_reused=True,
            sam2_target_index=self.index, sam2_frames_supplied=self.index + 1,
            sam2_propagated_frames=0 if self.cached else self.index + 1, sam2_mask_pixels=pixels,
            sam2_timing_scope='cache_read_only' if self.cached else 'live_inference',
            sam2_error=None if pixels >= 20 else "sam2_mask_too_small",
            sam2_frame_wall_ms=self.wall_ms, sam2_peak_allocated_mb=self.sam_diag.get("peak_allocated_mb"))
        return self.mask.copy(), pixels >= 20, diagnostics

    def register(self, rgb, depth, mask, K, mesh_file, refine_iter):
        if self.index < 0 or str(Path(mesh_file).resolve()) != self.mesh_file:
            raise ValueError("FoundationPose episode/mesh mismatch")
        rgb, depth, mask = np.asarray(rgb, dtype=np.uint8), np.asarray(depth, dtype=np.float32), np.asarray(mask, dtype=bool)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != mask.shape or rgb.shape[:2] != mask.shape:
            raise ValueError("FoundationPose RGB/depth/mask shape mismatch")
        if mask.sum() < 20:
            return None, False, {"foundationpose_error": "recovery_mask_too_small"}
        start = time.perf_counter()
        np.savez(self.scratch / "register_input.npz", rgb=rgb, depth=depth, mask=mask,
                 K=np.asarray(K, dtype=np.float64).reshape(3, 3))
        result = self.fp.call(operation="register", frame_index=self.index, iteration=int(refine_iter))
        if result.get("frame_index") != self.index:
            raise RuntimeError("FoundationPose returned a stale/future pose")
        self.fp_calls += 1
        wall_ms = (time.perf_counter() - start) * 1000.
        self.fp_total_ms += wall_ms
        pose = np.asarray(result["pose"], dtype=np.float64).reshape(4, 4)
        diagnostics = result["diagnostics"]
        diagnostics.update(foundationpose_persistent=True, foundationpose_wall_ms=wall_ms,
                           foundationpose_peak_allocated_mb=result.get('peak_allocated_mb'),
                           foundationpose_peak_reserved_mb=result.get('peak_reserved_mb'),
                           foundationpose_request_count=self.fp_calls)
        valid = bool(np.isfinite(pose).all() and diagnostics.get("raw_valid", False))
        return pose if valid else None, valid, diagnostics

    def close(self, complete=False):
        global _ACTIVE
        if self.closed:
            return
        self.closed = True
        try:
            try:
                for loader in self.loaders:
                    loader.close()
            finally:
                try:
                    if self.sam is not None:
                        self.sam.close()
                finally:
                    self.fp.close()
            cleanup_owned_mesh(self.scratch)
            self.summary_path.write_text(json.dumps(dict(config=CONFIG, execution_settings=runtime_settings.execution_config(), complete=bool(complete),
                frames_advanced=self.index + 1, manifest_frames=len(self.paths),
                sam2_wall_ms=self.sam_total_ms, foundationpose_wall_ms=self.fp_total_ms,
                foundationpose_calls=self.fp_calls, episode_scope_wall_ms=(time.perf_counter()-self.started)*1000.,
                sam2_log=str(self.sam.log_path) if self.sam else None,
                sam2_timing_scope='cache_read_only_not_online_fps' if self.cached else 'live_inference',
                sam2_cache_receipt=str(self.cached.directory/'COMPLETE.json') if self.cached else None,
                sam2_cache_receipt_sha256=self.cached.receipt_sha256 if self.cached else None,
                sam2_precompute_wall_ms=self.cached.receipt['generation_wall_ms'] if self.cached else None,
                foundationpose_log=str(self.fp.log_path)), indent=2), encoding="utf-8")
        finally:
            _ACTIVE = None
            self.temp.cleanup()


@atexit.register
def _cleanup():
    if _ACTIVE is not None:
        _ACTIVE.close()
