"""Content-addressed causal SAM2 masks. No GT poses or B5 decisions are used.

Only COMPLETE episodes may be consumed. Interrupted episodes replay from frame
zero in a NEW attempt (a saved mask is not a saved SAM2 temporal state).
"""
import runtime_settings
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import numpy as np

FORMAT = "causal_sam2_episode_cache_v1"
_PROFILES = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.part')
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def profile(args):
    """Hash model/config/code/environment once per process; RGB is always hashed."""
    repo = Path(args.sam2_dir).resolve()
    settings = runtime_settings.execution_config()
    key = (str(repo), str(Path(args.sam2_checkpoint).resolve()), str(args.sam2_config),
           str(Path(args.sam2_python).resolve()), digest(settings))
    if key not in _PROFILES:
        runtime_settings.validate_sam_pair(args.sam2_config, args.sam2_checkpoint)
        config = repo / 'sam2' / args.sam2_config
        if not config.is_file():
            config = repo / args.sam2_config
        code = {str(p.relative_to(repo)): sha(p) for p in sorted((repo/'sam2').rglob('*'))
                if p.is_file() and p.suffix in ('.py', '.yaml', '.yml', '.so')}
        local = Path(__file__).resolve().parent
        pip = subprocess.run([args.sam2_python, '-m', 'pip', 'freeze'], check=True,
                             capture_output=True, text=True).stdout.splitlines()
        _PROFILES[key] = dict(format=FORMAT, checkpoint_sha256=sha(args.sam2_checkpoint),
            config_name=args.sam2_config, config_sha256=sha(config), sam2_source=code,
            worker_sha256=sha(local/'perception_workers.py'),
            cache_code_sha256=sha(__file__), runtime_settings_sha256=sha(local/'runtime_settings.py'),
            packages=sorted(pip), execution_settings=settings, seed=42,
            propagation='first_frame_prompt_forward_only_no_future_corrections',
            mask_rule='object_1_logits_gt_zero', postprocessing=True)
    return _PROFILES[key]


def identity(args, paths, initial_mask):
    paths = [str(Path(p).resolve()) for p in paths]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('Cache requires nonempty unique ordered RGB paths')
    prompt = str(Path(initial_mask).resolve())
    return dict(profile=profile(args), initial_mask=dict(path=prompt, sha256=sha(prompt)),
                frames=[dict(path=p, sha256=sha(p)) for p in paths])


def folder(root, ident):
    if not root:
        raise ValueError('A shared sam2_cache_root is required')
    return Path(root).resolve() / FORMAT / digest(ident)


def unpack(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        shape = tuple(int(x) for x in z['shape'])
        bits = z['bits']
    if len(shape) != 2 or min(shape) <= 0 or bits.dtype != np.uint8 or bits.ndim != 1:
        raise ValueError('Invalid packed mask')
    size = shape[0] * shape[1]
    if len(bits) != (size + 7) // 8:
        raise ValueError('Invalid packed mask length')
    return np.unpackbits(bits)[:size].reshape(shape).astype(bool)


class CachedEpisode:
    def __init__(self, root, ident):
        self.directory = folder(root, ident)
        receipt = self.directory/'COMPLETE.json'
        if not receipt.is_file():
            raise FileNotFoundError('SAM2 episode cache incomplete/missing; run prepare_sam_cache.py first: ' + str(receipt))
        self.receipt_sha256 = sha(receipt)
        self.receipt = json.loads(receipt.read_text(encoding='utf-8'))
        if self.receipt.get('identity') != ident or self.receipt.get('status') != 'complete':
            raise ValueError('SAM2 cache identity/status mismatch')
        self.entries = self.receipt['masks']
        if len(self.entries) != len(ident['frames']):
            raise ValueError('SAM2 cache frame count mismatch')
        # Verify all hashes now; decode/hash again at point of use (no silent repair).
        for i, entry in enumerate(self.entries):
            p = (self.directory/entry['file']).resolve()
            if (self.directory not in p.parents or entry['frame_index'] != i
                    or entry['rgb_path'] != ident['frames'][i]['path'] or sha(p) != entry['sha256']):
                raise ValueError('SAM2 cache frame/hash/path mismatch: ' + str(p))

    def read(self, index):
        entry = self.entries[index]
        data = (self.directory/entry['file']).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('SAM2 cached mask changed during run')
        mask = unpack(data)
        if list(mask.shape) != entry['shape']:
            raise ValueError('SAM2 cached mask shape mismatch')
        return mask


def ensure_episode(args, paths, initial_mask, label='episode'):
    from perception_runtime import JsonWorker
    ident = identity(args, paths, initial_mask)
    directory = folder(args.sam2_cache_root, ident)
    if (directory/'COMPLETE.json').exists():
        cached = CachedEpisode(args.sam2_cache_root, ident)
        print('[SAM2 CACHE HIT] ' + label, flush=True)
        return cached
    directory.mkdir(parents=True, exist_ok=True)
    # Linux advisory lock is released by the OS on crash. A concurrent writer
    # fails clearly; it never publishes over another writer's COMPLETE marker.
    import fcntl
    with (directory/'build.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another process is building this SAM2 episode: ' + label)
        if (directory/'COMPLETE.json').exists():
            return CachedEpisode(args.sam2_cache_root, ident)
        attempt = directory/('attempt_' + uuid.uuid4().hex)
        attempt.mkdir()
        scratch = attempt/'worker'
        scratch.mkdir()
        cfg = dict(scratch=str(scratch), seed=42, repo=str(Path(args.sam2_dir).resolve()),
                   checkpoint=str(Path(args.sam2_checkpoint).resolve()), model_cfg=args.sam2_config,
                   rgb_paths=[f['path'] for f in ident['frames']], initial_mask=ident['initial_mask']['path'])
        worker = JsonWorker(args.sam2_python, 'sam2', cfg, scratch, attempt/'sam2.log')
        entries = []
        start = time.perf_counter()
        try:
            for i, rgb in enumerate(cfg['rgb_paths']):
                tick = time.perf_counter()
                result = worker.call(operation='advance', frame_index=i, rgb_path=rgb)
                if result.get('frame_index') != i:
                    raise RuntimeError('SAM2 cache received wrong frame')
                mask = np.load(scratch/'current_mask.npy', allow_pickle=False)
                if mask.dtype != np.bool_ or mask.ndim != 2:
                    raise ValueError('SAM2 produced invalid mask')
                path = attempt/('%08d.npz' % i)
                with path.with_suffix('.part').open('xb') as f:
                    np.savez_compressed(f, bits=np.packbits(mask), shape=np.asarray(mask.shape, dtype=np.int64))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(str(path.with_suffix('.part')), str(path))
                entries.append(dict(frame_index=i, rgb_path=rgb, file=path.relative_to(directory).as_posix(),
                    sha256=sha(path), shape=list(mask.shape), generation_wall_ms=(time.perf_counter()-tick)*1000))
                if i == 0 or (i+1) % 50 == 0 or i+1 == len(cfg['rgb_paths']):
                    elapsed = time.perf_counter()-start
                    print('[SAM2 BUILD] %s %d/%d %.2f fps elapsed=%.1fs' %
                          (label, i+1, len(cfg['rgb_paths']), (i+1)/elapsed, elapsed), flush=True)
            # Do not certify masks against inputs that changed while generating.
            if identity(args, paths, initial_mask) != ident:
                raise ValueError('SAM2 inputs changed during preprocessing')
            for e in entries:
                if sha(directory/e['file']) != e['sha256']:
                    raise ValueError('Mask changed before publication')
                unpack((directory/e['file']).read_bytes())
            atomic_json(directory/'COMPLETE.json', dict(status='complete', identity=ident,
                masks=entries, label=label, generation_wall_ms=(time.perf_counter()-start)*1000,
                note='Offline preprocessing cost; cached consumption is not end-to-end online FPS.'))
        finally:
            worker.close()
    return CachedEpisode(args.sam2_cache_root, ident)


def verify_index(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('format') != FORMAT or not data.get('episodes'):
        raise ValueError('Missing SAM2 cache index')
    for item in data['episodes']:
        receipt = Path(item['receipt'])
        if sha(receipt) != item['sha256']:
            raise ValueError('SAM2 cache receipt changed: ' + str(receipt))
        value = json.loads(receipt.read_text(encoding='utf-8'))
        cached = CachedEpisode(data['cache_root'], value['identity'])
        if cached.directory/'COMPLETE.json' != receipt:
            raise ValueError('SAM2 cache receipt location mismatch')
    return data
