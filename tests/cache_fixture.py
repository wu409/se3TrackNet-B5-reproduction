"""Synthetic cache provenance for CPU orchestration tests, not SAM2 output."""
from pathlib import Path
import numpy as np
import sam2_episode_cache as cache


def make_index(release, manifest='reference_manifest.csv'):
    release = Path(release)
    root = release.parent/'synthetic_shared_cache'
    ident = dict(profile={'synthetic': True}, initial_mask={}, frames=[{'path': 'fixture'}])
    folder = cache.folder(root, ident)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder/'mask.npz'
    with p.open('wb') as f:
        np.savez_compressed(f, bits=np.array([128], dtype=np.uint8), shape=np.array([1, 1]))
    cache.atomic_json(folder/'COMPLETE.json', dict(status='complete', identity=ident,
        generation_wall_ms=1, masks=[dict(frame_index=0, rgb_path='fixture', file='mask.npz',
                                         sha256=cache.sha(p), shape=[1, 1])]))
    cache.atomic_json(release/'sam2_cache_index.json', dict(format=cache.FORMAT,
        cache_root=str(root.resolve()), manifest_sha256=cache.sha(release/manifest),
        episodes=[dict(sequence='fixture', receipt=str(folder/'COMPLETE.json'),
                       sha256=cache.sha(folder/'COMPLETE.json'))]))
