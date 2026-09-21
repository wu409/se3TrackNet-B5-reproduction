"""Generate or verify all manifest episodes before B5 training; no GT scoring."""
import runtime_settings
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from sam2_episode_cache import ensure_episode, atomic_json, FORMAT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, help='Prepared release (effective_config + manifest)')
    parser.add_argument('--output', help='NEW index file outside a frozen release, if specified')
    args = parser.parse_args(argv)
    release = Path(args.release).resolve()
    output = Path(args.output).resolve() if args.output else release/'sam2_cache_index.json'
    if (release/'FROZEN.json').exists() and (output == release or release in output.parents):
        raise ValueError('Never write into a frozen release')
    if output.exists():
        raise FileExistsError('Use a NEW index output; shared cache is reused independently')
    effective = json.loads((release/'effective_config.json').read_text(encoding='utf-8'))
    runtime_settings.apply_frozen(effective['execution_settings'])
    env = effective['paths']
    cfg = SimpleNamespace(**{key.lower(): env[key] for key in (
        'SAM2_PYTHON', 'SAM2_DIR', 'SAM2_CONFIG', 'SAM2_CHECKPOINT', 'SAM2_CACHE_ROOT')})
    with (release/'reference_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    groups = {}
    for row in rows:
        groups.setdefault(row['sequence'], []).append(row)
    entries = []
    from sam2_episode_cache import sha
    for number, (seq, group) in enumerate(sorted(groups.items()), 1):
        group.sort(key=lambda r: int(r['sequence_index']))
        if [int(r['sequence_index']) for r in group] != list(range(len(group))):
            raise ValueError('Noncontiguous manifest: ' + seq)
        paths = []
        for row in group:
            p = Path(row['rgb_path'].replace('\\', '/'))
            if not p.is_absolute():
                p = Path(env['DATASET_ROOT'])/p
            if sha(p) != row['rgb_sha256'].lower():
                raise ValueError('RGB differs from frozen manifest: ' + str(p))
            paths.append(str(p.resolve()))
        bases = {r['base_sequence'] for r in group}
        if len(bases) != 1:
            raise ValueError('Ambiguous base sequence: ' + seq)
        initial = Path(env['GT_ROOT'])/next(iter(bases))/'init_mask.png'
        print('[SAM2 EPISODE %d/%d] %s' % (number, len(groups), seq), flush=True)
        cached = ensure_episode(cfg, paths, initial, seq)
        entries.append(dict(sequence=seq, frames=len(paths), receipt=str(cached.directory/'COMPLETE.json'),
                            sha256=cached.receipt_sha256))
    atomic_json(output, dict(format=FORMAT, cache_root=str(Path(cfg.sam2_cache_root).resolve()),
        manifest_sha256=sha(release/'reference_manifest.csv'), episodes=entries,
        note='All manifest RGB episodes, first-frame prompt only; no GT poses, labels, or quality fitting.'))
    print('SAM2 CACHE READY:', output, flush=True)


if __name__ == '__main__':
    main()
