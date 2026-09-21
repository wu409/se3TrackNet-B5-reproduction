"""Development-only GPU equivalence/smoke test; never changes a training release.

Run with the b5-main Python. The selected release supplies only frozen paths and
manifest. Legacy prefix replay runs first, then persistent workers, avoiding
simultaneous legacy/new model copies on the GPU. No test GT is used.
"""
import runtime_settings
import argparse
import csv
import json
from pathlib import Path
import time
import types

import numpy as np


def resolve(value, root):
    path = Path(value)
    return str((path if path.is_absolute() else Path(root)/path).resolve())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', required=True)
    p.add_argument('--sequence', default='mustard0_clean')
    p.add_argument('--frames', type=int, default=30)
    p.add_argument('--compare-at', nargs='+', type=int, help='Zero-based frame indices; default last two frames')
    p.add_argument('--output', required=True, help='NEW directory, outside the selected frozen release')
    p.add_argument('--translation-tolerance-m', type=float, default=1e-5)
    p.add_argument('--rotation-tolerance-deg', type=float, default=1e-3)
    p.add_argument('--with-observer', action='store_true', help='Also run/restart SE3 to smoke-test co-resident models')
    p.add_argument('--sam2-config', default=runtime_settings.SAM2_DEFAULT_CONFIG)
    p.add_argument('--sam2-checkpoint', help='Defaults to Small in the selected SAM2 repo, NOT the old Large checkpoint')
    args = p.parse_args()
    release, output = Path(args.release).resolve(), Path(args.output).resolve()
    if output.exists() or output == release or release in output.parents:
        p.error('Output must be NEW and outside the frozen release')
    if args.frames < 2 or args.translation_tolerance_m < 0 or args.rotation_tolerance_deg < 0:
        p.error('Need >=2 frames and non-negative tolerances')
    from train_release import BASES, CAD
    from perception_runtime import CONFIG, PerceptionSession
    from b5_policy import sam2_mask_at_recovery, foundationpose_register_from_mask
    import cv2
    env = json.loads((release/'effective_config.json').read_text(encoding='utf-8'))['paths']
    with (release/'reference_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['sequence'] == args.sequence]
    rows.sort(key=lambda r: int(r['sequence_index']))
    if not rows or rows[0]['base_sequence'] not in BASES:
        p.error('Only development sequences are allowed')
    if args.frames > len(rows):
        p.error('--frames exceeds the sequence length')
    rows = rows[:args.frames]
    if [int(r['sequence_index']) for r in rows] != list(range(len(rows))):
        p.error('Non-contiguous manifest frame order')
    selected = sorted(set(args.compare_at if args.compare_at is not None else [len(rows)-2, len(rows)-1]))
    if not selected or any(i < 0 or i >= len(rows) for i in selected):
        p.error('Invalid --compare-at indices')
    output.mkdir(parents=True)
    cfg = types.SimpleNamespace(sam2_live_diagnostic=True, sam2_python=env['SAM2_PYTHON'], sam2_dir=env['SAM2_DIR'],
        sam2_config=args.sam2_config, sam2_checkpoint=args.sam2_checkpoint or str(Path(env['SAM2_DIR'])/'checkpoints'/runtime_settings.SAM2_DEFAULT_CHECKPOINT),
        foundationpose_python=env['FOUNDATIONPOSE_PYTHON'], foundationpose_dir=env['FOUNDATIONPOSE_DIR'],
        foundationpose_refiner_weight=env['FOUNDATIONPOSE_REFINER_WEIGHT'])
    runtime_settings.validate_sam_pair(cfg.sam2_config, cfg.sam2_checkpoint)
    if not Path(cfg.sam2_checkpoint).is_file():
        raise FileNotFoundError('Small checkpoint is required; no automatic Large fallback: ' + cfg.sam2_checkpoint)
    runtime_settings.configure_libraries()
    paths = [resolve(r['rgb_path'], env['DATASET_ROOT']) for r in rows]
    base = rows[0]['base_sequence']
    initial_mask = str(Path(env['GT_ROOT'])/base/'init_mask.png')
    mesh = str(Path(env['CAD_MODEL_ROOT'])/CAD[base]/'textured.obj')
    # Same frozen intrinsics as both label and evaluation code.
    K = np.array([[319.582000732421875, 0., 320.2149847676955687],
                  [0., 417.118682861328125, 244.3486680871046701], [0., 0., 1.]])
    fp_args = dict(K=K, mesh_file=mesh, foundationpose_python=cfg.foundationpose_python,
        foundationpose_dir=cfg.foundationpose_dir, foundationpose_refiner_weight=cfg.foundationpose_refiner_weight,
        refine_iter=5)
    def rgb_depth(index):
        rgb = cv2.imread(paths[index], cv2.IMREAD_COLOR)
        depth = cv2.imread(resolve(rows[index]['depth_path'], env['DATASET_ROOT']), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise RuntimeError('Unreadable RGB/depth at index ' + str(index))
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth.astype(np.float32)/1000.

    protocol = dict(runtime_config=CONFIG, execution_settings=runtime_settings.execution_config(),
        sam2_config=cfg.sam2_config, sam2_checkpoint=cfg.sam2_checkpoint, release=str(release), sequence=args.sequence,
        frames=len(rows), compare_at=selected, with_observer=args.with_observer,
        translation_tolerance_m=args.translation_tolerance_m, rotation_tolerance_deg=args.rotation_tolerance_deg,
        note='Development component equivalence diagnostic, not tracking accuracy or end-to-end FPS. No GT scoring.')
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    references = {}
    for index in selected:
        print('Legacy reference frame:', index, flush=True)
        start = time.perf_counter()
        mask, mask_ok, _ = sam2_mask_at_recovery(paths[:index+1], initial_mask,
            sam2_python=cfg.sam2_python, sam2_dir=cfg.sam2_dir, sam2_config=cfg.sam2_config,
            sam2_checkpoint=cfg.sam2_checkpoint, cache_root=str(output/'legacy_mask_cache'))
        sam_ms = (time.perf_counter()-start)*1000.
        pose, fp_ok, fp_ms = None, False, 0.
        if mask_ok:
            rgb, depth = rgb_depth(index)
            start = time.perf_counter()
            pose, fp_ok, diag = foundationpose_register_from_mask(rgb, depth, mask, **fp_args)
            fp_ms = (time.perf_counter()-start)*1000.
            if not fp_ok:
                print('Legacy registration absent:', diag, flush=True)
        references[index] = dict(mask=mask, mask_ok=mask_ok, pose=pose, fp_ok=fp_ok,
                                  legacy_sam_ms=sam_ms, legacy_fp_ms=fp_ms)
        if mask is not None:
            np.save(output/('reference_mask_%d.npy' % index), mask)
        if fp_ok:
            np.savetxt(output/('reference_pose_%d.txt' % index), pose)

    session = PerceptionSession(cfg, paths, initial_mask, mesh, output/'persistent')
    observer = None
    results, frames = [], []
    completed = False
    try:
        if args.with_observer:
            from online_observer import RestartableObserver
            observer = RestartableObserver(release/'observer_config.json', base, output/'observer.log')
        for index in range(len(rows)):
            session.advance(index, paths[index])
            if observer is not None:
                prediction = np.loadtxt(resolve(rows[index]['pred_path'], env['RESULT_ROOT'])).reshape(4,4)
                observer.observe(prediction, paths[index], resolve(rows[index]['depth_path'], env['DATASET_ROOT']))
            frame = dict(frame_index=index, sam2_wall_ms=session.wall_ms,
                sam2_worker_peak_allocated_mb=session.sam_diag.get('peak_allocated_mb'),
                sam2_worker_peak_reserved_mb=session.sam_diag.get('peak_reserved_mb'))
            frames.append(frame)
            if index not in references:
                continue
            ref = references[index]
            mask, ok, _ = session.current_mask(paths[:index+1], initial_mask)
            same = ref['mask'] is not None and np.array_equal(mask, ref['mask'])
            row = dict(frame_index=index, mask_exact=bool(same), mask_status_same=ok == ref['mask_ok'],
                legacy_sam_ms=ref['legacy_sam_ms'], legacy_fp_ms=ref['legacy_fp_ms'],
                persistent_sam_frame_ms=session.wall_ms, fp_compared=False,
                translation_delta_m=None, rotation_delta_deg=None, fp_equal=False)
            if ref['mask_ok']:
                # Identical mask for both FP implementations isolates lifecycle
                # effects from any SAM2 difference. No quality/GT-based selection.
                rgb, depth = rgb_depth(index)
                pose, valid, diag = session.register(rgb, depth, ref['mask'], K, mesh, 5)
                row['fp_status_same'] = bool(valid == ref['fp_ok'])
                row['persistent_fp_ms'] = diag['foundationpose_wall_ms']
                if valid and ref['fp_ok']:
                    translation = float(np.linalg.norm(pose[:3,3]-ref['pose'][:3,3]))
                    cosine = (np.trace(pose[:3,:3].T @ ref['pose'][:3,:3])-1.)/2.
                    rotation = float(np.degrees(np.arccos(np.clip(cosine, -1., 1.))))
                    row.update(fp_compared=True, translation_delta_m=translation, rotation_delta_deg=rotation,
                        fp_equal=translation <= args.translation_tolerance_m and rotation <= args.rotation_tolerance_deg)
                    # Repeating an identical request tests stale-state contamination.
                    repeated, repeated_ok, _ = session.register(rgb, depth, ref['mask'], K, mesh, 5)
                    row['fp_repeat_equal'] = bool(repeated_ok and np.allclose(pose, repeated, rtol=0., atol=1e-6))
                    if observer is not None:
                        observer.restart(pose)  # diagnostic co-residency exercise, not a B5 policy change
                else:
                    row['fp_repeat_equal'] = False
            results.append(row)
            print('Comparison:', row, flush=True)
        completed = True
    finally:
        try:
            if observer is not None:
                observer.close()
        finally:
            session.close(complete=completed)
        for name, data in (('comparison.csv', results), ('per_frame_runtime.csv', frames)):
            if data:
                with (output/name).open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=sorted({k for row in data for k in row}))
                    writer.writeheader()
                    writer.writerows(data)
    passed = bool(results and all(r['mask_exact'] and r['mask_status_same'] and r['fp_compared']
                  and r['fp_equal'] and r.get('fp_repeat_equal') for r in results))
    (output/'VALIDATION.json').write_text(json.dumps(dict(passed=passed, comparisons=len(results),
        limitations='Short component test is not proof of full-sequence equivalence, no OOM, or speedup.'), indent=2), encoding='utf-8')
    if not passed:
        raise SystemExit('Validation did not pass. Inspect differences/absent proposals; do not mix runtimes or tune on test GT.')
    print('Selected development comparisons passed:', output)


if __name__ == '__main__':
    main()
