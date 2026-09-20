# Claim controls (post-development diagnostics)

Run from the Linux se3TrackNet-B5-reproduction checkout with the original
`b5-main` environment (`TEST_PYTHON` may override Python). The default release
paths are the exact 2026-09-17 `run_all.sh` pair; override `EXPERIMENT_RUN_ROOT`
or `Q0_RELEASE` and `FULL_RELEASE` explicitly. Set `CLAIM_OUTPUT` for a chosen
**new**, non-existing output path; otherwise each invocation creates a unique
directory under `claim_control_runs/`. Nothing writes into frozen releases,
`all_runs/`, or prior evaluation outputs. Run `bash claim_controls/01_...sh`, etc.
The online scripts accept `--check-only` and preserve its new preflight folder.

| Script | Question | Output |
|---|---|---|
| `01_shared_vs_source_specific.sh` | One frozen shared q1 versus independently fitted obs/prior heads on identical q0/q1 candidate tables | `shared_source_*/metrics.csv` |
| `02_rollout_refit_attribution.sh` | q0 versus q1 on the **same** q0-policy candidates, then on q1-policy candidates | `refit_*/metrics.csv`, `provenance.json` |
| `03_calibrated_vs_raw_threshold.sh` | Frozen isotonic risk cutoff versus raw predicted-cm cutoff tuned on development calibration frames | `calibration_*/metrics.csv`, route disagreement in `provenance.json` |
| `04_no_observer_reseed.sh` | Does registration's observer restart change subsequent observations and tracking? | `no_reseed_*/` standard per-frame/episode outputs |
| `05_mode3_motion_history_reset.sh` | Does zero-velocity restart after *used* MODE3 change subsequent priors and tracking? | `mode3_reset_*/` standard per-frame/episode outputs |

Example:

```bash
export TEST_PYTHON=/root/autodl-tmp/conda-envs/b5-main/bin/python3.8
bash claim_controls/01_shared_vs_source_specific.sh
bash claim_controls/02_rollout_refit_attribution.sh
bash claim_controls/03_calibrated_vs_raw_threshold.sh
bash claim_controls/04_no_observer_reseed.sh --check-only
bash claim_controls/05_mode3_motion_history_reset.sh --check-only
```

For the **prior examples versus policy-induced data** separation, the existing
q1 was fitted on q0-policy rollout pairs. Thus those pairs simultaneously add
priors *and* reflect q0 decisions. It is invalid to name q1 a "prior-only"
control. If an independent observation-only/fixed-policy rollout cache is
available, pass `FIXED_POLICY_SAMPLES=/absolute/path.csv` to script 02. Its
schema is the paired long-form development table: `sequence`, `sequence_index`,
`frame_id`, `D_obj_cm`, `hypothesis` (`obs` or `prior`), `target_E_cm`,
`target_e_norm`, and four features `x1_norm`, `x2_inlier_error`,
`x4_support_ratio`, `x5_geometry_inconsistency`. Generate it with the same
frozen observer and candidate-feature implementation, on the same development
episodes under a fixed observation-only policy; do **not** relabel q0-policy
rows as fixed-policy. Without this cache, script 02 still runs q0/q1
same-candidate comparisons and explicitly records the missing control.

All five evaluation sequences have informed development. Offline controls
do not establish closed-loop tracking gains; compare the online interventions
against the existing `03_full_simple/full` per-sequence outputs. Isotonic is
monotone, so a separately tuned raw cutoff may produce identical routing;
do not claim calibration adds discrimination solely from threshold performance.
The Windows copy of historical releases contains Linux absolute paths in its
freeze receipts, so full online preflight and GPU inference must run on the
original Linux checkout with assets and compatible environments.
