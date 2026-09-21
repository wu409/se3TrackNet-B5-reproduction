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
| `02_rollout_refit_attribution.sh` | Collect genuine fixed-policy pairs; compare observation-only, count-matched duplicated observations, fixed-policy pairs and q0-policy pairs | `refit_complete_*/metrics.csv`, `models/`, `fixed_policy_cache/`, `COMPLETE.json` |
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

Script 02 was completed on 2026-09-21: it now automatically generates the
missing observation-only-policy candidate cache and fits matched lightweight
controls using the exact frozen feature/fitting implementation. It supports
`--smoke` (four real development frames, no fitting) and `--check-only`.
Reuse a verified cache via `FIXED_POLICY_CACHE=/prior/output/fixed_policy_cache`;
bare unverified CSVs are rejected. The original limited q0/q1 diagnostic is
still available with `REFIT_SAME_CANDIDATE_ONLY=1`. Previously completed output
folders retain their original `NOT RUN` record. See
[REFIT_COMPLETE_GUIDE.md](REFIT_COMPLETE_GUIDE.md) for comparisons, outputs and
interpretation limits. The newly written collection pipeline has been tested;
its scientific result exists only after a full run writes `COMPLETE.json`.

All five evaluation sequences have informed development. Offline controls
do not establish closed-loop tracking gains; compare the online interventions
against the existing `03_full_simple/full` per-sequence outputs. Isotonic is
monotone, so a separately tuned raw cutoff may produce identical routing;
do not claim calibration adds discrimination solely from threshold performance.
The Windows copy of historical releases contains Linux absolute paths in its
freeze receipts, so full online preflight and GPU inference must run on the
original Linux checkout with assets and compatible environments.
