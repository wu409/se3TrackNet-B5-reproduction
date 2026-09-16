"""Combine matched runs, preserving all sequence outcomes and object clustering."""
from pathlib import Path
import numpy as np
import pandas as pd
import test_release as evaluation


def validate_sequences(table, variant):
    if table.base_sequence.duplicated().any() or set(table.base_sequence) != set(evaluation.TEST):
        raise ValueError("Missing/duplicate test sequence: " + variant)
    if any(row.object_id != evaluation.TEST[row.base_sequence] for row in table.itertuples()):
        raise ValueError("Incorrect object clustering: " + variant)
    for column in ("auc_percent", "failure_1cm_percent", "failure_2cm_percent"):
        if not np.isfinite(table[column]).all():
            raise ValueError("Nonfinite tracking outcome: " + variant + "/" + column)


def paired_tracking(sequences, seed):
    full = sequences[sequences.variant == "full"].set_index("base_sequence").sort_index()
    records = []
    controls = [v for v in sequences.variant.unique() if v != "full"] + ["B1"]
    for control in controls:
        other = full if control == "B1" else sequences[sequences.variant == control].set_index("base_sequence").reindex(full.index)
        for metric in ("auc_percent", "failure_1cm_percent", "failure_2cm_percent"):
            if control == "B1" and metric != "auc_percent":
                continue
            delta = full[metric] - other["b1_auc_percent" if control == "B1" else metric]
            if not np.isfinite(delta).all():
                raise ValueError("Invalid paired outcome")
            common = dict(comparison="full-minus-" + control, metric=metric,
                          unit="percentage_points", favorable_sign="positive" if metric == "auc_percent" else "negative")
            records.extend(dict(common, base_sequence=base, estimate=float(value), aggregation="sequence")
                           for base, value in delta.items())
            records.append(dict(common, base_sequence="SEQUENCE_MACRO", estimate=float(delta.mean()),
                                aggregation="equal_weight_five_sequences"))
            obj = delta.groupby(full.object_id).mean().to_numpy()
            rng = np.random.default_rng(seed)
            boot = rng.choice(obj, size=(10000, len(obj)), replace=True).mean(axis=1)
            records.append(dict(common, base_sequence="OBJECT_MACRO", estimate=float(obj.mean()),
                ci_low=float(np.percentile(boot, 2.5)), ci_high=float(np.percentile(boot, 97.5)),
                independent_objects=len(obj), aggregation="object_cluster_bootstrap_exploratory_n3"))
    return pd.DataFrame(records)


def recovery_summary(events):
    rows = []
    for variant, group in events.groupby("variant"):
        groups = list(group.groupby("base_sequence")) + [("ALL_EVENTS_DESCRIPTIVE", group)]
        for base, data in groups:
            row = dict(variant=variant, base_sequence=base, events=len(data))
            for column in ("trigger_count", "sam2_requested_count", "sam2_cache_hit_count",
                           "generated_count", "accepted_count", "used_count", "rejected_count",
                           "not_generated_count", "raw_good_count", "accepted_good_count",
                           "B5_latency_success", "B5_latency_censored"):
                row[column] = int(data[column].sum()) if column in data else np.nan
            for name, numerator, denominator in (
                ("raw_good_percent", "raw_good_count", "generated_count"),
                ("accepted_good_percent", "accepted_good_count", "accepted_count"),
                ("acceptance_percent", "accepted_count", "generated_count"),
                ("rejection_percent", "rejected_count", "generated_count"),
                ("successful_recovery_percent", "B5_latency_success", "events")):
                den = row[denominator]
                row[name] = 100. * row[numerator] / den if den > 0 else np.nan
            complete = data[data["B5_window_complete"] == 1] if "B5_window_complete" in data else data.iloc[:0]
            row["complete_window_events"] = len(complete)
            row["incomplete_window_events"] = len(data) - len(complete)
            for column in ("B5_window_auc_percent", "B5_window_failure_percent"):
                if column in data:
                    row[column] = float(complete[column].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(output, full_run, controls_run, q0_run, reuse_simple, seed):
    specs = [(Path(full_run), "full", "full")]
    if reuse_simple:
        specs.append((Path(full_run), "simple", "no_quality"))
    available = evaluation.read_json(Path(controls_run) / "COMPLETE.json")["variants"]
    specs += [(Path(controls_run), v, v) for v in available if v != "full"]
    specs.append((Path(q0_run), "no_rollout", "no_rollout"))
    combined = {name: [] for name in ("sequence_metrics", "episode_metrics", "recovery_events", "calibration_metrics")}
    inputs = {}
    for run, original, label in specs:
        for name in combined:
            path = run / (name + ".csv")
            inputs[str(path.resolve())] = evaluation.sha(path)
            data = pd.read_csv(path)
            data = data[data.variant == original].copy()
            data["variant"] = label
            data["source_variant"] = original
            data["source_run"] = str(run)
            if name == "sequence_metrics":
                validate_sequences(data, label)
            if name == "episode_metrics":
                expected = {b+c for b in evaluation.TEST for c in evaluation.CONDITIONS}
                if set(data.episode) != expected or data.episode.duplicated().any():
                    raise ValueError("Missing/duplicate episode outcome: " + label)
            combined[name].append(data)
    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in combined.items()}
    for name, data in tables.items():
        data.to_csv(output / ("ablation_" + name + ".csv"), index=False)
    sequences = tables["sequence_metrics"]
    paired_tracking(sequences, seed).to_csv(output / "ablation_paired_tracking.csv", index=False)
    numeric = [c for c in sequences.select_dtypes(include=[np.number]).columns]
    seqmacro = sequences.groupby("variant")[numeric].mean().reset_index()
    seqmacro["aggregation"] = "sequence_macro_n5"
    objmacro = sequences.groupby(["variant", "object_id"])[numeric].mean().groupby("variant").mean().reset_index()
    objmacro["aggregation"] = "object_macro_n3"
    pd.concat([seqmacro, objmacro]).to_csv(output / "ablation_aggregate_metrics.csv", index=False)
    events = tables["recovery_events"]
    recovery_summary(events).to_csv(output / "ablation_recovery_summary.csv", index=False)
    keys = ["episode", "recovery_index"]
    if set(keys + ["raw_recovery_error_cm", "generated_count"]) <= set(events.columns):
        selected = keys + ["raw_recovery_error_cm", "generated_count"]
        full = events[events.variant == "full"][selected]
        audits = []
        for variant, group in events[events.variant != "full"].groupby("variant"):
            audit = full.merge(group[selected], on=keys, how="outer", suffixes=("_full", "_control"),
                               indicator=True, validate="one_to_one")
            audit["control"] = variant
            audit["raw_error_delta_cm"] = audit.raw_recovery_error_cm_full - audit.raw_recovery_error_cm_control
            audits.append(audit)
        if audits:
            pd.concat(audits).to_csv(output / "ablation_raw_recovery_pair_audit.csv", index=False)
    for path, digest in inputs.items():
        if evaluation.sha(path) != digest:
            raise ValueError("Result changed during summary: " + path)
    evaluation.dump(output / "summary_input_hashes.json", inputs)
