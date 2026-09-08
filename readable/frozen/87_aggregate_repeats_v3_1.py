#!/usr/bin/env python3
"""Pre-specified repeat aggregator for M&Ms-2 scope control v3.

Frozen specification: AMENDMENT_v3_1_VARIANCE_AND_REPEATS.md section 5
(sha256 dc58318fe785ccd2f923a3a3047849341779b4ccc7a0a8f746f9d9f07472861e).

Adds, modifies or replaces NO statistical procedure. Every statistic is
computed by importing the frozen 85_analyze_mnms2_scope_control_v3.py.
This module only (a) verifies each LOCKED per-case table, (b) averages
ef_abs_error / ef_pred over repeats per (unit_id, arm, domain, key), and
(c) re-runs the frozen orchestration on the averaged table.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

BUNDLE = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE))

CASES_NAME = "mnms2_scope_control_v3_per_case_locked.csv"
MANIFEST_NAME = "mnms2_scope_control_v3_evaluation_manifest.json"

AVERAGED = ("ef_pred", "ef_abs_error")
CONSTANT = ("ef_gt", "block_key_sha256", "seed", "protocol_id")
BOOLEAN_OR = ("ef_nonfinite", "ef_invalid", "ae_gt20", "ae_gt40")
IDENTITY = ["unit_id", "arm", "domain", "key"]


def load_frozen(filename: str):
    stem = filename[:-3].replace(".", "_")
    spec = importlib.util.spec_from_file_location(f"frozen_{stem}", BUNDLE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


A = load_frozen("85_analyze_mnms2_scope_control_v3.py")
common = A.common


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Average R locked evaluations, then run frozen 85 statistics.")
    p.add_argument("--config", type=Path, default=BUNDLE / "mnms2_scope_control_config_v3.json")
    p.add_argument("--evaluation-dir", type=Path, nargs="+", required=True,
                   help="One or more LOCKED evaluation output directories.")
    p.add_argument("--freeze-receipt", type=Path, required=True)
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def load_and_verify(eval_dir: Path, config: dict) -> tuple[pd.DataFrame, dict]:
    cases = eval_dir / CASES_NAME
    manifest = eval_dir / MANIFEST_NAME
    if not cases.is_file() or not manifest.is_file():
        raise RuntimeError(f"Locked evaluation is incomplete: {eval_dir}")
    declared = A.verify_evaluation(manifest, cases)
    frame = pd.read_csv(cases, dtype={"key": str, "unit_id": str, "block_key_sha256": str})
    A.check_case_table(frame, config)
    return frame, declared


def average_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Average over repeats, preserving the row order of the first locked table.

    Row order is preserved so that floating-point summation order downstream is
    identical to the frozen analysis applied to a single locked table, making
    the pre-specified R=1 identity check exact. Ordering is arm-neutral.
    """
    ref = frames[0].copy()
    if ref.duplicated(IDENTITY).any():
        raise RuntimeError("Locked table contains duplicate identity rows")
    order = ref[IDENTITY].copy()
    order["_row_order"] = np.arange(len(ref))
    aligned: list[pd.DataFrame] = []
    for index, frame in enumerate(frames):
        if frame.duplicated(IDENTITY).any():
            raise RuntimeError(f"Repeat {index} contains duplicate identity rows")
        merged = order.merge(
            frame, on=IDENTITY, how="left", validate="one_to_one", indicator=True
        )
        if not (merged["_merge"] == "both").all():
            raise RuntimeError(f"Repeat {index} does not cover the same identity set")
        if len(merged) != len(ref):
            raise RuntimeError(f"Repeat {index} changed the row count after alignment")
        merged = merged.sort_values("_row_order").reset_index(drop=True)
        for column in CONSTANT:
            if column in ref.columns and column in merged.columns:
                if not (ref[column].to_numpy() == merged[column].to_numpy()).all():
                    raise RuntimeError(
                        f"Repeat {index} disagrees on constant column {column}"
                    )
        aligned.append(merged)
    out = ref.copy()
    for column in AVERAGED:
        stack = np.vstack([f[column].to_numpy(dtype=float) for f in aligned])
        out[column] = stack.mean(axis=0)
    for column in BOOLEAN_OR:
        if column not in out.columns:
            continue
        stack = np.vstack([f[column].to_numpy() for f in aligned])
        out[column] = stack.astype(bool).any(axis=0)
    if out.duplicated(IDENTITY).any():
        raise RuntimeError("Averaged table still contains duplicate identity rows")
    return out


def repeat_dispersion(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if len(frames) < 2:
        return pd.DataFrame()
    stack = np.vstack([f.sort_values(IDENTITY).reset_index(drop=True)["ef_abs_error"].to_numpy(dtype=float)
                       for f in frames])
    base = frames[0].sort_values(IDENTITY).reset_index(drop=True)
    out = base[IDENTITY].copy()
    out["ef_abs_error_mean"] = stack.mean(axis=0)
    out["ef_abs_error_sd_over_repeats"] = stack.std(axis=0, ddof=1)
    out["ef_abs_error_range_over_repeats"] = stack.max(axis=0) - stack.min(axis=0)
    out["n_repeats"] = stack.shape[0]
    out["n_nonfinite_repeats"] = (~np.isfinite(stack)).sum(axis=0)
    return out


def main() -> None:
    args = parse_args()
    config = A.load_config(args.config)
    if args.out_dir.exists():
        raise RuntimeError("Aggregate analysis output directory already exists")

    frozen_analysis = (BUNDLE / "85_analyze_mnms2_scope_control_v3.py").resolve()
    common_path = Path(common.__file__).resolve()
    receipt = A.verify_receipt(args.freeze_receipt, args.config,
                               [frozen_analysis, common_path], args.audit_dir)

    frames, manifests = [], []
    for eval_dir in args.evaluation_dir:
        frame, declared = load_and_verify(Path(eval_dir), config)
        frames.append(frame)
        manifests.append({"dir": str(Path(eval_dir).resolve()),
                          "manifest_sha256": A.sha256_file(Path(eval_dir) / MANIFEST_NAME),
                          "cases_sha256": A.sha256_file(Path(eval_dir) / CASES_NAME),
                          "status": declared.get("status")})

    frame = average_frames(frames)
    A.check_case_table(frame, config)
    dispersion = repeat_dispersion(frames)

    arrays, patient_deltas, units = A.build_delta_arrays(frame, config)
    replicates = int(config["analysis"]["bootstrap_replicates"])
    seed = int(config["analysis"]["rng_seed"])
    offsets = {k: int(v) for k, v in config["analysis"]["stream_offsets"].items()}

    external_raw = frame[frame["domain"].isin(A.EXTERNAL_DOMAINS)]
    estimable = bool(np.isfinite(
        external_raw[["ef_gt", "ef_pred", "ef_abs_error"]].to_numpy(dtype=float)).all())

    unit_results = A.unit_table(frame, arrays, units, replicates,
                                seed + offsets["unit_forest_base"])
    unit_values = unit_results["D_unit_B_minus_Cneg"].to_numpy(dtype=float)

    if estimable:
        primary_draws = A.crossed_bootstrap(arrays, replicates, seed)
        estimate = float(unit_values.mean())
        ci_low, ci_high = A.percentile_interval(primary_draws)
        bca = A.unit_only_bca(unit_values, replicates, seed + offsets["unit_only_bca"])
        fixed_estimate, fixed_draws = A.fixed_ensemble_patient_bootstrap(
            arrays, replicates, seed + offsets["fixed_ensemble"])
        fixed_low, fixed_high = A.percentile_interval(fixed_draws)
        support = int((unit_values < 0.0).sum())
        ties = int((unit_values == 0.0).sum())
        non_ties = len(unit_values) - ties
        sign_p = float(binomtest(support, non_ties, p=0.5,
                                 alternative="two-sided").pvalue) if non_ties else 1.0
    else:
        estimate = ci_low = ci_high = float("nan")
        bca = {"estimate": float("nan"), "ci95_bca_low": float("nan"),
               "ci95_bca_high": float("nan"), "bias_correction_z0": float("nan"),
               "jackknife_acceleration": float("nan"), "bootstrap_replicates": replicates}
        fixed_estimate = fixed_low = fixed_high = float("nan")
        support = ties = non_ties = 0
        sign_p = float("nan")

    status, statement = A.interpretation(estimate, ci_low, ci_high, estimable)

    primary = pd.DataFrame([{
        "estimand": ("mean over ten paired training units and pooled external patients of "
                     f"repeat-averaged (R={len(frames)}) absolute EF error, B minus Cneg"),
        "repeats": len(frames),
        "delta_B_minus_Cneg": estimate,
        "ci95_crossed_percentile_low": ci_low,
        "ci95_crossed_percentile_high": ci_high,
        "formal_units": len(units),
        "philips_patients": int(config["data"]["expected_external_patients"]["Philips"]),
        "ge_patients": int(config["data"]["expected_external_patients"]["GE"]),
        "support_units_favoring_B": support,
        "support_proportion": support / len(units) if estimable else float("nan"),
        "exact_sign_test_two_sided_p": sign_p,
        "status": status,
    }])

    vendor_rows = []
    for index, domain in enumerate(A.EXTERNAL_DOMAINS):
        values = arrays[domain]
        finite = bool(np.isfinite(values).all())
        if finite:
            draws = A.crossed_bootstrap(arrays, replicates,
                                        seed + offsets["vendor_base"] + index, (domain,))
            low, high = A.percentile_interval(draws)
            point = float(values.mean())
        else:
            point = low = high = float("nan")
        vendor_rows.append({"domain": domain, "patients": int(values.shape[1]),
                            "delta_B_minus_Cneg": point, "ci95_crossed_low": low,
                            "ci95_crossed_high": high,
                            "status": "ESTIMABLE" if finite else "NOT_ESTIMABLE"})

    sensitivities = pd.DataFrame([
        {"analysis": "unit_only_BCa", "delta_B_minus_Cneg": bca["estimate"],
         "ci95_low": bca["ci95_bca_low"], "ci95_high": bca["ci95_bca_high"],
         "note": "Ten D[u] values only; excludes patient sampling uncertainty."},
        {"analysis": "fixed_ensemble_patient_bootstrap", "delta_B_minus_Cneg": fixed_estimate,
         "ci95_low": fixed_low, "ci95_high": fixed_high,
         "note": "Mean d[u,p] across fixed units, then vendor-stratified patient bootstrap."},
    ])

    tails_avg = A.tail_risk_table(frame)
    tails_avg.insert(0, "table", "repeat_averaged_ensemble_prediction")
    tail_parts = [tails_avg]
    for index, single in enumerate(frames):
        part = A.tail_risk_table(single)
        part.insert(0, "table", f"locked_repeat_{index:02d}")
        tail_parts.append(part)
    tails = pd.concat(tail_parts, ignore_index=True)

    staging = args.out_dir.with_name(args.out_dir.name + ".STAGING")
    if staging.exists():
        raise RuntimeError(f"Aggregate staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    P = "mnms2_scope_control_v3_R_aggregate_"
    names = {"primary": P + "primary_results.csv", "units": P + "unit_results.csv",
             "vendors": P + "vendor_results.csv", "sensitivities": P + "sensitivity_results.csv",
             "patient_deltas": P + "patient_unit_deltas.csv", "tails": P + "tail_risk.csv",
             "averaged_cases": P + "averaged_per_case.csv",
             "dispersion": P + "repeat_dispersion.csv",
             "forest": P + "unit_forest.png", "summary": P + "summary.json",
             "manifest": P + "manifest.json"}

    A.atomic_csv(primary, staging / names["primary"])
    A.atomic_csv(unit_results, staging / names["units"])
    A.atomic_csv(pd.DataFrame(vendor_rows), staging / names["vendors"])
    A.atomic_csv(sensitivities, staging / names["sensitivities"])
    A.atomic_csv(patient_deltas, staging / names["patient_deltas"])
    A.atomic_csv(tails, staging / names["tails"])
    A.atomic_csv(frame, staging / names["averaged_cases"])
    if not dispersion.empty:
        A.atomic_csv(dispersion, staging / names["dispersion"])
    A.forest_plot(unit_results, staging / names["forest"])

    summary = {
        "protocol_id": A.PROTOCOL_ID,
        "status": status,
        "permitted_interpretation": statement,
        "repeats": len(frames),
        "primary_endpoint": ("mean over training units and pooled external patients of "
                             "repeat-averaged absolute EF error, B minus Cneg"),
        "negative_favors": "B",
        "primary_result": primary.iloc[0].to_dict(),
        "source_evaluations": manifests,
        "bootstrap": {"replicates": replicates, "rng_seed": seed, "stream_offsets": offsets,
                      "primary": "crossed percentile bootstrap: units plus vendor-stratified patients"},
        "unit_only_bca_details": bca,
        "nonfinite_rule": ("No clipping or deletion. Averaging propagates non-finiteness: a cell is "
                           "non-finite if ANY repeat is non-finite, and any non-finite external EF "
                           "makes the primary endpoint not estimable."),
        "averaging_rule": {"averaged": list(AVERAGED), "constant_verified": list(CONSTANT),
                           "boolean_or": list(BOOLEAN_OR)},
        "tail_risk_note": ("Reported for the repeat-averaged table (ensemble prediction semantics) "
                           "AND separately for each locked single-run table."),
        "code_provenance": ("All statistics imported from the frozen analysis script. This "
                            "aggregator is not in the freeze receipt; the receipt is verified "
                            "against the frozen analysis and common modules it depends on."),
        "claim_boundary": ("External propagation-component evidence, not a second "
                           "breadth-versus-depth strategy comparison."),
    }
    A.atomic_json(summary, staging / names["summary"])

    manifest_out = {
        "protocol_id": A.PROTOCOL_ID, "status": "COMPLETED", "completed_at_utc": utc_now(),
        "analysis_status": status, "repeats": len(frames),
        "freeze_receipt_sha256": A.sha256_file(args.freeze_receipt),
        "freeze_receipt_status": receipt.get("status"),
        "frozen_analysis_sha256": A.sha256_file(frozen_analysis),
        "common_code_sha256": A.sha256_file(common_path),
        "aggregator_sha256": A.sha256_file(Path(__file__).resolve()),
        "source_evaluations": manifests,
        "outputs": {k: {"path": str((args.out_dir / v).resolve()),
                        "sha256": A.sha256_file(staging / v)}
                    for k, v in names.items()
                    if k != "manifest" and (staging / v).exists()},
    }
    A.atomic_json(manifest_out, staging / names["manifest"])
    os.replace(staging, args.out_dir)
    final = args.out_dir / names["manifest"]
    print(json.dumps({"status": status, "permitted_interpretation": statement,
                      "repeats": len(frames),
                      "summary": str((args.out_dir / names["summary"]).resolve()),
                      "manifest": str(final.resolve()),
                      "manifest_sha256": A.sha256_file(final)},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
