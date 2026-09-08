#!/usr/bin/env python3
"""Frozen training-unit crossed-bootstrap analysis for M&Ms-2 v3."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

import mnms2_scope_control_v3_common as common
from mnms2_scope_control_v3_common import (
    ARMS,
    PROTOCOL_ID,
    atomic_csv,
    atomic_json,
    load_config,
    sha256_file,
    verify_receipt,
)


EXTERNAL_DOMAINS = ("Philips", "GE")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("mnms2_scope_control_config_v3.json"),
    )
    parser.add_argument("--per-case", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def verify_evaluation(
    manifest_path: Path, case_path: Path
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != "LOCKED"
    ):
        raise RuntimeError("Evaluation manifest is not LOCKED")
    declared = manifest.get("outputs", {}).get("cases", {})
    if declared.get("sha256") != sha256_file(case_path):
        raise RuntimeError("Locked per-case table SHA256 mismatch")
    return manifest


def check_case_table(
    frame: pd.DataFrame, config: dict[str, Any]
) -> None:
    required = {
        "protocol_id",
        "unit_id",
        "block_key_sha256",
        "seed",
        "arm",
        "domain",
        "key",
        "ef_gt",
        "ef_pred",
        "ef_abs_error",
        "ef_nonfinite",
        "ef_invalid",
        "ae_gt20",
        "ae_gt40",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Locked case table missing columns: {missing}")
    if set(frame["protocol_id"].unique()) != {PROTOCOL_ID}:
        raise RuntimeError("Protocol ID mismatch in locked case table")
    expected_units = set(config["data"]["formal_units"])
    if set(frame["unit_id"].astype(str).unique()) != expected_units:
        raise RuntimeError("Formal unit set mismatch")
    if set(frame["arm"].astype(str).unique()) != set(ARMS):
        raise RuntimeError("Arm set mismatch")
    expected_domains = set(config["external_evaluation"]["domains"])
    if set(frame["domain"].astype(str).unique()) != expected_domains:
        raise RuntimeError("Domain set mismatch")
    identity = ["unit_id", "arm", "domain", "key"]
    if frame.duplicated(identity).any():
        raise RuntimeError("Duplicate model-patient rows in locked case table")
    expected_conditions = {
        (unit, arm, domain)
        for unit in expected_units
        for arm in ARMS
        for domain in expected_domains
    }
    observed = {
        (str(row.unit_id), str(row.arm), str(row.domain))
        for row in frame[
            ["unit_id", "arm", "domain"]
        ].drop_duplicates().itertuples()
    }
    if observed != expected_conditions:
        raise RuntimeError("Locked table does not contain the full 20 x 3 design")
    for unit, group in frame.groupby("unit_id", sort=True):
        if group["seed"].nunique() != 1:
            raise RuntimeError(f"Unit {unit} has inconsistent model seeds")
        if group["block_key_sha256"].nunique() != 1:
            raise RuntimeError(f"Unit {unit} has inconsistent block hashes")
    expected_counts = {
        "SIEMENS_val": int(config["data"]["validation_patients"]),
        **{
            name: int(value)
            for name, value in config["data"]["expected_external_patients"].items()
        },
    }
    observed_counts = (
        frame.groupby(["unit_id", "arm", "domain"])["key"].nunique()
    )
    for (unit, arm, domain), count in observed_counts.items():
        if int(count) != expected_counts[str(domain)]:
            raise RuntimeError(
                f"Patient count mismatch for {unit}/{arm}/{domain}: "
                f"{count} != {expected_counts[str(domain)]}"
            )


def build_delta_arrays(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[dict[str, np.ndarray], pd.DataFrame, list[str]]:
    units = list(map(str, config["data"]["formal_units"]))
    external = frame[frame["domain"].isin(EXTERNAL_DOMAINS)].copy()
    long_parts: list[pd.DataFrame] = []
    arrays: dict[str, np.ndarray] = {}
    for domain in EXTERNAL_DOMAINS:
        subset = external[external["domain"] == domain]
        wide = subset.pivot(
            index="key", columns=["unit_id", "arm"], values="ef_abs_error"
        )
        expected_columns = {(unit, arm) for unit in units for arm in ARMS}
        if set(wide.columns) != expected_columns:
            raise RuntimeError(f"Incomplete paired columns in {domain}")
        wide = wide.sort_index()
        values = np.vstack(
            [
                wide[(unit, "B")].to_numpy(dtype=float)
                - wide[(unit, "Cneg")].to_numpy(dtype=float)
                for unit in units
            ]
        )
        arrays[domain] = values
        long = pd.DataFrame(
            values.T,
            columns=[f"delta_{unit}" for unit in units],
        )
        long.insert(0, "key", wide.index.astype(str))
        long.insert(0, "domain", domain)
        long["delta_mean_units"] = np.mean(values, axis=0)
        long_parts.append(long)
    return arrays, pd.concat(long_parts, ignore_index=True), units


def crossed_bootstrap(
    arrays: dict[str, np.ndarray],
    replicates: int,
    seed: int,
    domain_subset: tuple[str, ...] = EXTERNAL_DOMAINS,
) -> np.ndarray:
    units = next(iter(arrays.values())).shape[0]
    if any(array.shape[0] != units for array in arrays.values()):
        raise RuntimeError("Unit dimension differs between domains")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled_units = rng.integers(0, units, units)
        sampled_domains: list[np.ndarray] = []
        for domain in domain_subset:
            values = arrays[domain]
            sampled_patients = rng.integers(
                0, values.shape[1], values.shape[1]
            )
            sampled_domains.append(
                values[np.ix_(sampled_units, sampled_patients)]
            )
        draws[index] = np.concatenate(sampled_domains, axis=1).mean()
    return draws


def unit_patient_bootstrap(
    arrays: dict[str, np.ndarray],
    unit_index: int,
    replicates: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = []
        for domain in EXTERNAL_DOMAINS:
            values = arrays[domain][unit_index]
            patient_indices = rng.integers(0, len(values), len(values))
            sampled.append(values[patient_indices])
        draws[index] = np.concatenate(sampled).mean()
    return draws


def fixed_ensemble_patient_bootstrap(
    arrays: dict[str, np.ndarray], replicates: int, seed: int
) -> tuple[float, np.ndarray]:
    patient_values = {
        domain: np.mean(values, axis=0) for domain, values in arrays.items()
    }
    estimate = float(
        np.concatenate([patient_values[domain] for domain in EXTERNAL_DOMAINS]).mean()
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = []
        for domain in EXTERNAL_DOMAINS:
            values = patient_values[domain]
            patient_indices = rng.integers(0, len(values), len(values))
            sampled.append(values[patient_indices])
        draws[index] = np.concatenate(sampled).mean()
    return estimate, draws


def unit_only_bca(
    unit_values: np.ndarray, replicates: int, seed: int
) -> dict[str, float]:
    values = np.asarray(unit_values, dtype=float)
    estimate = float(values.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    draws = values[indices].mean(axis=1)
    probability_less = (float((draws < estimate).sum()) + 0.5) / (
        replicates + 1.0
    )
    z0 = float(norm.ppf(probability_less))
    jackknife = np.asarray(
        [np.delete(values, index).mean() for index in range(len(values))],
        dtype=float,
    )
    jackknife_mean = float(jackknife.mean())
    differences = jackknife_mean - jackknife
    denominator = 6.0 * float(np.sum(differences**2) ** 1.5)
    acceleration = (
        float(np.sum(differences**3) / denominator)
        if denominator > 0
        else 0.0
    )

    def adjusted_probability(alpha: float) -> float:
        z_alpha = float(norm.ppf(alpha))
        denominator_local = 1.0 - acceleration * (z0 + z_alpha)
        if abs(denominator_local) < 1e-12:
            return alpha
        return float(
            norm.cdf(z0 + (z0 + z_alpha) / denominator_local)
        )

    low_probability = min(max(adjusted_probability(0.025), 0.0), 1.0)
    high_probability = min(max(adjusted_probability(0.975), 0.0), 1.0)
    low_probability, high_probability = sorted(
        (low_probability, high_probability)
    )
    low, high = np.quantile(draws, [low_probability, high_probability])
    return {
        "estimate": estimate,
        "ci95_bca_low": float(low),
        "ci95_bca_high": float(high),
        "bias_correction_z0": z0,
        "jackknife_acceleration": acceleration,
        "bootstrap_replicates": replicates,
    }


def unit_table(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    units: list[str],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    external = frame[frame["domain"].isin(EXTERNAL_DOMAINS)]
    rows: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        unit_frame = external[external["unit_id"] == unit]
        pooled_delta = np.concatenate(
            [arrays[domain][index] for domain in EXTERNAL_DOMAINS]
        )
        finite = bool(np.isfinite(pooled_delta).all())
        if finite:
            draws = unit_patient_bootstrap(
                arrays, index, replicates, seed + index
            )
            ci_low, ci_high = percentile_interval(draws)
        else:
            ci_low = ci_high = float("nan")
        arm_mae: dict[str, float] = {}
        for arm in ARMS:
            values = unit_frame.loc[
                unit_frame["arm"] == arm, "ef_abs_error"
            ].to_numpy(dtype=float)
            arm_mae[arm] = (
                float(np.mean(values))
                if np.isfinite(values).all()
                else float("nan")
            )
        rows.append(
            {
                "unit_id": unit,
                "block_key_sha256": str(
                    unit_frame["block_key_sha256"].iloc[0]
                ),
                "seed": int(unit_frame["seed"].iloc[0]),
                "external_patients": int(len(pooled_delta)),
                "B_ef_mae": arm_mae["B"],
                "Cneg_ef_mae": arm_mae["Cneg"],
                "D_unit_B_minus_Cneg": (
                    float(np.mean(pooled_delta)) if finite else float("nan")
                ),
                "ci95_patient_low": ci_low,
                "ci95_patient_high": ci_high,
                "favors_B": (
                    bool(np.mean(pooled_delta) < 0.0) if finite else None
                ),
            }
        )
    return pd.DataFrame(rows)


def tail_risk_table(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    pooled = work[work["domain"].isin(EXTERNAL_DOMAINS)].copy()
    pooled["domain"] = "External_pooled"
    work = pd.concat([work, pooled], ignore_index=True)
    rows: list[dict[str, Any]] = []
    grouping = ["unit_id", "arm", "domain"]
    for identity, group in work.groupby(grouping, sort=True):
        prediction = group["ef_pred"].to_numpy(dtype=float)
        absolute_error = group["ef_abs_error"].to_numpy(dtype=float)
        finite_prediction = np.isfinite(prediction)
        finite_error = np.isfinite(absolute_error)
        invalid = (
            (~finite_prediction)
            | (prediction < 0.0)
            | (prediction > 100.0)
        )
        ae_gt20 = finite_error & (absolute_error > 20.0)
        ae_gt40 = finite_error & (absolute_error > 40.0)
        rows.append(
            {
                "unit_id": identity[0],
                "arm": identity[1],
                "domain": identity[2],
                "prediction_instances": int(len(group)),
                "nonfinite_ef_n": int((~finite_prediction).sum()),
                "invalid_ef_n": int(invalid.sum()),
                "invalid_ef_pct": float(invalid.mean() * 100.0),
                "ae_gt20_n": int(ae_gt20.sum()),
                "ae_gt20_pct": float(ae_gt20.mean() * 100.0),
                "ae_gt40_n": int(ae_gt40.sum()),
                "ae_gt40_pct": float(ae_gt40.mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def forest_plot(unit_results: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    y = np.arange(len(unit_results))
    estimate = unit_results["D_unit_B_minus_Cneg"].to_numpy(dtype=float)
    low = unit_results["ci95_patient_low"].to_numpy(dtype=float)
    high = unit_results["ci95_patient_high"].to_numpy(dtype=float)
    finite = np.isfinite(estimate) & np.isfinite(low) & np.isfinite(high)
    if finite.any():
        axis.hlines(
            y[finite],
            low[finite],
            high[finite],
            color="#2455A4",
            linewidth=1.5,
        )
        axis.plot(
            estimate[finite],
            y[finite],
            "o",
            color="#2455A4",
        )
    for index in y[~finite]:
        axis.text(
            0.02,
            index,
            "not estimable",
            transform=axis.get_yaxis_transform(),
            va="center",
            color="#8B1A1A",
        )
    axis.axvline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_yticks(y)
    axis.set_yticklabels(unit_results["unit_id"].astype(str))
    axis.invert_yaxis()
    axis.set_xlabel("Unit mean Δ absolute EF error (B − Cneg), EF points")
    axis.set_ylabel("Paired training unit")
    axis.set_title("M&Ms-2 scope-control v3: unit heterogeneity")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def interpretation(
    estimate: float, ci_low: float, ci_high: float, estimable: bool
) -> tuple[str, str]:
    if not estimable:
        return (
            "NOT_ESTIMABLE",
            "At least one external EF is non-finite; no M&Ms-2 propagation-specific claim is permitted.",
        )
    if estimate < 0.0 and ci_high < 0.0:
        return (
            "EXTERNAL_SCOPE_MATCHED_SUPPORT",
            "The pooled crossed-bootstrap interval is entirely below zero, supporting an independent contribution of span-internal forward pseudo-label supervision under the selected recipe.",
        )
    if estimate < 0.0:
        return (
            "DIRECTIONAL_ONLY",
            "The point estimate favors B but the primary crossed interval includes zero; external support is directional/descriptive only.",
        )
    return (
        "NO_PROPAGATION_SPECIFIC_CLAIM",
        "The pooled estimate is zero or favors Cneg; do not claim M&Ms-2 propagation-specific benefit.",
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.out_dir.exists():
        raise RuntimeError("Analysis output directory already exists")
    this_script = Path(__file__).resolve()
    common_path = Path(common.__file__).resolve()
    receipt = verify_receipt(
        args.freeze_receipt,
        args.config,
        [this_script, common_path],
        args.audit_dir,
    )
    evaluation = verify_evaluation(
        args.evaluation_manifest, args.per_case
    )
    frame = pd.read_csv(
        args.per_case,
        dtype={"key": str, "unit_id": str, "block_key_sha256": str},
    )
    check_case_table(frame, config)
    arrays, patient_deltas, units = build_delta_arrays(frame, config)
    replicates = int(config["analysis"]["bootstrap_replicates"])
    seed = int(config["analysis"]["rng_seed"])
    stream_offsets = {
        name: int(value)
        for name, value in config["analysis"]["stream_offsets"].items()
    }
    external_raw = frame[frame["domain"].isin(EXTERNAL_DOMAINS)]
    estimable = bool(
        np.isfinite(
            external_raw[["ef_gt", "ef_pred", "ef_abs_error"]].to_numpy(
                dtype=float
            )
        ).all()
    )
    unit_results = unit_table(
        frame,
        arrays,
        units,
        replicates,
        seed + stream_offsets["unit_forest_base"],
    )
    unit_values = unit_results["D_unit_B_minus_Cneg"].to_numpy(dtype=float)
    if estimable:
        primary_draws = crossed_bootstrap(
            arrays, replicates, seed
        )
        estimate = float(unit_values.mean())
        ci_low, ci_high = percentile_interval(primary_draws)
        bca = unit_only_bca(
            unit_values,
            replicates,
            seed + stream_offsets["unit_only_bca"],
        )
        fixed_estimate, fixed_draws = fixed_ensemble_patient_bootstrap(
            arrays,
            replicates,
            seed + stream_offsets["fixed_ensemble"],
        )
        fixed_low, fixed_high = percentile_interval(fixed_draws)
        support = int((unit_values < 0.0).sum())
        ties = int((unit_values == 0.0).sum())
        non_ties = len(unit_values) - ties
        sign_p = (
            float(
                binomtest(
                    support,
                    non_ties,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
            if non_ties
            else 1.0
        )
    else:
        estimate = ci_low = ci_high = float("nan")
        bca = {
            "estimate": float("nan"),
            "ci95_bca_low": float("nan"),
            "ci95_bca_high": float("nan"),
            "bias_correction_z0": float("nan"),
            "jackknife_acceleration": float("nan"),
            "bootstrap_replicates": replicates,
        }
        fixed_estimate = fixed_low = fixed_high = float("nan")
        support = ties = non_ties = 0
        sign_p = float("nan")
    status, statement = interpretation(
        estimate, ci_low, ci_high, estimable
    )

    primary = pd.DataFrame(
        [
            {
                "estimand": "mean over ten paired training units and pooled 141 external patients",
                "delta_B_minus_Cneg": estimate,
                "ci95_crossed_percentile_low": ci_low,
                "ci95_crossed_percentile_high": ci_high,
                "formal_units": 10,
                "philips_patients": int(
                    config["data"]["expected_external_patients"]["Philips"]
                ),
                "ge_patients": int(
                    config["data"]["expected_external_patients"]["GE"]
                ),
                "support_units_favoring_B": support,
                "support_proportion": support / 10.0 if estimable else float("nan"),
                "exact_sign_test_two_sided_p": sign_p,
                "status": status,
            }
        ]
    )
    vendor_rows: list[dict[str, Any]] = []
    for domain_index, domain in enumerate(EXTERNAL_DOMAINS):
        values = arrays[domain]
        finite = bool(np.isfinite(values).all())
        if finite:
            draws = crossed_bootstrap(
                arrays,
                replicates,
                seed + stream_offsets["vendor_base"] + domain_index,
                (domain,),
            )
            low, high = percentile_interval(draws)
            point = float(values.mean())
        else:
            point = low = high = float("nan")
        vendor_rows.append(
            {
                "domain": domain,
                "patients": int(values.shape[1]),
                "delta_B_minus_Cneg": point,
                "ci95_crossed_low": low,
                "ci95_crossed_high": high,
                "status": "ESTIMABLE" if finite else "NOT_ESTIMABLE",
            }
        )
    vendors = pd.DataFrame(vendor_rows)
    sensitivities = pd.DataFrame(
        [
            {
                "analysis": "unit_only_BCa",
                "delta_B_minus_Cneg": bca["estimate"],
                "ci95_low": bca["ci95_bca_low"],
                "ci95_high": bca["ci95_bca_high"],
                "note": "Ten D[u] values only; excludes patient sampling uncertainty.",
            },
            {
                "analysis": "fixed_ensemble_patient_bootstrap",
                "delta_B_minus_Cneg": fixed_estimate,
                "ci95_low": fixed_low,
                "ci95_high": fixed_high,
                "note": "Mean d[u,p] across fixed ten units, then vendor-stratified patient bootstrap.",
            },
        ]
    )
    tails = tail_risk_table(frame)

    staging = args.out_dir.with_name(args.out_dir.name + ".STAGING")
    if staging.exists():
        raise RuntimeError(f"Analysis staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    names = {
        "primary": "mnms2_scope_control_v3_primary_results.csv",
        "units": "mnms2_scope_control_v3_unit_results.csv",
        "vendors": "mnms2_scope_control_v3_vendor_results.csv",
        "sensitivities": "mnms2_scope_control_v3_sensitivity_results.csv",
        "patient_deltas": "mnms2_scope_control_v3_patient_unit_deltas.csv",
        "tails": "mnms2_scope_control_v3_tail_risk.csv",
        "forest": "mnms2_scope_control_v3_unit_forest.png",
        "summary": "mnms2_scope_control_v3_analysis_summary.json",
        "manifest": "mnms2_scope_control_v3_analysis_manifest.json",
    }
    atomic_csv(primary, staging / names["primary"])
    atomic_csv(unit_results, staging / names["units"])
    atomic_csv(vendors, staging / names["vendors"])
    atomic_csv(sensitivities, staging / names["sensitivities"])
    atomic_csv(patient_deltas, staging / names["patient_deltas"])
    atomic_csv(tails, staging / names["tails"])
    forest_plot(unit_results, staging / names["forest"])
    summary = {
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "permitted_interpretation": statement,
        "primary_endpoint": "mean over training units and pooled external patients of absolute EF error(B) minus Cneg",
        "negative_favors": "B",
        "primary_result": primary.iloc[0].to_dict(),
        "development_conditioning": evaluation.get("selected_recipe"),
        "bootstrap": {
            "replicates": replicates,
            "rng_seed": seed,
            "stream_offsets": stream_offsets,
            "primary": "crossed percentile bootstrap: ten units plus vendor-stratified patients",
        },
        "unit_only_bca_details": bca,
        "support_count_role": "heterogeneity transparency; not a replacement for the primary crossed interval",
        "nonfinite_rule": "No clipping or deletion; any non-finite external EF makes the primary endpoint not estimable.",
        "claim_boundary": "This is external propagation-component evidence, not a second breadth-versus-depth strategy comparison.",
    }
    atomic_json(summary, staging / names["summary"])
    output_manifest = {
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETED",
        "completed_at_utc": utc_now(),
        "analysis_status": status,
        "freeze_receipt_sha256": sha256_file(args.freeze_receipt),
        "evaluation_manifest_sha256": sha256_file(
            args.evaluation_manifest
        ),
        "locked_per_case_sha256": sha256_file(args.per_case),
        "analysis_code_sha256": sha256_file(this_script),
        "common_code_sha256": sha256_file(common_path),
        "freeze_receipt_status": receipt.get("status"),
        "outputs": {
            key: {
                "path": str((args.out_dir / filename).resolve()),
                "sha256": sha256_file(staging / filename),
            }
            for key, filename in names.items()
            if key != "manifest"
        },
    }
    atomic_json(output_manifest, staging / names["manifest"])
    os.replace(staging, args.out_dir)
    final_manifest = args.out_dir / names["manifest"]
    print(
        json.dumps(
            {
                "status": status,
                "permitted_interpretation": statement,
                "analysis_summary": str(
                    (args.out_dir / names["summary"]).resolve()
                ),
                "analysis_manifest": str(final_manifest.resolve()),
                "analysis_manifest_sha256": sha256_file(final_manifest),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
