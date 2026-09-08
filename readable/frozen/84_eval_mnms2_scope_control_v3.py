#!/usr/bin/env python3
"""One-time locked evaluator for all 20 formal v3 checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import mnms2_scope_control_v3_common as common
import mnms2_scope_training_engine_v3 as engine
from mnms2_scope_control_v3_common import (
    ARMS,
    PROTOCOL_ID,
    atomic_csv,
    atomic_json,
    load_config,
    load_partition,
    sha256_file,
    vendor_mask,
    verify_receipt,
    verify_selected_recipe_artifact,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("mnms2_scope_control_config_v3.json"),
    )
    parser.add_argument(
        "--mroot", type=Path, default=Path("/root/autodl-tmp/mnms/seg_lv")
    )
    parser.add_argument(
        "--root", type=Path, default=Path("/root/autodl-tmp/mnms/MnM2")
    )
    parser.add_argument(
        "--code-dir", type=Path, default=Path("/root/autodl-tmp/sam3-main")
    )
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument(
        "--dev-matrix",
        type=Path,
        default=Path(__file__).with_name("mnms2_scope_control_dev_matrix_v3.csv"),
    )
    parser.add_argument("--selected-recipe", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument(
        "--formal-matrix",
        type=Path,
        default=Path(__file__).with_name("mnms2_scope_control_formal_matrix_v3.csv"),
    )
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--historical-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--unlock-confirmation", required=True)
    parser.add_argument("--retry-after-failure-log", type=Path)
    return parser.parse_args()


def split_domains(
    frame: pd.DataFrame,
    partition: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    validation_keys = set(
        partition.loc[partition["role"] == "validation", "key"].astype(str)
    )
    siemens = frame[vendor_mask(frame, "SIEMENS")].copy()
    domains = {
        "SIEMENS_val": siemens[siemens["key"].isin(validation_keys)].copy(),
        "Philips": frame[vendor_mask(frame, "Philips")].copy(),
        "GE": frame[vendor_mask(frame, "GE MEDICAL")].copy(),
    }
    expected = {
        "SIEMENS_val": int(config["data"]["validation_patients"]),
        **{
            name: int(value)
            for name, value in config["data"]["expected_external_patients"].items()
        },
    }
    for name, domain in domains.items():
        count = int(domain["key"].nunique())
        if count != expected[name]:
            raise RuntimeError(f"{name} patient count {count} != {expected[name]}")
    return domains


def verify_evaluation_inputs(audit_dir: Path) -> dict[str, Any]:
    summary_path = audit_dir / "input_audit_summary_v3.json"
    physical_path = audit_dir / "input_file_manifest_v3.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise RuntimeError("Input audit is not PASS")
    if (
        summary.get("outputs", {}).get(physical_path.name, {}).get("sha256")
        != sha256_file(physical_path)
    ):
        raise RuntimeError("Audited physical-input manifest changed")
    physical = pd.read_csv(physical_path)
    selected = physical[
        physical["roles"].astype(str).map(
            lambda value: (
                "external_Philips:" in value
                or "external_GE:" in value
                or "validation:" in value
            )
        )
    ].copy()
    if selected.empty:
        raise RuntimeError("No locked evaluation files in the input audit")
    for _, row in selected.iterrows():
        path = Path(str(row["absolute_path"]))
        if not path.is_file() or sha256_file(path) != str(row["sha256"]):
            raise RuntimeError(f"Locked evaluation input changed: {path}")
    return {
        "audit_summary_sha256": sha256_file(summary_path),
        "input_file_manifest_sha256": sha256_file(physical_path),
        "rehash_file_count": int(len(selected)),
    }


def verify_formal_cells(
    formal_root: Path,
    matrix_path: Path,
    receipt: dict[str, Any],
    selected_recipe_path: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    matrix = pd.read_csv(matrix_path)
    unit_seed = {
        str(unit): int(seed)
        for unit, seed in config["formal"]["unit_seed_map"].items()
    }
    expected = {
        (unit, seed, arm)
        for unit, seed in unit_seed.items()
        for arm in ARMS
    }
    observed = {
        (str(row.unit_id), int(row.seed), str(row.arm))
        for row in matrix.itertuples()
    }
    if len(matrix) != 20 or observed != expected:
        raise RuntimeError("Formal matrix differs from the frozen 20-cell design")
    selected_sha = sha256_file(selected_recipe_path)
    trainer_sha = receipt["files"].get("83_train_mnms2_scope_control_v3.py")
    cells: list[dict[str, Any]] = []
    for unit_id, seed, arm in sorted(expected):
        cell_id = f"{unit_id}_{arm}"
        manifest_path = formal_root / "run_manifests" / f"{cell_id}.json"
        checkpoint = (
            formal_root / "checkpoints" / unit_id / f"student_{arm}.pt"
        )
        if not manifest_path.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"Incomplete formal cell: {cell_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "COMPLETED"
            or manifest.get("protocol_id") != PROTOCOL_ID
        ):
            raise RuntimeError(f"Formal manifest is not COMPLETED: {cell_id}")
        declared = manifest.get("cell", {})
        identity = (
            str(declared.get("unit_id")),
            int(declared.get("seed", -1)),
            str(declared.get("arm")),
        )
        if identity != (unit_id, seed, arm):
            raise RuntimeError(f"Formal manifest identity mismatch: {cell_id}")
        checkpoint_sha = sha256_file(checkpoint)
        if manifest.get("checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError(f"Formal checkpoint hash mismatch: {cell_id}")
        if (
            manifest.get("selected_recipe", {}).get("sha256")
            != selected_sha
        ):
            raise RuntimeError(f"Selected-recipe hash mismatch: {cell_id}")
        if (
            manifest.get("code_sha256", {}).get(
                "83_train_mnms2_scope_control_v3.py"
            )
            != trainer_sha
        ):
            raise RuntimeError(f"Formal trainer hash mismatch: {cell_id}")
        if not bool(manifest.get("checkpoint_reload_match", False)):
            raise RuntimeError(f"Checkpoint reload gate failed: {cell_id}")
        cells.append(
            {
                "unit_id": unit_id,
                "seed": seed,
                "arm": arm,
                "cell_id": cell_id,
                "block_key_sha256": str(manifest["block_key_sha256"]),
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_sha,
                "run_manifest": manifest_path,
                "run_manifest_sha256": sha256_file(manifest_path),
                "best_val_dice": float(manifest["best_val_dice"]),
                "hit_step_cap": bool(manifest["hit_step_cap"]),
            }
        )
    return cells


def load_model(
    checkpoint: Path, trainer: Any, device: str
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = engine.torch_load_trusted(checkpoint)
    model = trainer.UNet(
        payload.get("encoder", "resnet34"),
        1,
        1,
        pretrained=False,
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def summarize_cases(cases: pd.DataFrame, evaluator: Any) -> dict[str, Any]:
    finite = np.isfinite(cases["ef_gt"].to_numpy(dtype=float)) & np.isfinite(
        cases["ef_pred"].to_numpy(dtype=float)
    )
    ground_truth = cases.loc[finite, "ef_gt"].to_numpy(dtype=float)
    predicted = cases.loc[finite, "ef_pred"].to_numpy(dtype=float)
    error = predicted - ground_truth
    icc = (
        evaluator.icc_2_1(ground_truth, predicted)
        if len(ground_truth) > 3
        else float("nan")
    )
    correlation = (
        np.corrcoef(ground_truth, predicted)[0, 1]
        if len(ground_truth) > 3
        else float("nan")
    )
    return {
        "patients": int(len(cases)),
        "finite_ef_patients": int(finite.sum()),
        "ef_mae": float(np.abs(error).mean()) if len(error) else float("nan"),
        "ef_bias": float(error.mean()) if len(error) else float("nan"),
        "ef_icc": float(icc),
        "ef_r": float(correlation),
        "invalid_ef_n": int(cases["ef_invalid"].sum()),
        "ae_gt20_n": int(cases["ae_gt20"].sum()),
        "ae_gt40_n": int(cases["ae_gt40"].sum()),
    }


@torch.no_grad()
def evaluate_condition(
    model: torch.nn.Module,
    domain: pd.DataFrame,
    root_split: Path,
    data_root: Path,
    evaluator: Any,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pred_px, dice_values = evaluator.predict_all(
        model, domain, str(root_split), device
    )
    dice_values = np.asarray(dice_values, dtype=float)
    if len(dice_values) != len(domain):
        raise RuntimeError("Evaluator returned a misaligned Dice vector")
    cases, _ = engine.validation_ef_table(
        domain, pred_px, root_split, data_root
    )
    cases["ef_nonfinite"] = ~np.isfinite(cases["ef_pred"].to_numpy(dtype=float))
    cases["ef_invalid"] = (
        cases["ef_nonfinite"]
        | (cases["ef_pred"] < 0.0)
        | (cases["ef_pred"] > 100.0)
    )
    cases["ae_gt20"] = np.isfinite(cases["ef_abs_error"]) & (
        cases["ef_abs_error"] > 20.0
    )
    cases["ae_gt40"] = np.isfinite(cases["ef_abs_error"]) & (
        cases["ef_abs_error"] > 40.0
    )
    slice_columns = [
        column
        for column in (
            "key",
            "phase",
            "z",
            "vendor",
            "disease",
            "gt_px",
            "in_lvspan",
            "dist",
            "is_anchor",
        )
        if column in domain.columns
    ]
    slices = domain[slice_columns].copy()
    slices["dice"] = dice_values
    slices["pred_px"] = [
        int(pred_px.get((str(row.key), str(row.phase), int(row.z)), 0))
        for row in domain.itertuples()
    ]
    present = slices[slices["gt_px"] > 0] if "gt_px" in slices else slices
    dice_summary = {
        "dice_lv_present": (
            float(present["dice"].mean()) if len(present) else float("nan")
        ),
        "n_present_slices": int(len(present)),
    }
    return cases, slices, dice_summary


def historical_parity(
    checkpoint: Path,
    domains: dict[str, pd.DataFrame],
    root_split: Path,
    data_root: Path,
    trainer: Any,
    evaluator: Any,
    config: dict[str, Any],
) -> pd.DataFrame:
    model, _ = load_model(checkpoint, trainer, "cuda")
    rows: list[dict[str, Any]] = []
    references = config["external_evaluation"]["historical_parity_ef_mae"]
    tolerance = float(
        config["external_evaluation"]["historical_parity_tolerance"]
    )
    for domain_name in config["external_evaluation"]["domains"]:
        cases, _, dice_summary = evaluate_condition(
            model,
            domains[domain_name],
            root_split,
            data_root,
            evaluator,
            "cuda",
        )
        observed = summarize_cases(cases, evaluator)["ef_mae"]
        reference = float(references[domain_name])
        rows.append(
            {
                "domain": domain_name,
                "observed_ef_mae": observed,
                "reference_ef_mae": reference,
                "difference": observed - reference,
                "absolute_tolerance": tolerance,
                "pass": (
                    math.isfinite(observed)
                    and abs(observed - reference) <= tolerance
                ),
                **dice_summary,
            }
        )
    del model
    torch.cuda.empty_cache()
    result = pd.DataFrame(rows)
    if not bool(result["pass"].all()):
        raise RuntimeError("Historical evaluator parity failed; test remains locked")
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    unlock = str(config["external_evaluation"]["unlock_phrase"])
    if args.unlock_confirmation != unlock:
        raise RuntimeError(f"One-time test unlock requires exact phrase: {unlock}")
    if not torch.cuda.is_available():
        raise RuntimeError("Locked evaluation requires CUDA")
    if args.out_dir.exists():
        raise RuntimeError("Locked evaluation output directory already exists")
    staging = args.out_dir.with_name(args.out_dir.name + ".STAGING_DO_NOT_INSPECT")
    marker = args.out_dir.with_name(args.out_dir.name + ".TEST_UNLOCK_IN_PROGRESS.json")
    if staging.exists():
        raise RuntimeError(
            f"An interrupted staging directory exists and must be archived: {staging}"
        )
    previous_attempt = None
    if marker.exists():
        if args.retry_after_failure_log is None:
            raise RuntimeError("An interrupted unlock marker exists")
        if not args.retry_after_failure_log.is_file():
            raise FileNotFoundError(args.retry_after_failure_log)
        previous_attempt = {
            "marker_before_retry": json.loads(marker.read_text(encoding="utf-8")),
            "failure_log": str(args.retry_after_failure_log.resolve()),
            "failure_log_sha256": sha256_file(args.retry_after_failure_log),
        }
    elif args.retry_after_failure_log is not None:
        raise RuntimeError("Retry log supplied without an interrupted unlock marker")

    this_script = Path(__file__).resolve()
    common_path = Path(common.__file__).resolve()
    engine_path = Path(engine.__file__).resolve()
    receipt = verify_receipt(
        args.freeze_receipt,
        args.config,
        [
            this_script,
            common_path,
            engine_path,
            args.dev_matrix,
            args.formal_matrix,
        ],
        args.audit_dir,
    )
    receipt_sha = sha256_file(args.freeze_receipt)
    selected, _ = verify_selected_recipe_artifact(
        args.selected_recipe,
        args.development_root,
        args.dev_matrix,
        config,
        receipt_sha,
    )
    audit = verify_evaluation_inputs(args.audit_dir)
    cells = verify_formal_cells(
        args.formal_root,
        args.formal_matrix,
        receipt,
        args.selected_recipe,
        config,
    )
    trainer, evaluator, primitive_sources = engine.load_primitives(
        args.code_dir, config, need_evaluator=True
    )
    frame = engine.load_normalized_manifest(args.mroot, config)
    partition = load_partition(args.audit_dir)
    domains = split_domains(frame, partition, config)
    root_split = args.mroot / "all"
    atomic_json(
        {
            "protocol_id": PROTOCOL_ID,
            "status": "IN_PROGRESS_DO_NOT_INSPECT_PARTIAL_OUTPUTS",
            "started_at_utc": utc_now(),
            "argv": sys.argv,
            "freeze_receipt_sha256": receipt_sha,
            "formal_matrix_sha256": sha256_file(args.formal_matrix),
            "selected_recipe_sha256": sha256_file(args.selected_recipe),
            "previous_attempt": previous_attempt,
        },
        marker,
    )
    staging.mkdir(parents=True)
    try:
        parity = historical_parity(
            args.historical_checkpoint,
            domains,
            root_split,
            args.root,
            trainer,
            evaluator,
            config,
        )
        print("[gate] historical evaluator parity PASS", flush=True)
        case_parts: list[pd.DataFrame] = []
        slice_parts: list[pd.DataFrame] = []
        aggregate_rows: list[dict[str, Any]] = []
        domain_order = tuple(config["external_evaluation"]["domains"])
        for index, cell in enumerate(cells, start=1):
            model, payload = load_model(cell["checkpoint"], trainer, "cuda")
            identity = (
                str(payload.get("unit_id")),
                int(payload.get("seed", -1)),
                str(payload.get("arm")),
            )
            expected_identity = (
                cell["unit_id"],
                cell["seed"],
                cell["arm"],
            )
            if (
                identity != expected_identity
                or payload.get("protocol_id") != PROTOCOL_ID
            ):
                raise RuntimeError(
                    f"Formal checkpoint metadata mismatch: {cell['cell_id']}"
                )
            for domain_name in domain_order:
                cases, slices, dice_summary = evaluate_condition(
                    model,
                    domains[domain_name],
                    root_split,
                    args.root,
                    evaluator,
                    "cuda",
                )
                identifiers = {
                    "protocol_id": PROTOCOL_ID,
                    "unit_id": cell["unit_id"],
                    "block_key_sha256": cell["block_key_sha256"],
                    "seed": cell["seed"],
                    "arm": cell["arm"],
                    "domain": domain_name,
                    "best_val_dice": cell["best_val_dice"],
                    "hit_step_cap": cell["hit_step_cap"],
                    "checkpoint": str(cell["checkpoint"]),
                    "checkpoint_sha256": cell["checkpoint_sha256"],
                }
                for key, value in reversed(list(identifiers.items())):
                    cases.insert(0, key, value)
                    slices.insert(0, key, value)
                aggregate_rows.append(
                    {
                        **identifiers,
                        **dice_summary,
                        **summarize_cases(cases, evaluator),
                    }
                )
                case_parts.append(cases)
                slice_parts.append(slices)
            del model
            torch.cuda.empty_cache()
            print(
                f"[unlock] evaluated frozen cell {index}/20: {cell['cell_id']}",
                flush=True,
            )

        cases = pd.concat(case_parts, ignore_index=True, sort=False)
        slices = pd.concat(slice_parts, ignore_index=True, sort=False)
        aggregate = pd.DataFrame(aggregate_rows)
        expected_patients = int(config["data"]["validation_patients"]) + sum(
            int(value)
            for value in config["data"]["expected_external_patients"].values()
        )
        if len(cases) != 20 * expected_patients or len(aggregate) != 60:
            raise RuntimeError("Locked evaluation row counts do not close")

        names = {
            "cases": "mnms2_scope_control_v3_per_case_locked.csv",
            "slices": "mnms2_scope_control_v3_per_slice_locked.csv",
            "aggregate": "mnms2_scope_control_v3_aggregate_locked.csv",
            "parity": "mnms2_scope_control_v3_evaluator_parity.csv",
            "manifest": "mnms2_scope_control_v3_evaluation_manifest.json",
        }
        atomic_csv(cases, staging / names["cases"])
        atomic_csv(slices, staging / names["slices"])
        atomic_csv(aggregate, staging / names["aggregate"])
        atomic_csv(parity, staging / names["parity"])
        evaluation_manifest = {
            "protocol_id": PROTOCOL_ID,
            "status": "LOCKED",
            "completed_at_utc": utc_now(),
            "matrix_cells": 20,
            "case_rows": int(len(cases)),
            "slice_rows": int(len(slices)),
            "aggregate_rows": int(len(aggregate)),
            "freeze_receipt_sha256": receipt_sha,
            "formal_matrix_sha256": sha256_file(args.formal_matrix),
            "selected_recipe": {
                "selection_status": selected["selection_status"],
                "recipe_id": selected["selected_recipe_id"],
                "failed_gate": selected["failed_gate"],
                "sha256": sha256_file(args.selected_recipe),
            },
            "historical_checkpoint": str(
                args.historical_checkpoint.resolve()
            ),
            "historical_checkpoint_sha256": sha256_file(
                args.historical_checkpoint
            ),
            "audit": audit,
            "code_sha256": {
                this_script.name: sha256_file(this_script),
                common_path.name: sha256_file(common_path),
                engine_path.name: sha256_file(engine_path),
                **{
                    name: sha256_file(path)
                    for name, path in primitive_sources.items()
                },
            },
            "outputs": {
                key: {
                    "path": str((args.out_dir / filename).resolve()),
                    "sha256": sha256_file(staging / filename),
                }
                for key, filename in names.items()
                if key != "manifest"
            },
        }
        atomic_json(evaluation_manifest, staging / names["manifest"])
        os.replace(staging, args.out_dir)
        marker.unlink()
        final_manifest = args.out_dir / names["manifest"]
        print(
            json.dumps(
                {
                    "status": "LOCKED",
                    "evaluation_manifest": str(final_manifest.resolve()),
                    "evaluation_manifest_sha256": sha256_file(final_manifest),
                    "per_case": str(
                        (args.out_dir / names["cases"]).resolve()
                    ),
                    "per_case_sha256": sha256_file(
                        args.out_dir / names["cases"]
                    ),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception:
        print(
            f"[locked] evaluation interrupted; archive {staging} and the failure log before any identical retry",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
