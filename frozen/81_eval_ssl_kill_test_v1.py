#!/usr/bin/env python3
"""Single frozen test inference pass for the 36-run SSL kill experiment.

This evaluator is authorized only by the exact automatic unlock artifact.  It
performs no model selection and prints no comparative metric before all 36
checkpoints have been evaluated.  Outputs are patient/slice-level raw tables;
the separately frozen analyzer applies the preregistered statistics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import torch


EXPECTED_TRAINER_SHA256 = "25926c413f9a1a329a220ebe4e798989b277bd9d2a1ed6a01651f7b43b05dd93"
EXPECTED_CONFIG_SHA256 = "b6a2ddd257b2d77eda61ea65f3529a83a5018344f81b17e0585427fa547c9658"
EXPECTED_MATRIX_SHA256 = "57f233b4984c2e70cf40367854900540a4a78193263766dbd7ff426ddb7552e1"
EXPECTED_FORMAL_AUDIT_SHA256 = "b755cd962a18eed0eeb7a2ccf4ca7e1a0be3692d04b5f9d1c53d73425dcc9890"
EXPECTED_UNLOCK_SHA256 = "f242719c3ebb275c0ed9e740cbf4d97f4789e37d74fb9b9b88be9fed39505210"
METHODS = ("SPARSE_FWD", "DENSE_SUP", "DENSE_MT", "DENSE_BCP")
DATA_SEEDS = (3, 4, 5)
TRAIN_SEEDS = (3, 4, 5)
THRESHOLD = 0.5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_once_or_same(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_text(encoding="utf-8") == text, f"refusing to overwrite: {path}")
        return
    path.write_text(text, encoding="utf-8")


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    require(not path.exists(), f"refusing to overwrite test output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    os.replace(temporary, path)


def resolve_under(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f"{label} escapes test root")
    require(path.is_file(), f"{label} missing: {path}")
    return path


def import_trainer(path: Path):
    spec = importlib.util.spec_from_file_location("ssl_frozen_trainer", path)
    require(spec is not None and spec.loader is not None, "cannot import frozen trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dice_binary(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction.astype(bool)
    gt = target.astype(bool)
    denominator = int(pred.sum()) + int(gt.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(pred, gt).sum() / denominator)


def checkpoint_map(formal_audit: dict[str, Any]) -> dict[tuple[int, int, str], dict[str, Any]]:
    runs = formal_audit.get("runs", [])
    require(len(runs) == 36, "formal audit does not contain 36 runs")
    result = {(int(r["data_seed"]), int(r["train_seed"]), r["method"]): r for r in runs}
    expected = {(d, t, m) for d in DATA_SEEDS for t in TRAIN_SEEDS for m in METHODS}
    require(set(result) == expected, "formal audit cell set mismatch")
    return result


def audit_existing(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    manifest = load_json(manifest_path)
    require(manifest.get("status") == "COMPLETE", "existing evaluation manifest is incomplete")
    require(manifest.get("evaluator_sha256") == sha256_file(Path(__file__).resolve()), "existing evaluator SHA mismatch")
    for name, item in manifest.get("outputs", {}).items():
        path = Path(item["path"])
        require(path.is_file(), f"existing output missing: {name}")
        require(sha256_file(path) == item["sha256"], f"existing output SHA mismatch: {name}")
    log("REUSE: complete frozen test evaluation already exists")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mroot", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--matrix-csv", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--formal-audit", type=Path, required=True)
    parser.add_argument("--unlock", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    mroot = args.mroot.resolve()
    test_root = (mroot / "test").resolve()
    data_dir = args.data_dir.resolve()
    formal_root = args.formal_root.resolve()
    matrix_path = args.matrix_csv.resolve()
    config_path = args.frozen_config.resolve()
    audit_path = args.formal_audit.resolve()
    unlock_path = args.unlock.resolve()
    trainer_path = args.trainer.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    final_manifest_path = out_dir / "ssl_test_evaluation_manifest_v1.json"
    if audit_existing(final_manifest_path):
        return

    exact_hashes = {
        trainer_path: EXPECTED_TRAINER_SHA256,
        config_path: EXPECTED_CONFIG_SHA256,
        matrix_path: EXPECTED_MATRIX_SHA256,
        audit_path: EXPECTED_FORMAL_AUDIT_SHA256,
        unlock_path: EXPECTED_UNLOCK_SHA256,
    }
    for path, expected in exact_hashes.items():
        require(path.is_file() and sha256_file(path) == expected, f"frozen prerequisite SHA mismatch: {path}")

    config = load_json(config_path)
    audit = load_json(audit_path)
    unlock = load_json(unlock_path)
    require(audit.get("status") == "PASS" and audit.get("formal_run_count") == 36, "formal audit invalid")
    require(audit.get("test_evaluator_invoked") is False and audit.get("test_data_opened") is False, "formal audit reports prior test access")
    require(unlock.get("status") == "PASS" and unlock.get("test_evaluation_allowed") is True, "test unlock invalid")
    require(unlock.get("test_evaluator_invoked") is False, "unlock reports prior test evaluator")
    require(unlock.get("formal_matrix_audit_sha256") == EXPECTED_FORMAL_AUDIT_SHA256, "unlock/audit binding mismatch")
    require(unlock.get("frozen_config_sha256") == EXPECTED_CONFIG_SHA256, "unlock/config binding mismatch")
    require(config.get("optimizer_gate", {}).get("selected_optimizer_key") == "adam", "frozen optimizer mismatch")

    evaluator_sha = sha256_file(Path(__file__).resolve())
    started = {
        "schema_version": "ssl_test_evaluation_started_v1",
        "state": "STARTED",
        "test_access_authorized": True,
        "test_evaluator_invoked": True,
        "evaluator_sha256": evaluator_sha,
        "trainer_sha256": EXPECTED_TRAINER_SHA256,
        "frozen_config_sha256": EXPECTED_CONFIG_SHA256,
        "matrix_sha256": EXPECTED_MATRIX_SHA256,
        "formal_audit_sha256": EXPECTED_FORMAL_AUDIT_SHA256,
        "unlock_sha256": EXPECTED_UNLOCK_SHA256,
        "threshold": THRESHOLD,
        "postprocessing": "none",
        "volume_rule": "sum foreground pixels times NIfTI affine voxel volume",
        "argv": sys.argv,
    }
    write_once_or_same(out_dir / "ssl_test_evaluation_started_v1.json", canonical_json(started))

    # First test access occurs only after the immutable start record exists.
    test_manifest_path = mroot / "manifest_test.csv"
    require(test_manifest_path.is_file(), f"test manifest missing: {test_manifest_path}")
    test_manifest_sha = sha256_file(test_manifest_path)
    frame = pd.read_csv(test_manifest_path)
    required_columns = {"key", "phase", "z", "img", "gt", "gt_px"}
    require(required_columns <= set(frame.columns), "test manifest required columns missing")
    frame["key"] = frame["key"].astype(str)
    frame["phase"] = frame["phase"].astype(str).str.lower()
    frame["z"] = frame["z"].astype(int)
    require(set(frame["phase"]) == {"ed", "es"}, "test phases must be ed/es")
    require(not frame.duplicated(["key", "phase", "z"]).any(), "duplicate test slice identifier")

    cached: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        image_path = resolve_under(test_root, str(row["img"]), "test image")
        gt_path = resolve_under(test_root, str(row["gt"]), "test GT")
        image_u8 = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        gt_u8 = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        require(image_u8 is not None and gt_u8 is not None, "failed to read test PNG")
        require(image_u8.shape == gt_u8.shape, "test image/GT shape mismatch")
        gt = gt_u8 > 127
        require(int(gt.sum()) == int(row["gt_px"]), "test gt_px does not match GT PNG")
        cached.append(
            {
                "meta": row,
                "image": image_u8.astype(np.float32) / 255.0,
                "gt": gt,
            }
        )
    require(cached, "empty test manifest")

    patients = sorted(frame["key"].unique().tolist())
    voxel_ml: dict[tuple[str, str], float] = {}
    for key in patients:
        for phase in ("ed", "es"):
            path = (data_dir / key / f"{key}_sax_{phase}_gt.nii.gz").resolve()
            require(path.is_relative_to(data_dir), "NIfTI path escapes data root")
            require(path.is_file(), f"test NIfTI missing: {path}")
            affine = nib.load(str(path)).affine
            value = abs(float(np.linalg.det(affine[:3, :3]))) / 1000.0
            require(math.isfinite(value) and value > 0, f"invalid voxel volume: {path}")
            voxel_ml[(key, phase)] = value

    with matrix_path.open(newline="", encoding="utf-8") as handle:
        matrix_rows = list(csv.DictReader(handle))
    require(len(matrix_rows) == 36, "matrix row count mismatch")
    audited = checkpoint_map(audit)
    trainer = import_trainer(trainer_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "CUDA GPU required for frozen test inference")

    slice_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    start_time = time.time()
    for run_index, matrix_row in enumerate(matrix_rows, 1):
        data_seed = int(matrix_row["data_seed"])
        train_seed = int(matrix_row["train_seed"])
        method = matrix_row["method"]
        cell = (data_seed, train_seed, method)
        require(cell in audited, f"cell absent from formal audit: {cell}")
        run_dir = (mroot / matrix_row["output_dir"]).resolve()
        require(run_dir.is_relative_to(formal_root), "checkpoint directory outside formal root")
        run_manifest_path = run_dir / "run_manifest.json"
        require(sha256_file(run_manifest_path) == audited[cell]["manifest_sha256"], "run manifest/audit SHA mismatch")
        run_manifest = load_json(run_manifest_path)
        checkpoint_path = Path(run_manifest["checkpoint"]).resolve()
        require(checkpoint_path.is_relative_to(run_dir), "checkpoint escapes run directory")
        require(sha256_file(checkpoint_path) == audited[cell]["checkpoint_sha256"], "checkpoint/audit SHA mismatch")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        require(checkpoint.get("method") == method, "checkpoint method mismatch")
        require(checkpoint.get("data_seed") == data_seed and checkpoint.get("train_seed") == train_seed, "checkpoint seed mismatch")
        model = trainer.UNet(checkpoint.get("encoder", "resnet34"), 1, 1, pretrained=False).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        pred_pixels: dict[tuple[str, str], int] = {(k, p): 0 for k in patients for p in ("ed", "es")}
        gt_pixels: dict[tuple[str, str], int] = {(k, p): 0 for k in patients for p in ("ed", "es")}
        with torch.inference_mode():
            for item in cached:
                meta = item["meta"]
                image = torch.from_numpy(item["image"])[None, None].to(device)
                prediction = torch.sigmoid(model(image))[0, 0].cpu().numpy() > THRESHOLD
                gt = item["gt"]
                key = str(meta["key"])
                phase = str(meta["phase"])
                pred_count = int(prediction.sum())
                gt_count = int(gt.sum())
                pred_pixels[(key, phase)] += pred_count
                gt_pixels[(key, phase)] += gt_count
                result = {
                    "data_seed": data_seed,
                    "train_seed": train_seed,
                    "method": method,
                    "key": key,
                    "phase": phase,
                    "z": int(meta["z"]),
                    "gt_px": gt_count,
                    "pred_px": pred_count,
                    "gt_present": int(gt_count > 0),
                    "pred_present": int(pred_count > 0),
                    "dice": dice_binary(prediction, gt),
                }
                for optional in ("dist", "in_lvspan", "is_anchor"):
                    if optional in meta:
                        result[optional] = meta[optional]
                slice_rows.append(result)

        for key in patients:
            gt_volumes = {
                phase: gt_pixels[(key, phase)] * voxel_ml[(key, phase)] for phase in ("ed", "es")
            }
            pred_volumes = {
                phase: pred_pixels[(key, phase)] * voxel_ml[(key, phase)] for phase in ("ed", "es")
            }
            require(gt_volumes["ed"] > 1e-12, f"zero GT EDV: {key}")
            ef_gt = 100.0 * (gt_volumes["ed"] - gt_volumes["es"]) / gt_volumes["ed"]
            ef_pred = (
                100.0 * (pred_volumes["ed"] - pred_volumes["es"]) / pred_volumes["ed"]
                if pred_volumes["ed"] > 1e-12
                else float("nan")
            )
            case_rows.append(
                {
                    "data_seed": data_seed,
                    "train_seed": train_seed,
                    "method": method,
                    "key": key,
                    "edv_gt": gt_volumes["ed"],
                    "esv_gt": gt_volumes["es"],
                    "ef_gt": ef_gt,
                    "edv_pred": pred_volumes["ed"],
                    "esv_pred": pred_volumes["es"],
                    "ef_pred": ef_pred,
                    "edv_signed_error": pred_volumes["ed"] - gt_volumes["ed"],
                    "edv_absolute_error": abs(pred_volumes["ed"] - gt_volumes["ed"]),
                    "esv_signed_error": pred_volumes["es"] - gt_volumes["es"],
                    "esv_absolute_error": abs(pred_volumes["es"] - gt_volumes["es"]),
                    "ef_signed_error": ef_pred - ef_gt,
                    "ef_absolute_error": abs(ef_pred - ef_gt),
                }
            )

        del model, checkpoint
        torch.cuda.empty_cache()
        log(f"EVAL [{run_index:02d}/36] complete: ds{data_seed}/ts{train_seed}/{method}")

    slice_frame = pd.DataFrame(slice_rows)
    case_frame = pd.DataFrame(case_rows)
    expected_slice_rows = 36 * len(frame)
    expected_case_rows = 36 * len(patients)
    require(len(slice_frame) == expected_slice_rows, "test slice output row count mismatch")
    require(len(case_frame) == expected_case_rows, "test case output row count mismatch")
    require(np.isfinite(case_frame[["edv_gt", "esv_gt", "ef_gt", "edv_pred", "esv_pred", "ef_pred"]].to_numpy()).all(), "non-finite case metric")

    slice_path = out_dir / "ssl_test_per_slice_long_v1.csv"
    case_path = out_dir / "ssl_test_per_case_long_v1.csv"
    write_csv_atomic(slice_frame, slice_path)
    write_csv_atomic(case_frame, case_path)
    outputs = {
        "per_slice": {"path": str(slice_path), "sha256": sha256_file(slice_path), "rows": len(slice_frame)},
        "per_case": {"path": str(case_path), "sha256": sha256_file(case_path), "rows": len(case_frame)},
    }
    manifest = {
        "status": "COMPLETE",
        "schema_version": "ssl_test_evaluation_manifest_v1",
        "test_evaluator_invoked": True,
        "test_data_opened": True,
        "model_selection_using_test": False,
        "evaluator_sha256": evaluator_sha,
        "trainer_sha256": EXPECTED_TRAINER_SHA256,
        "frozen_config_sha256": EXPECTED_CONFIG_SHA256,
        "matrix_sha256": EXPECTED_MATRIX_SHA256,
        "formal_audit_sha256": EXPECTED_FORMAL_AUDIT_SHA256,
        "unlock_sha256": EXPECTED_UNLOCK_SHA256,
        "test_manifest_path": str(test_manifest_path),
        "test_manifest_sha256": test_manifest_sha,
        "test_patient_keys": patients,
        "test_patient_keys_sha256": sha256_json(patients),
        "test_patient_count": len(patients),
        "test_slice_count": len(frame),
        "threshold": THRESHOLD,
        "postprocessing": "none",
        "numeric_precision": "unrounded float64 volumes and EF; CSV written with 17 significant digits",
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "wall_seconds": time.time() - start_time,
        "outputs": outputs,
    }
    write_once_or_same(final_manifest_path, canonical_json(manifest))
    log("PASS: frozen test inference complete for 36/36 checkpoints")
    log(f"Evaluation manifest SHA256: {sha256_file(final_manifest_path)}")
    log("NO TEST-BASED MODEL OR HYPERPARAMETER SELECTION PERFORMED")


if __name__ == "__main__":
    main()
