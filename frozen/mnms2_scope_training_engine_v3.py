#!/usr/bin/env python3
"""Shared CUDA training and SIEMENS-validation utilities for v3."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mnms2_scope_control_v3_common import (
    PROTOCOL_ID,
    build_label_scope,
    load_partition,
    norm_key,
    partition_block_hash,
    sha256_file,
    sha256_keys,
    sha256_records,
    vendor_mask,
)


class NonFiniteTrainingError(RuntimeError):
    """A scientific technical-failure outcome, not an infrastructure retry."""


def torch_load_trusted(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256_tensor_state(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def atomic_torch_save(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


class Fit256(Dataset):
    def __init__(self, base: Dataset, size: int = 256):
        self.base = base
        self.size = size

    def __len__(self) -> int:
        return len(self.base)

    @staticmethod
    def fit(tensor: torch.Tensor, size: int) -> torch.Tensor:
        _, height, width = tensor.shape
        if height > size:
            top = (height - size) // 2
            tensor = tensor[:, top : top + size, :]
            height = size
        if width > size:
            left = (width - size) // 2
            tensor = tensor[:, :, left : left + size]
            width = size
        pad_h, pad_w = size - height, size - width
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(
                tensor,
                (
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h // 2,
                    pad_h - pad_h // 2,
                ),
                value=0.0,
            )
        return tensor

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.base[index]
        return self.fit(image, self.size), self.fit(label, self.size)


def environment_fingerprint() -> dict[str, Any]:
    try:
        import timm

        timm_version = getattr(timm, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - diagnostic only
        timm_version = f"ERROR:{type(exc).__name__}:{exc}"
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(properties.total_memory),
        }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "opencv": cv2.__version__,
        "nibabel": nib.__version__,
        "timm": timm_version,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
    }


def load_primitives(
    code_dir: Path, config: dict[str, Any], need_evaluator: bool
) -> tuple[Any, Any | None, dict[str, Path]]:
    sources = {"74_train_seg_students.py": code_dir / "74_train_seg_students.py"}
    expected = {
        "74_train_seg_students.py": config["data"]["trainer_primitives_sha256"]
    }
    if need_evaluator:
        sources["75_eval_seg_and_ef.py"] = code_dir / "75_eval_seg_and_ef.py"
        expected["75_eval_seg_and_ef.py"] = config["data"][
            "evaluator_primitives_sha256"
        ]
    for name, path in sources.items():
        if not path.is_file() or sha256_file(path) != expected[name]:
            raise RuntimeError(f"Primitive source SHA256 mismatch: {path}")
    code_text = str(code_dir.resolve())
    if code_text not in sys.path:
        sys.path.insert(0, code_text)
    trainer = importlib.import_module("74_train_seg_students")
    evaluator = (
        importlib.import_module("75_eval_seg_and_ef") if need_evaluator else None
    )
    return trainer, evaluator, sources


def load_normalized_manifest(mroot: Path, config: dict[str, Any]) -> pd.DataFrame:
    path = mroot / "manifest_mnms2.csv"
    if not path.is_file() or sha256_file(path) != config["data"]["manifest_sha256"]:
        raise RuntimeError("manifest_mnms2.csv is absent or has changed")
    frame = pd.read_csv(path)
    frame["key"] = frame["key"].map(norm_key)
    frame["pf"] = frame["pf"].fillna("")
    frame["gt"] = frame["gt"].astype(str)
    frame["img"] = frame["img"].astype(str)
    frame["phase"] = frame["phase"].astype(str).str.lower()
    return frame


def frames_for_block(
    frame: pd.DataFrame,
    partition: pd.DataFrame,
    block_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    siemens = frame[vendor_mask(frame, "SIEMENS")].copy()
    train_keys = sorted(
        partition.loc[partition["block_id"] == block_id, "key"].astype(str)
    )
    validation_keys = sorted(
        partition.loc[partition["role"] == "validation", "key"].astype(str)
    )
    if len(train_keys) != 10 or len(validation_keys) != 43:
        raise RuntimeError("Frozen block/validation patient counts do not close")
    train = siemens[siemens["key"].isin(train_keys)].copy()
    validation = siemens[siemens["key"].isin(validation_keys)].copy()
    return train, validation, train_keys, validation_keys


def verify_audited_inputs(
    audit_dir: Path,
    block_id: str,
    arm: str,
) -> dict[str, Any]:
    summary_path = audit_dir / "input_audit_summary_v3.json"
    input_path = audit_dir / "input_file_manifest_v3.csv"
    scope_path = audit_dir / "label_scope_audit_v3.csv"
    for path in (summary_path, input_path, scope_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise RuntimeError("Input audit is not PASS")
    declared_outputs = summary.get("outputs", {})
    for path in (input_path, scope_path):
        if (
            declared_outputs.get(path.name, {}).get("sha256")
            != sha256_file(path)
        ):
            raise RuntimeError(f"Audited artifact changed: {path}")

    scope = pd.read_csv(scope_path)
    matches = scope[(scope["block_id"] == block_id) & (scope["arm"] == arm)]
    if len(matches) != 1:
        raise RuntimeError(f"Missing audited label scope for {block_id}/{arm}")
    physical = pd.read_csv(input_path)
    role_token = f"{block_id}:{arm}:"
    chosen = physical[
        physical["roles"].astype(str).map(
            lambda value: role_token in value or "validation:" in value
        )
    ].copy()
    if chosen.empty:
        raise RuntimeError("No audited physical files match this training cell")
    for _, row in chosen.iterrows():
        path = Path(str(row["absolute_path"]))
        if not path.is_file() or sha256_file(path) != str(row["sha256"]):
            raise RuntimeError(f"Physical training/validation input changed: {path}")
    return {
        "audit_summary_sha256": sha256_file(summary_path),
        "input_file_manifest_sha256": sha256_file(input_path),
        "label_scope_audit_sha256": sha256_file(scope_path),
        "rehash_file_count": int(len(chosen)),
        "audited_label_scope": matches.iloc[0].to_dict(),
    }


def prepare_training_cell(
    mroot: Path,
    audit_dir: Path,
    block_id: str,
    arm: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    frame = load_normalized_manifest(mroot, config)
    partition = load_partition(audit_dir)
    train, validation, train_keys, validation_keys = frames_for_block(
        frame, partition, block_id
    )
    pairs, scope = build_label_scope(train, arm)
    audited = verify_audited_inputs(audit_dir, block_id, arm)
    expected_rows = int(audited["audited_label_scope"]["label_rows"])
    if len(pairs) != expected_rows:
        raise RuntimeError(
            f"Label-row count {len(pairs)} differs from audit {expected_rows}"
        )
    root_split = mroot / "all"
    for image, label in pairs:
        if not (root_split / image).is_file() or not (root_split / label).is_file():
            raise FileNotFoundError(f"Missing training pair: {image} / {label}")
    return {
        "frame": frame,
        "partition": partition,
        "train": train,
        "validation": validation,
        "train_keys": train_keys,
        "validation_keys": validation_keys,
        "pairs": pairs,
        "scope": scope,
        "root_split": root_split,
        "block_key_sha256": partition_block_hash(partition, block_id),
        "train_key_sha256": sha256_keys(train_keys),
        "validation_key_sha256": sha256_keys(validation_keys),
        "label_scope_sha256": sha256_records(scope),
        "audit": audited,
    }


def _next_batch(
    iterator: Any,
    loader: DataLoader,
) -> tuple[Any, tuple[torch.Tensor, torch.Tensor], bool]:
    try:
        return iterator, next(iterator), False
    except StopIteration:
        iterator = iter(loader)
        return iterator, next(iterator), True


def train_step_regime(
    model: torch.nn.Module,
    pairs: list[tuple[str, str]],
    validation: pd.DataFrame,
    root_split: Path,
    primitives: Any,
    device: str,
    seed: int,
    learning_rate: float,
    config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    training = config["training"]
    batch_size = int(training["batch_size"])
    max_steps = int(training["max_optimizer_steps"])
    interval = int(training["validation_interval_steps"])
    patience = int(training["patience_validation_checks"])
    min_delta = float(training["min_delta"])
    workers = int(training["num_workers"])
    if bool(training["drop_last"]):
        raise RuntimeError("v3 requires drop_last=False")
    if max_steps % interval:
        raise RuntimeError("max_optimizer_steps must be divisible by validation interval")

    dataset = Fit256(
        primitives.SliceDS(pairs, str(root_split), aug=False),
        256,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=False,
        pin_memory=True,
        generator=generator,
    )
    if len(loader) == 0:
        raise RuntimeError("Training DataLoader contains zero batches")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda")
    best_dice = -1.0
    best_step = -1
    best_state: dict[str, torch.Tensor] | None = None
    checks_without_improvement = 0
    history: list[dict[str, Any]] = []
    iterator = iter(loader)
    completed_passes = 0
    interval_loss = 0.0
    interval_batches = 0
    start = time.monotonic()
    final_step = 0
    for step in range(1, max_steps + 1):
        iterator, batch, wrapped = _next_batch(iterator, loader)
        if wrapped:
            completed_passes += 1
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = primitives.seg_loss(model(images), labels)
        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            raise NonFiniteTrainingError(
                f"Non-finite loss at optimizer step {step}: {loss_value}"
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        interval_loss += loss_value
        interval_batches += 1
        final_step = step
        if step % interval:
            continue

        val_dice, val_negative = primitives.val_dice(
            model, validation, str(root_split), device
        )
        val_dice = float(val_dice)
        val_negative = float(val_negative)
        if not math.isfinite(val_dice) or not math.isfinite(val_negative):
            raise NonFiniteTrainingError(
                f"Non-finite validation metric at step {step}"
            )
        mean_loss = interval_loss / max(interval_batches, 1)
        improved = val_dice > best_dice + min_delta
        if improved:
            best_dice = val_dice
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1
        history.append(
            {
                "optimizer_step": step,
                "mean_train_loss_since_check": mean_loss,
                "val_dice_lv_present": val_dice,
                "val_negative_accuracy": val_negative,
                "strict_improvement": improved,
                "checks_without_improvement": checks_without_improvement,
                "completed_loader_passes_before_step": completed_passes,
            }
        )
        print(
            f"step={step:04d} loss={mean_loss:.6f} "
            f"val_dice={val_dice:.6f} val_negative={val_negative:.6f} "
            f"wait={checks_without_improvement}/{patience}",
            flush=True,
        )
        interval_loss = 0.0
        interval_batches = 0
        if checks_without_improvement >= patience:
            break

    if best_state is None:
        raise NonFiniteTrainingError("Training produced no checkpoint candidate")
    metadata = {
        "best_val_dice": best_dice,
        "best_optimizer_step": best_step,
        "final_optimizer_step": final_step,
        "hit_step_cap": final_step >= max_steps,
        "stopped_by_patience": final_step < max_steps,
        "validation_checks": len(history),
        "training_rows": len(pairs),
        "steps_per_full_traversal": int(math.ceil(len(pairs) / batch_size)),
        "full_traversal_equivalent": float(
            final_step / math.ceil(len(pairs) / batch_size)
        ),
        "elapsed_seconds": float(time.monotonic() - start),
        "all_training_losses_finite": True,
    }
    return best_state, history, metadata


def validation_ef_table(
    validation: pd.DataFrame,
    pred_px: dict[tuple[str, str, int], int],
    root_split: Path,
    data_root: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    case_rows: list[dict[str, Any]] = []
    phase_checks: list[dict[str, Any]] = []
    for key in sorted(validation["key"].unique()):
        ground_truth: dict[str, float] = {}
        predicted: dict[str, float] = {}
        for suffix, phase in (("ED", "ed"), ("ES", "es")):
            subset = validation[
                (validation["key"] == key) & (validation["phase"] == phase)
            ]
            if subset.empty:
                raise RuntimeError(f"Missing validation slices for {key}/{phase}")
            nifti = data_root / "dataset" / key / f"{key}_SA_{suffix}_gt.nii.gz"
            voxel_ml = abs(np.linalg.det(nib.load(nifti).affine[:3, :3])) / 1000.0
            gt_pixels = float(subset["gt_px"].sum())
            predicted_pixels = float(
                sum(
                    pred_px.get((key, phase, int(z)), 0)
                    for z in subset["z"]
                )
            )
            capacity = 0
            all_slices_foreground = True
            for row in subset.itertuples():
                image = cv2.imread(
                    str(root_split / str(row.img)), cv2.IMREAD_GRAYSCALE
                )
                if image is None:
                    raise FileNotFoundError(root_split / str(row.img))
                pixels = int(pred_px.get((key, phase, int(row.z)), 0))
                slice_capacity = int(image.shape[0] * image.shape[1])
                capacity += slice_capacity
                all_slices_foreground = (
                    all_slices_foreground and pixels >= slice_capacity
                )
            ground_truth[phase] = gt_pixels * voxel_ml
            predicted[phase] = predicted_pixels * voxel_ml
            phase_checks.append(
                {
                    "key": key,
                    "phase": phase,
                    "predicted_pixels": predicted_pixels,
                    "pixel_capacity": capacity,
                    "all_empty": predicted_pixels == 0,
                    "all_foreground": all_slices_foreground,
                }
            )

        def ef(volumes: dict[str, float]) -> float:
            if volumes["ed"] <= 1e-6:
                return float("nan")
            return 100.0 * (volumes["ed"] - volumes["es"]) / volumes["ed"]

        ef_gt = ef(ground_truth)
        ef_pred = ef(predicted)
        case_rows.append(
            {
                "key": key,
                "edv_gt": ground_truth["ed"],
                "esv_gt": ground_truth["es"],
                "ef_gt": ef_gt,
                "edv_pred": predicted["ed"],
                "esv_pred": predicted["es"],
                "ef_pred": ef_pred,
                "ef_error": ef_pred - ef_gt,
                "ef_abs_error": abs(ef_pred - ef_gt),
            }
        )
    return pd.DataFrame(case_rows), phase_checks


@torch.no_grad()
def evaluate_siemens_validation(
    model: torch.nn.Module,
    validation: pd.DataFrame,
    root_split: Path,
    data_root: Path,
    evaluator: Any,
    device: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model.eval()
    pred_px, dice = evaluator.predict_all(
        model, validation, str(root_split), device
    )
    dice = np.asarray(dice, dtype=float)
    if len(dice) != len(validation):
        raise RuntimeError("Validation evaluator returned a misaligned Dice vector")
    cases, phase_checks = validation_ef_table(
        validation, pred_px, root_split, data_root
    )
    numeric = cases[
        ["edv_gt", "esv_gt", "ef_gt", "edv_pred", "esv_pred", "ef_pred"]
    ].to_numpy(dtype=float)
    finite = bool(np.isfinite(numeric).all() and np.isfinite(dice).all())
    degenerates = [
        row
        for row in phase_checks
        if bool(row["all_empty"]) or bool(row["all_foreground"])
    ]
    summary = {
        "validation_patients": int(len(cases)),
        "validation_ef_bias": (
            float(cases["ef_error"].mean()) if finite else float("nan")
        ),
        "validation_ef_mae": (
            float(cases["ef_abs_error"].mean()) if finite else float("nan")
        ),
        "all_validation_outputs_finite": finite,
        "degenerate_patient_phase": bool(degenerates),
        "degenerate_patient_phase_records": degenerates,
        "prediction_pixel_count_records": phase_checks,
    }
    return cases, summary


def checkpoint_payload_base(
    *,
    state: dict[str, torch.Tensor],
    config: dict[str, Any],
    learning_rate: float,
    seed: int,
    arm: str,
    block_id: str,
    cell: dict[str, Any],
    training_metadata: dict[str, Any],
    prepared: dict[str, Any],
    code_sha256: dict[str, str],
    environment: dict[str, Any],
    encoder_sha256: str,
    initial_model_sha256: str,
) -> dict[str, Any]:
    training = config["training"]
    return {
        "protocol_id": PROTOCOL_ID,
        "model": {key: value.detach().cpu() for key, value in state.items()},
        "encoder": training["encoder"],
        "pretrained": bool(training["pretrained"]),
        "augmentation": bool(training["augmentation"]),
        "optimizer": training["optimizer"],
        "learning_rate": learning_rate,
        "batch_size": int(training["batch_size"]),
        "max_optimizer_steps": int(training["max_optimizer_steps"]),
        "validation_interval_steps": int(training["validation_interval_steps"]),
        "patience_validation_checks": int(training["patience_validation_checks"]),
        "min_delta": float(training["min_delta"]),
        "num_workers": int(training["num_workers"]),
        "drop_last": bool(training["drop_last"]),
        "training_vendor": "SIEMENS",
        "seed": seed,
        "arm": arm,
        "block_id": block_id,
        "cell": cell,
        "train_keys": prepared["train_keys"],
        "validation_keys": prepared["validation_keys"],
        "train_key_sha256": prepared["train_key_sha256"],
        "validation_key_sha256": prepared["validation_key_sha256"],
        "block_key_sha256": prepared["block_key_sha256"],
        "label_rows": len(prepared["pairs"]),
        "label_scope_sha256": prepared["label_scope_sha256"],
        **training_metadata,
        "pretrained_encoder_tensor_sha256": encoder_sha256,
        "initial_model_tensor_sha256": initial_model_sha256,
        "code_sha256": code_sha256,
        "audit": prepared["audit"],
        "environment": environment,
    }


def save_checkpoint_and_verify(
    payload: dict[str, Any], path: Path
) -> tuple[str, str, bool]:
    expected_state_sha = sha256_tensor_state(payload["model"])
    atomic_torch_save(payload, path)
    reloaded = torch_load_trusted(path)
    observed_state_sha = sha256_tensor_state(reloaded["model"])
    return sha256_file(path), observed_state_sha, observed_state_sha == expected_state_sha


def initialize_model(
    primitives: Any, config: dict[str, Any], seed: int
) -> tuple[torch.nn.Module, str, str]:
    primitives.set_seed(seed)
    training = config["training"]
    model = primitives.UNet(
        str(training["encoder"]),
        1,
        1,
        pretrained=bool(training["pretrained"]),
    ).to("cuda")
    encoder_sha = sha256_tensor_state(
        {
            key: value
            for key, value in model.state_dict().items()
            if key.startswith("encoder.")
        }
    )
    initial_sha = sha256_tensor_state(model.state_dict())
    return model, encoder_sha, initial_sha
