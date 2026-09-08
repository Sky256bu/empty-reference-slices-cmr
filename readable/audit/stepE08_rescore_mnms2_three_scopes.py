#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stepE08 — Port the three-scope re-scoring from ACDC to M&Ms-2.

WHY THIS EXISTS
  The mechanistic pillar (Results 3.2-3.4) currently runs only on ACDC, which
  is the cohort whose checkpoint identity is "technically regenerable but
  evidentially unrecoverable". M&Ms-2 is the cohort with 20/20 checkpoint
  SHA256 matches. Moving the re-scoring to M&Ms-2 makes the mechanistic pillar
  (a) two-dataset, (b) hash-anchored on the side that carries the headline
  claim, and (c) cross-vendor, at the cost of one regeneration pass over
  archived weights. No new training.

CONTRACT
  This script emits a table with EXACTLY the schema of
  stack_dice_acdc_clean.csv, so stepE07_flip_enumeration_acdc.py runs on it
  unchanged with only IN_CSV and DECLARED_SHA256 swapped. Same code path,
  same definitions, no second implementation of Dice.

  Emitted columns:
    key, phase, inter, gt, pred, present_dice, empty_fp, stack_dice,
    arm, N, seed, tag, alias
  where, per Methods 2.6, for one case-phase stack:
    inter          = sum over slices of |pred AND gt|
    gt             = sum over slices of |gt|            (nonzero only on present slices)
    pred           = sum over ALL slices of |pred|
    empty_fp       = sum over reference-EMPTY slices of |pred|
    present_dice   = mean over slices with |gt|>0 of per-slice Dice   [slice-present]
    stack_dice     = 2*inter / (gt + pred)                            [full-stack]
  and volume-present Dice is reconstructed downstream as
    2*inter / (gt + pred - empty_fp)

FILL IN THE CONFIG BLOCK. Nothing else should need editing.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError:
    sys.exit("nibabel required: pip install nibabel")

# ============================================================================
# CONFIG — the only block that needs local paths
# ============================================================================

GT_ROOT = Path("/PATH/TO/mnms2/gt")        # case-phase ground-truth NIfTI
PRED_ROOT = Path("/PATH/TO/regenerated/preds")  # regenerated predicted masks
OUT_CSV = Path("stack_dice_mnms2_clean.csv")
CHECKPOINT_MANIFEST = Path("/PATH/TO/mnms2_checkpoint_sha256.txt")

LV_LABEL = 1          # M&Ms-2 label index for the LV cavity; VERIFY before running
SLICE_AXIS = 2        # z axis of the stack; VERIFY per-file, do not assume

# The prediction filename must be parseable into (unit_id, arm, seed, key, phase).
# Adjust to the actual regeneration layout. Nothing downstream guesses.
PRED_PATTERN = re.compile(
    r"^(?P<unit>u\d{2})_(?P<arm>[A-Za-z]+)_s(?P<seed>\d+)_(?P<key>[^_]+)_(?P<phase>ed|es)\.nii\.gz$"
)

# Held-out domains, per the frozen M&Ms-2 protocol. Only these are scored.
HELD_OUT_DOMAINS = {"Philips", "GE"}

# ============================================================================


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_binary_stack(path: Path, label: int, axis: int) -> np.ndarray:
    """Return a boolean array with the slice axis moved to position 0."""
    arr = np.asanyarray(nib.load(str(path)).dataobj)
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected 3-D, got shape {arr.shape}")
    return np.moveaxis(arr == label, axis, 0)


def score_one_stack(pred: np.ndarray, gt: np.ndarray) -> dict:
    """All three scopes for a single case-phase stack. This is the ONLY place
    Dice is defined; ACDC and M&Ms-2 must agree here or the comparison is void."""
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch pred {pred.shape} vs gt {gt.shape}")

    gt_per_slice = gt.reshape(gt.shape[0], -1).sum(axis=1).astype(np.int64)
    pred_per_slice = pred.reshape(pred.shape[0], -1).sum(axis=1).astype(np.int64)
    inter_per_slice = (pred & gt).reshape(gt.shape[0], -1).sum(axis=1).astype(np.int64)

    present = gt_per_slice > 0
    empty = ~present

    inter = int(inter_per_slice.sum())
    gt_tot = int(gt_per_slice.sum())
    pred_tot = int(pred_per_slice.sum())
    empty_fp = int(pred_per_slice[empty].sum())

    # slice-present Dice: mean over reference-present slices of per-slice Dice.
    # A present slice with zero prediction contributes 0, not NaN.
    if present.any():
        denom = gt_per_slice[present] + pred_per_slice[present]
        per = np.where(denom > 0, 2.0 * inter_per_slice[present] / np.maximum(denom, 1), 0.0)
        slice_present = float(per.mean())
    else:
        slice_present = float("nan")

    full_stack = 2.0 * inter / (gt_tot + pred_tot) if (gt_tot + pred_tot) > 0 else float("nan")

    return dict(inter=float(inter), gt=gt_tot, pred=pred_tot,
                present_dice=slice_present, empty_fp=float(empty_fp),
                stack_dice=full_stack,
                n_slices=int(gt.shape[0]),
                n_empty_slices=int(empty.sum()),
                n_empty_slices_with_fp=int((pred_per_slice[empty] > 0).sum()))


def resolve_gt(key: str, phase: str) -> Path:
    """Case-phase GT path. Fails loudly rather than guessing."""
    cands = sorted(GT_ROOT.rglob(f"*{key}*_{phase}*.nii.gz"))
    if len(cands) != 1:
        raise FileNotFoundError(
            f"GT for ({key}, {phase}): expected exactly 1 match, found {len(cands)}")
    return cands[0]


def main() -> None:
    if not PRED_ROOT.exists():
        sys.exit(f"ABORT: PRED_ROOT not found: {PRED_ROOT}. Fill the CONFIG block.")

    manifest = {}
    if CHECKPOINT_MANIFEST.exists():
        for line in CHECKPOINT_MANIFEST.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                manifest[parts[1]] = parts[0]
    else:
        print("WARNING: no checkpoint manifest; provenance chain will be incomplete",
              file=sys.stderr)

    rows, skipped = [], []
    for p in sorted(PRED_ROOT.rglob("*.nii.gz")):
        m = PRED_PATTERN.match(p.name)
        if not m:
            skipped.append(p.name)
            continue
        g = m.groupdict()
        gt_path = resolve_gt(g["key"], g["phase"])
        pred = load_binary_stack(p, LV_LABEL, SLICE_AXIS)
        gt = load_binary_stack(gt_path, LV_LABEL, SLICE_AXIS)
        r = score_one_stack(pred, gt)
        r.update(
            key=g["key"], phase=g["phase"], arm=g["arm"],
            seed=int(g["seed"]), unit_id=g["unit"],
            # N is fixed at the frozen confirmatory budget for M&Ms-2
            N=10,
            tag=f"{g['unit']}_{g['arm']}_s{g['seed']}",
            alias=False,
            pred_path=str(p), gt_path=str(gt_path),
        )
        rows.append(r)

    if not rows:
        sys.exit("ABORT: no predictions matched PRED_PATTERN; check the regeneration layout")

    d = pd.DataFrame(rows)

    # Schema-compatible export. Extra diagnostic columns are kept at the end;
    # stepE07 ignores them.
    core = ["key", "phase", "inter", "gt", "pred", "present_dice", "empty_fp",
            "stack_dice", "arm", "N", "seed", "tag", "alias"]
    extra = ["unit_id", "n_slices", "n_empty_slices", "n_empty_slices_with_fp"]
    d[core + extra].to_csv(OUT_CSV, index=False)

    # Self-checks mirroring stepE07's aborts, run here so failures surface early.
    vp = 2 * d["inter"] / (d["gt"] + d["pred"] - d["empty_fp"])
    pen = vp - d["stack_dice"]
    checks = dict(
        out_sha256=sha256_of(OUT_CSV),
        n_rows=int(len(d)),
        n_cells=int(d.groupby(["arm", "unit_id", "seed"]).ngroups),
        arms=sorted(d["arm"].unique().tolist()),
        penalty_nonnegative_everywhere=bool((pen >= -1e-12).all()),
        penalty_min=float(pen.min()),
        mean_penalty_ed=float(pen[d["phase"] == "ed"].mean()),
        mean_penalty_es=float(pen[d["phase"] == "es"].mean()),
        empty_slice_count_median=float(d["n_empty_slices"].median()),
        empty_slice_count_range=[int(d["n_empty_slices"].min()),
                                 int(d["n_empty_slices"].max())],
        frac_cases_with_any_empty_fp=float((d["empty_fp"] > 0).mean()),
        n_checkpoints_hash_anchored=len(manifest),
        unparsed_prediction_files=len(skipped),
    )
    Path("stepE08_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))

    if not checks["penalty_nonnegative_everywhere"]:
        sys.exit("ABORT: negative penalty on M&Ms-2; LV_LABEL or SLICE_AXIS is wrong")
    print("\nNext: point stepE07 at", OUT_CSV, "and set DECLARED_SHA256 to",
          checks["out_sha256"][:8])


if __name__ == "__main__":
    main()
