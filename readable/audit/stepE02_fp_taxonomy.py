#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stepE02_fp_taxonomy.py
======================
**Reference implementation + self-test** of the empty-slice false-positive taxonomy
(connected / isolated)

Upstream: EVAL_VALIDITY_EXTENSION_v0_3_DRAFT_FP_TAXONOMY.md
Identity: a design-phase implementation; **not yet validated on any real segmentation mask**.
      Every self-test in this file rests on **synthetic phantoms** (whose answers are known by
      construction). Validation on real ACDC masks must be added before locking (see spec 7).

Reads no frozen data and produces no number related to the manuscript.

Usage:
  python3 stepE02_fp_taxonomy.py --selftest
"""

import argparse
import sys

import numpy as np
from scipy import ndimage

# --------------------------------------------------------------- connectivity structuring element
STRUCT = {
    26: np.ones((3, 3, 3), dtype=bool),
    6: ndimage.generate_binary_structure(3, 1),
}


def _label(mask, connectivity):
    if connectivity not in STRUCT:
        raise ValueError("connectivity must be 26 or 6")
    lab, n = ndimage.label(mask, structure=STRUCT[connectivity])
    return lab, n


def decompose_empty_slice_fp(pred, gt, spacing=(1.0, 1.0, 1.0), connectivity=26):
    """
    Input
    ----
    pred, gt : (Z, Y, X) boolean arrays, **single structure, single phase, single case**
               axis 0 is the slice axis (the through-plane direction of a short-axis stack / the z of a CT)
    spacing  : (dz, dy, dx), in mm
    connectivity : 26 (main analysis) or 6 (sensitivity analysis)

    Output
    ----
    dict, volumes in mm^3
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if pred.shape != gt.shape or pred.ndim != 3:
        raise ValueError("pred/gt must have the same shape and be 3D")

    vox_mm3 = float(np.prod(spacing))
    Z = pred.shape[0]

    # ---- GT axial support
    z_present = np.array([gt[z].any() for z in range(Z)], dtype=bool)
    out = {
        "n_slices": Z,
        "n_gt_present_slices": int(z_present.sum()),
        "gt_volume_mm3": float(gt.sum() * vox_mm3),
        "pred_volume_mm3": float(pred.sum() * vox_mm3),
        "connectivity": connectivity,
    }

    # cases with a wholly empty GT: undefined for this taxonomy, so flag explicitly rather than silently returning 0
    if not z_present.any():
        out.update(dict.fromkeys(
            ["empty_fp_total_mm3", "empty_fp_connected_mm3",
             "empty_fp_isolated_mm3", "empty_fp_beyond_mm3",
             "empty_fp_interior_mm3", "empty_fp_after_lcc_mm3",
             "lcc_removable_frac"], np.nan))
        out["gt_all_empty"] = True
        return out
    out["gt_all_empty"] = False

    z_lo, z_hi = int(np.argmax(z_present)), int(Z - 1 - np.argmax(z_present[::-1]))
    z_empty = ~z_present
    z_interior = z_empty.copy()
    z_interior[:z_lo] = False
    z_interior[z_hi + 1:] = False
    z_beyond = z_empty & ~z_interior

    out["n_empty_slices"] = int(z_empty.sum())
    out["n_empty_interior"] = int(z_interior.sum())
    out["n_empty_beyond"] = int(z_beyond.sum())

    # ---- predicted foreground on empty slices (GT is by definition all zero there, so all of it is FP)
    empty_mask = np.zeros_like(pred)
    empty_mask[z_empty] = pred[z_empty]
    out["empty_fp_total_mm3"] = float(empty_mask.sum() * vox_mm3)

    # ---- connected-component decomposition
    lab, n = _label(pred, connectivity)
    out["n_components"] = int(n)

    if n == 0:
        out.update({"empty_fp_connected_mm3": 0.0,
                    "empty_fp_isolated_mm3": 0.0,
                    "empty_fp_beyond_mm3": 0.0,
                    "empty_fp_interior_mm3": 0.0,
                    "empty_fp_after_lcc_mm3": 0.0,
                    "lcc_removable_frac": np.nan,
                    "n_gt_anchored_components": 0})
        return out

    # GT-anchored criterion: **spatial overlap with a GT voxel** (not "lying on a GT-present slice")
    anchored_ids = set(np.unique(lab[gt & (lab > 0)]).tolist()) - {0}
    out["n_gt_anchored_components"] = len(anchored_ids)

    anchored_mask = np.isin(lab, list(anchored_ids)) if anchored_ids \
        else np.zeros_like(pred)

    conn = empty_mask & anchored_mask
    iso = empty_mask & ~anchored_mask
    out["empty_fp_connected_mm3"] = float(conn.sum() * vox_mm3)
    out["empty_fp_isolated_mm3"] = float(iso.sum() * vox_mm3)

    beyond = np.zeros_like(pred); beyond[z_beyond] = empty_mask[z_beyond]
    inter = np.zeros_like(pred); inter[z_interior] = empty_mask[z_interior]
    out["empty_fp_beyond_mm3"] = float(beyond.sum() * vox_mm3)
    out["empty_fp_interior_mm3"] = float(inter.sum() * vox_mm3)

    # ---- canonical LCC operator (simulating "wouldn't a largest-connected-component post-process just fix this")
    #      this is a different thing from the pipeline's own post-processing factor (v0.3 3.4) and must not be conflated with it
    sizes = ndimage.sum_labels(np.ones_like(pred, dtype=np.int64), lab,
                               index=np.arange(1, n + 1))
    keep = int(np.argmax(sizes)) + 1          # on a tie take the smallest label, for determinism
    pred_lcc = (lab == keep)
    empty_after = np.zeros_like(pred)
    empty_after[z_empty] = pred_lcc[z_empty]
    out["empty_fp_after_lcc_mm3"] = float(empty_after.sum() * vox_mm3)
    tot = out["empty_fp_total_mm3"]
    out["lcc_removable_frac"] = (
        float((tot - out["empty_fp_after_lcc_mm3"]) / tot) if tot > 0 else np.nan)

    # ---- relative burden (dimensionless)
    out["empty_fp_rel_burden"] = (
        tot / out["gt_volume_mm3"] if out["gt_volume_mm3"] > 0 else np.nan)
    return out


# =============================================================== self-test (synthetic phantoms)
def _blank(Z=10, Y=20, X=20):
    return np.zeros((Z, Y, X), dtype=bool)


def _selftest():
    fails = []

    def chk(name, got, exp, tol=1e-9):
        ok = (np.isnan(got) and np.isnan(exp)) or abs(got - exp) <= tol
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: got={got} exp={exp}")
        if not ok:
            fails.append(name)

    sp = (2.0, 1.0, 1.0)          # dz=2mm -> 2 mm^3 per voxel
    v = 2.0

    # T1 the prediction spills one slice beyond the span and fuses with the main body -> connected & beyond
    print("\n[T1] one slice beyond the span, fused with the main body")
    gt = _blank(); gt[3:7, 8:12, 8:12] = True
    pr = _blank(); pr[3:8, 8:12, 8:12] = True          # an extra 4x4=16 voxels at z=7
    r = decompose_empty_slice_fp(pr, gt, sp)
    chk("T1 total", r["empty_fp_total_mm3"], 16 * v)
    chk("T1 connected", r["empty_fp_connected_mm3"], 16 * v)
    chk("T1 isolated", r["empty_fp_isolated_mm3"], 0.0)
    chk("T1 beyond", r["empty_fp_beyond_mm3"], 16 * v)
    chk("T1 interior", r["empty_fp_interior_mm3"], 0.0)
    chk("T1 LCC removable fraction", r["lcc_removable_frac"], 0.0)   # LCC cannot remove it

    # T2 an isolated small block on an empty slice -> isolated; LCC can remove it
    print("\n[T2] isolated small block on an empty slice")
    gt = _blank(); gt[3:7, 8:12, 8:12] = True
    pr = _blank(); pr[3:7, 8:12, 8:12] = True
    pr[8, 2:4, 2:4] = True                              # 4 voxels, far from the main body
    r = decompose_empty_slice_fp(pr, gt, sp)
    chk("T2 total", r["empty_fp_total_mm3"], 4 * v)
    chk("T2 connected", r["empty_fp_connected_mm3"], 0.0)
    chk("T2 isolated", r["empty_fp_isolated_mm3"], 4 * v)
    chk("T2 LCC removable fraction", r["lcc_removable_frac"], 1.0)

    # T3 a slice missing inside the GT, filled by the prediction -> connected & interior
    print("\n[T3] an empty slice inside the GT is filled")
    gt = _blank(); gt[3:7, 8:12, 8:12] = True; gt[5] = False
    pr = _blank(); pr[3:7, 8:12, 8:12] = True
    r = decompose_empty_slice_fp(pr, gt, sp)
    chk("T3 total", r["empty_fp_total_mm3"], 16 * v)
    chk("T3 connected", r["empty_fp_connected_mm3"], 16 * v)
    chk("T3 interior", r["empty_fp_interior_mm3"], 16 * v)
    chk("T3 beyond", r["empty_fp_beyond_mm3"], 0.0)

    # T4 a perfect prediction -> all zero
    print("\n[T4] perfect prediction")
    gt = _blank(); gt[3:7, 8:12, 8:12] = True
    r = decompose_empty_slice_fp(gt.copy(), gt, sp)
    chk("T4 total", r["empty_fp_total_mm3"], 0.0)
    chk("T4 LCC fraction (undefined)", r["lcc_removable_frac"], np.nan)

    # T5 an empty prediction -> all zero, no error raised
    print("\n[T5] empty prediction")
    gt = _blank(); gt[3:7, 8:12, 8:12] = True
    r = decompose_empty_slice_fp(_blank(), gt, sp)
    chk("T5 total", r["empty_fp_total_mm3"], 0.0)
    chk("T5 components", float(r["n_components"]), 0.0)

    # T6 a discrete error block on a present slice -> **not counted as empty-slice FP** (the key point: the taxonomy stays in scope)
    print("\n[T6] a discrete error block on a present slice must not be counted as empty-slice FP")
    gt = _blank(); gt[3:7, 8:12, 8:12] = True
    pr = _blank(); pr[3:7, 8:12, 8:12] = True
    pr[4, 2:4, 2:4] = True                              # lies on a GT-present slice
    r = decompose_empty_slice_fp(pr, gt, sp)
    chk("T6 total", r["empty_fp_total_mm3"], 0.0)
    chk("T6 isolated", r["empty_fp_isolated_mm3"], 0.0)

    # T7 connected + isolated mixed: LCC can remove only the isolated part
    print("\n[T7] mixed: LCC removes only the isolated part")
    gt = _blank(); gt[3:7, 8:12, 8:12] = True
    pr = _blank(); pr[3:8, 8:12, 8:12] = True           # 16 voxels beyond the span (connected)
    pr[9, 2:4, 2:4] = True                              # 4 isolated voxels
    r = decompose_empty_slice_fp(pr, gt, sp)
    chk("T7 total", r["empty_fp_total_mm3"], 20 * v)
    chk("T7 connected", r["empty_fp_connected_mm3"], 16 * v)
    chk("T7 isolated", r["empty_fp_isolated_mm3"], 4 * v)
    chk("T7 LCC removable fraction", r["lcc_removable_frac"], 4 / 20)

    # T8 the connectivity choice changes the conclusion: a block beyond the span that is only diagonally adjacent
    print("\n[T8] 26 vs 6 connectivity: the assignment of a diagonally adjacent block flips")
    gt = _blank(); gt[3:7, 8:12, 8:12] = True
    pr = _blank(); pr[3:7, 8:12, 8:12] = True
    pr[7, 12:14, 12:14] = True                          # touches the main body at a corner only
    r26 = decompose_empty_slice_fp(pr, gt, sp, connectivity=26)
    r6 = decompose_empty_slice_fp(pr, gt, sp, connectivity=6)
    chk("T8 26-connected", r26["empty_fp_connected_mm3"], 4 * v)
    chk("T8 26-isolated", r26["empty_fp_isolated_mm3"], 0.0)
    chk("T8 6-connected", r6["empty_fp_connected_mm3"], 0.0)
    chk("T8 6-isolated", r6["empty_fp_isolated_mm3"], 4 * v)
    chk("T8 26 LCC removable fraction", r26["lcc_removable_frac"], 0.0)
    chk("T8 6 LCC removable fraction", r6["lcc_removable_frac"], 1.0)

    # T9 a wholly empty GT -> explicit NaN, not a silent 0
    print("\n[T9] a case with wholly empty GT yields an explicit NaN")
    pr = _blank(); pr[5, 8:12, 8:12] = True
    r = decompose_empty_slice_fp(pr, _blank(), sp)
    chk("T9 total", r["empty_fp_total_mm3"], np.nan)
    print(f"  INFO  gt_all_empty={r['gt_all_empty']}")

    print("\n" + "=" * 60)
    if fails:
        print(f"[SELFTEST FAILED] {len(fails)} items: {fails}")
        return 1
    print("[SELFTEST PASSED] all 9 phantom groups pass")
    print("Warning: passing on synthetic phantoms does not mean the code works on real masks. Validation on real ACDC masks must be added before locking (spec 7)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    ap.print_help()
