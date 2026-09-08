#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
73b_export_interp_pseudo.py -- two-anchor SDF shape-interpolation pseudo-labels (non-SAM densification control)
================================================================================
Control for question 2: show that what wins at a low budget is "SAM propagation" and not
  "filling the interior slices by any means at all".
  Using only the same two anchor slices GT(lo, hi) as C/B, fill every slice in the span by
  **linear interpolation of the signed distance field (SDF)**:
    sdf(m) = edt(m) - edt(~m)   (positive inside)
    s_z    = (1-t)*sdf_lo + t*sdf_hi ,  t=(z-lo)/(hi-lo)
    mask_z = s_z > 0
  Anchor slices take pi=GT; slices outside the LV span are empty. Written to
  {out}/{split}/pi/{key}_{phase}_z{z:02d}.png (same naming convention as 73).
  Arm I of 74 takes its labels from imgs/ -> pi/ by matching names. No SAM, pure scipy, seconds.

Usage (same arguments as 73; train is required, test optional):
    python 73b_export_interp_pseudo.py --data_dir /root/autodl-tmp/acdc/train \
        --out_dir /root/autodl-tmp/acdc/seg_lv --split train --lv_label 3 --min_lv_px 15
Products: {out}/{split}/pi/*.png (plus printed interpolation self/neighbour sanity checks)
Note: out_dir/split must match the seg_lv exported by 73 (imgs/gts are shared; only pi is added).
================================================================================
"""
import argparse, glob, os
import cv2, numpy as np, nibabel as nib
from scipy.ndimage import distance_transform_edt as edt


def log(m): print(m, flush=True)


def sdf(mask):
    mask = mask.astype(bool)
    if not mask.any():
        return np.full(mask.shape, -1e4, np.float32)
    if mask.all():
        return np.full(mask.shape, 1e4, np.float32)
    return (edt(mask) - edt(~mask)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="/root/autodl-tmp/acdc/seg_lv")
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--lv_label", type=int, default=3)
    ap.add_argument("--min_lv_px", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    pi_dir = os.path.join(args.out_dir, args.split, "pi")
    os.makedirs(pi_dir, exist_ok=True)

    pats = sorted(glob.glob(os.path.join(args.data_dir, "patient*")))
    if args.limit > 0: pats = pats[: args.limit]
    log(f"[interp] split={args.split} {len(pats)} cases -> {pi_dir}")

    n, dsc = 0, []
    for pdir in pats:
        key = os.path.basename(pdir)
        for phase in ("ed", "es"):
            gp = os.path.join(pdir, f"{key}_sax_{phase}_gt.nii.gz")
            if not os.path.exists(gp): continue
            gt = np.rint(nib.load(gp).get_fdata()).astype(np.int16)
            gtz = np.transpose(gt, (2, 0, 1))
            lv = [(gtz[z] == args.lv_label) for z in range(gtz.shape[0])]
            present = [z for z in range(len(lv)) if lv[z].sum() >= args.min_lv_px]
            if len(present) < 3: continue
            lo, hi = present[0], present[-1]
            s_lo, s_hi = sdf(lv[lo]), sdf(lv[hi])
            for z in range(lo, hi + 1):
                base = f"{key}_{phase}_z{z:02d}.png"
                if z == lo or z == hi:
                    m = lv[z]
                else:
                    t = (z - lo) / (hi - lo)
                    m = ((1 - t) * s_lo + t * s_hi) > 0
                    # sanity: dice against the true gt (reported only, not used for training)
                    if lv[z].sum() > 0:
                        inter = np.logical_and(m, lv[z]).sum()
                        dsc.append(2 * inter / (m.sum() + lv[z].sum() + 1e-6))
                cv2.imwrite(os.path.join(pi_dir, base), (m.astype(np.uint8) * 255))
                n += 1
    log(f"[interp] wrote {n} slices. Interpolated interior slices vs GT dice: "
        f"mean={np.mean(dsc):.3f} (n={len(dsc)})  <- compare against SAM propagation quality")


if __name__ == "__main__":
    main()
