#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
81b_export_mnms2_interp.py -- M&Ms-2 two-anchor SDF interpolation pseudo-labels (arm I, cross-vendor control)
================================================================================
Cross-vendor version of control 2: show that what wins at a low budget in 82 is "SAM
  inter-slice propagation (B)" and not "filling the interior slices by any means (I)".
  Same SDF shape interpolation as 73b, with the data source switched to M&Ms-2 and the label
  convention to LV=1:
    using the two anchor slices GT(lo, hi) only, linearly interpolate the SDF for every slice
    in the span -> mask_z=(s_z>0); anchor slices take GT.
  Written to {out_dir}/all/pi/{key}_{phase}_z{z:02d}.png (same naming convention as the imgs/ of 81).
  Arm I of 74 takes its labels by matching imgs/ -> pi/ and does not read a manifest column, so
  this script leaves the manifest untouched.
  No SAM, pure scipy, CPU, seconds.

Key point: only the **interior slices** of the SIEMENS training pool are read during arm I
      training in 82; but for robustness (so that changing the training vendor later does not
      break), pi is exported here for **every** stack whose LV span is >= 3. Test-domain
      evaluation uses the true GT and does not touch pi.

Usage:
    cd /root/autodl-tmp/sam3-main
    python 81b_export_mnms2_interp.py --root /root/autodl-tmp/mnms/MnM2 \
        --out_dir /root/autodl-tmp/mnms/seg_lv --lv_label 1 --min_lv_px 15
Products: {out_dir}/all/pi/*.png
Self-check: the **dice of interpolated interior slices vs GT (overall and per vendor)** is printed
      at the end -- compare it against the SAM self-dice of 81. If the interpolation dice is
      markedly lower (e.g. LV~0.645 on ACDC), that forecasts arm I being poisoned (densification
      is harmful).
Next: for N in 10 25 50 0; do for S in 0 1 2; do
          python 82_mnms2_crossvendor.py --mroot .../seg_lv --root .../MnM2 \
              --n_train_pat $N --seed $S --arms I ; done; done
================================================================================
"""
import argparse, glob, importlib, os, sys
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


def dice(a, b):
    a = np.asarray(a).astype(bool); b = np.asarray(b).astype(bool)
    d = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / d) if d > 0 else float("nan")


def vendor_label(v):
    vl = str(v).lower()
    if "siemens" in vl: return "SIEMENS"
    if "philip" in vl:  return "Philips"
    if "ge" in vl:      return "GE"
    return str(v).strip() or "UNK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/mnms/MnM2")
    ap.add_argument("--code_dir", default="/root/autodl-tmp/sam3-main")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/mnms/seg_lv")
    ap.add_argument("--lv_label", type=int, default=1, help="M&Ms: LV=1")
    ap.add_argument("--min_lv_px", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pi_dir = os.path.join(args.out_dir, "all", "pi")
    os.makedirs(pi_dir, exist_ok=True)

    # reuse the vendor map of 81 (for per-vendor sanity statistics only)
    code_dir = os.path.abspath(args.code_dir)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    try:
        E81 = importlib.import_module("81_export_mnms2_pseudo")
        vmap = E81.load_vendor_map(args.root)
    except Exception as e:
        log(f"[!] failed to reuse 81.load_vendor_map ({type(e).__name__}); vendor statistics will be empty: {e}")
        vmap = {}

    ds = os.path.join(args.root, "dataset")
    if not os.path.isdir(ds):
        ds = args.root
    cases = sorted([d for d in glob.glob(os.path.join(ds, "*")) if os.path.isdir(d)])
    if args.limit > 0:
        cases = cases[:args.limit]
    log(f"[interp] {len(cases)} cases -> {pi_dir}  (LV label={args.lv_label})")

    n_png, dsc_all = 0, []
    dsc_by_v = {}
    for ci, cdir in enumerate(cases, 1):
        code = os.path.basename(cdir)
        vend = vendor_label(vmap.get(code, ("", ""))[0])
        for PH, phase in (("ED", "ed"), ("ES", "es")):
            gp = os.path.join(cdir, f"{code}_SA_{PH}_gt.nii.gz")
            if not os.path.exists(gp):
                continue
            gt = np.rint(nib.load(gp).get_fdata()).astype(np.int16)
            if gt.ndim != 3:
                continue
            gtz = np.transpose(gt, (2, 0, 1))            # [Z,H,W]
            Z = gtz.shape[0]
            lv = [(gtz[z] == args.lv_label) for z in range(Z)]
            present = [z for z in range(Z) if lv[z].sum() >= args.min_lv_px]
            if len(present) < 3:
                continue
            lo, hi = present[0], present[-1]
            s_lo, s_hi = sdf(lv[lo]), sdf(lv[hi])
            for z in range(lo, hi + 1):
                base = f"{code}_{phase}_z{z:02d}.png"
                if z == lo or z == hi:
                    m = lv[z]
                else:
                    t = (z - lo) / (hi - lo)
                    m = ((1 - t) * s_lo + t * s_hi) > 0
                    if lv[z].sum() > 0:                    # statistics only, not used in training
                        dd = dice(m, lv[z])
                        dsc_all.append(dd)
                        dsc_by_v.setdefault(vend, []).append(dd)
                cv2.imwrite(os.path.join(pi_dir, base), (m.astype(np.uint8) * 255))
                n_png += 1
        if ci % 50 == 0:
            log(f"  ...{ci}/{len(cases)}")

    log(f"\n[out] wrote {n_png} pi slices -> {pi_dir}")
    if dsc_all:
        a = np.array(dsc_all)
        log(f"[interp vs GT] interior-slice dice: overall mean={a.mean():.3f} (n={len(a)})")
        for v in ["SIEMENS", "Philips", "GE"]:
            if v in dsc_by_v:
                b = np.array(dsc_by_v[v])
                log(f"    {v:>8s}: mean={b.mean():.3f}  (n={len(b)})")
        log("  [reading] compare against the SAM self/propagation quality of 81: the lower the interpolation dice,")
        log("         the more arm I is poisoned, and the harder the cross-vendor counter-evidence that")
    else:
        log("[!] no interior slice available for statistics (check lv_label/min_lv_px or the data path)")


if __name__ == "__main__":
    main()
