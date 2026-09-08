#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
73_export_lv_pseudo.py -- export ACDC slices as a "student dataset" + two-anchor propagated pseudo-labels
================================================================================
Purpose (the foundation of the load-bearing experiment):
  Provide the A/B/C student arms (full-dense / 2-anchor pseudo-label / sparse-only)
  with **one and the same** slice-level png dataset + a single manifest.csv. The arms
  differ only in which rows/labels they select from the manifest, which guarantees a
  controlled comparison of "same architecture, same recipe, label source only".

For each phase (ed/es) of each patient:
  - Read img (percentile-stretched u8) + gt (0=bg 1=RV 2=MYO 3=LV; this script keeps LV=3 only).
  - Find the slices containing LV; the bottom/top slices lo/hi serve as anchors (the two
    clinically annotated slices). SAM3.1 bidirectional inter-slice propagation:
      fwd = forward from the bottom anchor lo;  bwd = backward from the top anchor hi.
  - **Every** slice z is exported: img.png + gt.png (0/255).
    * Anchor slices (lo, hi): pseudo-label = GT (clinically annotated already).
    * Interior slices: pseudo_fwd = fwd[z] , pseudo_bwd = bwd[z] , disagree = 1-dice(fwd,bwd).
    * Slices outside the LV span (apex/base without LV): empty gt mask, no pseudo-label
      (pf/pb are left as blank pngs).
  - One manifest row per slice recording split/key/phase/z/is_anchor/dist/in_lvspan/disagree/gt_px
    plus the relative paths of the four pngs.

How the three arms read it (the downstream 74 training script filters accordingly):
  A (full-dense): use gt.png on all slices (including empty slices outside the LV span, so
    the student learns "there is no LV here").
  B (2-anchor pseudo-label): gt.png on anchor slices; pseudo_fwd.png on interior slices
    inside the LV span; empty gt outside the span.
  C (sparse-only): gt.png on slices with is_anchor==1 only.
  B+ (QC-weighted, optional follow-up): as B, but interior slices are weighted/filtered by
    disagree (this script already stores disagree, at no extra propagation cost).

Reuses the 29_baseline_50 loading chain + propagate_dir + probe_ted_03_bestof2.frames_to_mp4,
i.e. **exactly the same primitives** as 70/72; no new API is explored.

Usage:
    cd /root/autodl-tmp/sam3-main
    # train (required, produces pseudo-labels):
    python 73_export_lv_pseudo.py \
        --data_dir /root/autodl-tmp/acdc/train \
        --out_dir  /root/autodl-tmp/acdc/seg_lv \
        --split train --limit 5          # smoke first; once self-dice ~1 checks out, use --limit 0
    # test (for evaluation; propagation is run only to store disagree, and is dispensable):
    python 73_export_lv_pseudo.py \
        --data_dir /root/autodl-tmp/acdc/test \
        --out_dir  /root/autodl-tmp/acdc/seg_lv \
        --split test --limit 0
Products: {out_dir}/{split}/imgs|gts|pf|pb/*.png  +  {out_dir}/manifest_{split}.csv
Self-check: self-dice is printed at the end (anchors against themselves should be ~1.000).
A mismatch means an injection or resolution problem: stop.
================================================================================
"""
import argparse, glob, importlib, os, sys, csv
import cv2, numpy as np, nibabel as nib

try:
    from scipy.ndimage import distance_transform_edt  # noqa (same environment as 70/72)
except Exception:
    pass


def log(m): print(m, flush=True)


def vol_to_u8_frames(vol_hwz, plo=1.0, phi=99.0):
    """(H,W,Z) MRI -> uint8 [Z,H,W], 1/99 percentile stretch (must not use /256)."""
    v = np.transpose(vol_hwz, (2, 0, 1)).astype(np.float32)
    lo, hi = np.percentile(v, plo), np.percentile(v, phi)
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max() + 1e-6)
    return np.ascontiguousarray(
        (np.clip((v - lo) / (hi - lo), 0, 1) * 255 + 0.5).astype(np.uint8))


def dice(a, b):
    a = np.asarray(a).astype(bool); b = np.asarray(b).astype(bool)
    d = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / d) if d > 0 else float("nan")


def to_shape(m, shape):
    m = np.asarray(m)
    if m.shape != shape:
        m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    return m.astype(bool)


def save_png(path, arr_u8):
    cv2.imwrite(path, arr_u8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="acdc/train or acdc/test")
    ap.add_argument("--code_dir", default="/root/autodl-tmp/sam3-main")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/acdc/seg_lv")
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--lv_label", type=int, default=3)
    ap.add_argument("--min_lv_px", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0, help="0=all")
    args = ap.parse_args()

    for sub in ("imgs", "gts", "pf", "pb"):
        os.makedirs(os.path.join(args.out_dir, args.split, sub), exist_ok=True)

    code_dir = os.path.abspath(args.code_dir)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    os.chdir(code_dir)
    B = importlib.import_module("29_baseline_50")
    P = importlib.import_module("probe_ted_03_bestof2")
    import torch
    assert torch.cuda.is_available(), "CUDA required"

    from types import SimpleNamespace
    predictor = B.build(SimpleNamespace(ckpt=B.CKPT, bpe=B.BPE))
    B.patch_load_video_frames()
    inner = predictor.model.tracker.model
    B.graft_backbone(predictor, inner); B.shim_backbone_forward(inner)
    B.guard_highres_convs(inner); B.patch_preflight(inner)

    work = os.path.join(args.out_dir, "_w"); os.makedirs(work, exist_ok=True)
    pats = sorted(glob.glob(os.path.join(args.data_dir, "patient*")))
    if args.limit > 0: pats = pats[: args.limit]
    log(f"[data] split={args.split}  {len(pats)} cases  (LV label={args.lv_label})")

    man_path = os.path.join(args.out_dir, f"manifest_{args.split}.csv")
    fld = ["split", "key", "phase", "z", "is_anchor", "dist", "in_lvspan",
           "disagree", "gt_px", "img", "gt", "pf", "pb"]
    fo = open(man_path, "w", newline=""); wr = csv.DictWriter(fo, fieldnames=fld); wr.writeheader()

    self_dices, n_rows, n_stack = [], 0, 0
    for pi, pdir in enumerate(pats, 1):
        key = os.path.basename(pdir)
        for phase in ("ed", "es"):
            img_p = os.path.join(pdir, f"{key}_sax_{phase}.nii.gz")
            gt_p = os.path.join(pdir, f"{key}_sax_{phase}_gt.nii.gz")
            if not (os.path.exists(img_p) and os.path.exists(gt_p)):
                continue
            vol = nib.load(img_p).get_fdata()
            gt = np.rint(nib.load(gt_p).get_fdata()).astype(np.int16)
            frames = vol_to_u8_frames(vol)                 # [Z,H,W] u8
            Z, H, W = frames.shape
            gtz = np.transpose(gt, (2, 0, 1))              # [Z,H,W]
            lv = [(gtz[z] == args.lv_label) for z in range(Z)]
            present = [z for z in range(Z) if lv[z].sum() >= args.min_lv_px]
            if len(present) < 3:
                # Too little LV: still export img/gt (empty or nearly empty), but no anchor and no pseudo-label; enters arm A as a negative sample only
                for z in range(Z):
                    _dump_row(wr, args, key, phase, z, frames[z], lv[z],
                              None, None, is_anchor=0, dist=-1, in_span=0,
                              disagree=float("nan"))
                    n_rows += 1
                continue
            lo, hi = present[0], present[-1]

            mp4 = P.frames_to_mp4(frames, os.path.join(work, f"{key}_{phase}.mp4"))
            try:
                st = B.fresh_inject(inner, mp4, lo, lv[lo], 1)
                fwd = B.propagate_dir(inner, st, lo, reverse=False, span=Z)
                del st; torch.cuda.empty_cache()
            except Exception as e:
                log(f"  {key}/{phase} fwd exception {type(e).__name__}: {e}"); fwd = {}
            try:
                st = B.fresh_inject(inner, mp4, hi, lv[hi], 1)
                bwd = B.propagate_dir(inner, st, hi, reverse=True, span=Z)
                del st; torch.cuda.empty_cache()
            except Exception as e:
                log(f"  {key}/{phase} bwd exception {type(e).__name__}: {e}"); bwd = {}

            if lo in fwd: self_dices.append(dice(to_shape(fwd[lo], lv[lo].shape), lv[lo]))
            if hi in bwd: self_dices.append(dice(to_shape(bwd[hi], lv[hi].shape), lv[hi]))

            for z in range(Z):
                g = lv[z]
                in_span = 1 if (lo <= z <= hi) else 0
                is_anchor = 1 if (z == lo or z == hi) else 0
                dist = min(z - lo, hi - z) if in_span else -1
                pf = pb = None
                disagree = float("nan")
                if in_span and not is_anchor:
                    f = to_shape(fwd[z], g.shape) if z in fwd else np.zeros_like(g)
                    b = to_shape(bwd[z], g.shape) if z in bwd else np.zeros_like(g)
                    dd = dice(f, b)
                    disagree = 0.0 if dd != dd else (1.0 - dd)
                    pf, pb = f, b
                elif is_anchor:
                    pf = pb = g            # anchor pseudo-label = GT
                    disagree = 0.0
                _dump_row(wr, args, key, phase, z, frames[z], g, pf, pb,
                          is_anchor, dist, in_span, disagree)
                n_rows += 1
            n_stack += 1
        if pi % 10 == 0: log(f"  ...{pi}/{len(pats)}")

    fo.close()
    log(f"\n[out] {n_rows} slices (from {n_stack} propagable stacks) -> {man_path}")
    if self_dices:
        sd = np.array([x for x in self_dices if x == x])
        log(f"[self-dice] anchors against themselves mean={sd.mean():.3f} min={sd.min():.3f} (expected ~1.000)")
        if sd.mean() < 0.95:
            log("  [!] self-dice is low -> injection or resolution problem; do not train students yet, check propagation.")


def _dump_row(wr, args, key, phase, z, img_u8, gt_bool, pf, pb,
              is_anchor, dist, in_span, disagree):
    base = f"{key}_{phase}_z{z:02d}"
    d = os.path.join(args.out_dir, args.split)
    ip = os.path.join("imgs", base + ".png")
    gp = os.path.join("gts", base + ".png")
    save_png(os.path.join(d, ip), img_u8)
    save_png(os.path.join(d, gp), (gt_bool.astype(np.uint8) * 255))
    pfp = pbp = ""
    if pf is not None:
        pfp = os.path.join("pf", base + ".png")
        save_png(os.path.join(d, pfp), (np.asarray(pf).astype(np.uint8) * 255))
    if pb is not None:
        pbp = os.path.join("pb", base + ".png")
        save_png(os.path.join(d, pbp), (np.asarray(pb).astype(np.uint8) * 255))
    wr.writerow(dict(split=args.split, key=key, phase=phase, z=z,
                     is_anchor=is_anchor, dist=dist, in_lvspan=in_span,
                     disagree=round(disagree, 4) if disagree == disagree else "",
                     gt_px=int(gt_bool.sum()), img=ip, gt=gp, pf=pfp, pb=pbp))


if __name__ == "__main__":
    main()
