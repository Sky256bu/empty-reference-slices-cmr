#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
81_export_mnms2_pseudo.py -- M&Ms-2 SA short-axis slice export + inter-slice two-anchor propagation (after 73)
================================================================================
Purpose: provide the cross-vendor OOD experiment (82) with a slice-level png dataset + a
  manifest carrying vendor/disease. Uses the **same SAM primitives** as
  73_export_lv_pseudo.py (29_baseline_50 + probe_ted_03_bestof2), with the data source
  switched to M&Ms-2 and two critical changes:
    (1) Label convention: M&Ms uses LV=1/MYO=2/RV=3 (the reverse of ACDC LV=3!) -> --lv_label defaults to 1.
    (2) SAM inter-slice propagation (producing the pf/pb pseudo-labels) is run only for the
        vendors listed in --prop_vendors; the remaining vendors (the test domains) get
        img+gt only (plus anchor/dist/in_span metadata), which saves propagation compute.
        Test-domain evaluation uses the true GT and needs no pseudo-labels.

Data layout (confirmed by probing):
  {root}/dataset/<code>/<code>_SA_ED.nii.gz     (H,W,Z) 3D, ED phase
                        <code>_SA_ED_gt.nii.gz   same shape, GT (1=LV, 2=MYO, 3=RV)
                        <code>_SA_ES(.._gt).nii.gz
  {root}/dataset_information.csv  columns: SUBJECT_CODE (float 1.0..), DISEASE, VENDOR, SCANNER, FIELD
    -> SUBJECT_CODE is aligned to the folder name via int(float(x)).zfill(3); there are
       ~1e6 blank rows, so dropna.

Per stack (one phase of one code), the logic follows 73:
  slices containing the target structure are present, the bottom/top lo/hi are the two anchors;
  fwd propagates forward from lo and bwd backward from hi.
  img.png+gt.png are exported slice by slice; anchor slices take pseudo-label=GT; interior
  slices take pf=fwd/pb=bwd plus disagree=1-dice(fwd,bwd).

Usage:
    cd /root/autodl-tmp/sam3-main
    # 1) smoke first (SIEMENS, first 3 cases only) and check that self-dice ~1.000 at the end:
    python 81_export_mnms2_pseudo.py --root /root/autodl-tmp/mnms/MnM2 \
        --out_dir /root/autodl-tmp/mnms/seg_lv --lv_label 1 \
        --prop_vendors SIEMENS --limit 3
    # 2) once that checks out, run the full set (propagate SIEMENS, export slices for all vendors):
    python 81_export_mnms2_pseudo.py --root /root/autodl-tmp/mnms/MnM2 \
        --out_dir /root/autodl-tmp/mnms/seg_lv --lv_label 1 \
        --prop_vendors SIEMENS --limit 0
Products: {out_dir}/all/{imgs,gts,pf,pb}/*.png + {out_dir}/manifest_mnms2.csv
      manifest columns: key,phase,z,vendor,disease,is_anchor,dist,in_lvspan,do_prop,disagree,gt_px,img,gt,pf,pb
Self-check: anchor self-dice is printed at the end (should be ~1.000); if it is low that is an
      injection or resolution problem: stop, do not proceed to 82.
================================================================================
"""
import argparse, glob, importlib, os, sys, csv
import cv2, numpy as np, nibabel as nib, pandas as pd


def log(m): print(m, flush=True)


def vol_to_u8_frames(vol_hwz, plo=1.0, phi=99.0):
    """(H,W,Z) MRI -> uint8 [Z,H,W], 1/99 percentile stretch (same as 73)."""
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


def load_vendor_map(root):
    """Read dataset_information.csv -> {code(zfill3): (vendor, disease)}."""
    cpath = os.path.join(root, "dataset_information.csv")
    if not os.path.exists(cpath):
        cpath = next(iter(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)), None)
    m = {}
    if cpath is None:
        log("[!] dataset_information.csv not found; the vendor column will be empty"); return m
    df = None
    for sep in (",", ";", "\t"):
        try:
            t = pd.read_csv(cpath, sep=sep, low_memory=False)
            if t.shape[1] > 1: df = t; break
        except Exception:
            pass
    if df is None:
        log(f"[!] cannot read {cpath}"); return m
    cl = {str(x).strip().lower(): x for x in df.columns}
    scol = next((cl[k] for k in cl if k in ("subject_code", "code", "subject")), None)
    vcol = next((cl[k] for k in cl if "vendor" in k), None)
    dcol = next((cl[k] for k in cl if "disease" in k or "patholog" in k), None)
    if scol is None:
        log("[!] csv has no SUBJECT_CODE column"); return m
    df = df.dropna(subset=[scol])
    for _, r in df.iterrows():
        try:
            code = str(int(float(r[scol]))).zfill(3)
        except Exception:
            code = str(r[scol]).strip().zfill(3)
        vend = str(r[vcol]).strip() if vcol else ""
        dis = str(r[dcol]).strip() if dcol else ""
        m[code] = (vend, dis)
    log(f"[vendor] read vendor/disease for {len(m)} cases")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/mnms/MnM2")
    ap.add_argument("--code_dir", default="/root/autodl-tmp/sam3-main")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/mnms/seg_lv")
    ap.add_argument("--lv_label", type=int, default=1, help="M&Ms: LV=1,MYO=2,RV=3")
    ap.add_argument("--min_lv_px", type=int, default=15)
    ap.add_argument("--prop_vendors", default="SIEMENS",
                    help="Comma-separated; run SAM propagation (producing pseudo-labels) for these vendors only. Substring match, case-insensitive.")
    ap.add_argument("--limit", type=int, default=0, help="0=all; use a small number for debugging")
    args = ap.parse_args()

    ds = os.path.join(args.root, "dataset")
    if not os.path.isdir(ds):
        ds = args.root
    for sub in ("imgs", "gts", "pf", "pb"):
        os.makedirs(os.path.join(args.out_dir, "all", sub), exist_ok=True)

    vmap = load_vendor_map(args.root)
    prop_keys = [x.strip().lower() for x in args.prop_vendors.split(",") if x.strip()]

    def need_prop(vendor):
        vl = vendor.lower()
        return any(k in vl for k in prop_keys)

    # ---- SAM primitives (same as 73) ----
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
    cases = sorted([d for d in glob.glob(os.path.join(ds, "*")) if os.path.isdir(d)])
    if args.limit > 0:
        # smoke: take the vendors that need propagation first, to make self-dice easy to check
        cases = sorted(cases, key=lambda c: 0 if need_prop(
            vmap.get(os.path.basename(c), ("", ""))[0]) else 1)[:args.limit]
    log(f"[data] {len(cases)} cases  (LV label={args.lv_label}, propagation vendors={args.prop_vendors})")

    man_path = os.path.join(args.out_dir, "manifest_mnms2.csv")
    fld = ["key", "phase", "z", "vendor", "disease", "is_anchor", "dist",
           "in_lvspan", "do_prop", "disagree", "gt_px", "img", "gt", "pf", "pb"]
    fo = open(man_path, "w", newline=""); wr = csv.DictWriter(fo, fieldnames=fld); wr.writeheader()

    self_dices, n_rows, n_stack, n_prop = [], 0, 0, 0
    for ci, cdir in enumerate(cases, 1):
        code = os.path.basename(cdir)
        vendor, disease = vmap.get(code, ("", ""))
        dop = need_prop(vendor)
        for PH, phase in (("ED", "ed"), ("ES", "es")):
            img_p = os.path.join(cdir, f"{code}_SA_{PH}.nii.gz")
            gt_p = os.path.join(cdir, f"{code}_SA_{PH}_gt.nii.gz")
            if not (os.path.exists(img_p) and os.path.exists(gt_p)):
                continue
            vol = nib.load(img_p).get_fdata()
            gt = np.rint(nib.load(gt_p).get_fdata()).astype(np.int16)
            if vol.ndim != 3:
                log(f"  [skip] {code}/{phase} SA is not 3D shape={vol.shape}"); continue
            frames = vol_to_u8_frames(vol)               # [Z,H,W] u8
            Z, H, W = frames.shape
            gtz = np.transpose(gt, (2, 0, 1))            # [Z,H,W]
            lv = [(gtz[z] == args.lv_label) for z in range(Z)]
            present = [z for z in range(Z) if lv[z].sum() >= args.min_lv_px]
            if len(present) < 3:
                for z in range(Z):
                    _dump_row(wr, args, code, phase, z, frames[z], lv[z], None, None,
                              vendor, disease, is_anchor=0, dist=-1, in_span=0,
                              do_prop=int(dop), disagree=float("nan"))
                    n_rows += 1
                continue
            lo, hi = present[0], present[-1]

            fwd, bwd = {}, {}
            if dop:
                mp4 = P.frames_to_mp4(frames, os.path.join(work, f"{code}_{phase}.mp4"))
                try:
                    st = B.fresh_inject(inner, mp4, lo, lv[lo], 1)
                    fwd = B.propagate_dir(inner, st, lo, reverse=False, span=Z)
                    del st; torch.cuda.empty_cache()
                except Exception as e:
                    log(f"  {code}/{phase} fwd exception {type(e).__name__}: {e}")
                try:
                    st = B.fresh_inject(inner, mp4, hi, lv[hi], 1)
                    bwd = B.propagate_dir(inner, st, hi, reverse=True, span=Z)
                    del st; torch.cuda.empty_cache()
                except Exception as e:
                    log(f"  {code}/{phase} bwd exception {type(e).__name__}: {e}")
                if lo in fwd: self_dices.append(dice(to_shape(fwd[lo], lv[lo].shape), lv[lo]))
                if hi in bwd: self_dices.append(dice(to_shape(bwd[hi], lv[hi].shape), lv[hi]))
                n_prop += 1

            for z in range(Z):
                g = lv[z]
                in_span = 1 if (lo <= z <= hi) else 0
                is_anchor = 1 if (z == lo or z == hi) else 0
                dist = min(z - lo, hi - z) if in_span else -1
                pf = pb = None
                disagree = float("nan")
                if is_anchor:
                    pf = pb = g; disagree = 0.0
                elif in_span and dop:
                    f = to_shape(fwd[z], g.shape) if z in fwd else np.zeros_like(g)
                    b = to_shape(bwd[z], g.shape) if z in bwd else np.zeros_like(g)
                    dd = dice(f, b)
                    disagree = 0.0 if dd != dd else (1.0 - dd)
                    pf, pb = f, b
                _dump_row(wr, args, code, phase, z, frames[z], g, pf, pb,
                          vendor, disease, is_anchor, dist, in_span, int(dop), disagree)
                n_rows += 1
            n_stack += 1
        if ci % 20 == 0: log(f"  ...{ci}/{len(cases)}  (stacks propagated={n_prop})")

    fo.close()
    log(f"\n[out] {n_rows} slices (from {n_stack} stacks, {n_prop} of which were propagated) -> {man_path}")
    if self_dices:
        sd = np.array([x for x in self_dices if x == x])
        log(f"[self-dice] anchors against themselves mean={sd.mean():.3f} min={sd.min():.3f} (expected ~1.000)")
        if sd.mean() < 0.95:
            log("  [!] self-dice is low -> injection or resolution problem; do not proceed to 82, check propagation.")
    else:
        log("[self-dice] none (no stack was propagated this run; check whether --prop_vendors matches)")


def _dump_row(wr, args, key, phase, z, img_u8, gt_bool, pf, pb, vendor, disease,
              is_anchor, dist, in_span, do_prop, disagree):
    base = f"{key}_{phase}_z{z:02d}"
    d = os.path.join(args.out_dir, "all")
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
    wr.writerow(dict(key=key, phase=phase, z=z, vendor=vendor, disease=disease,
                     is_anchor=is_anchor, dist=dist, in_lvspan=in_span, do_prop=do_prop,
                     disagree=round(disagree, 4) if disagree == disagree else "",
                     gt_px=int(gt_bool.sum()), img=ip, gt=gp, pf=pfp, pb=pbp))


if __name__ == "__main__":
    main()
