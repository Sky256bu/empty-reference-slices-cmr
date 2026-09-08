#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
75_eval_seg_and_ef.py -- evaluate the A/B/C student arms on test: stratified Dice + the EF loop
================================================================================
On the held-out test set (never used in training), for each student arm:
  (1) Per-slice Dice against the true GT, stratified by the distance dist to the nearest anchor
      (to see whether B holds up at the far end and whether C collapses).
  (2) EF: method-of-disks (per-slice predicted LV pixels x voxel volume, summed -> EDV/ESV -> EF),
      with agreement against an internally computed true-GT EF: bias/MAE/LoA/Pearson/ICC(2,1) +
      the HFrEF subgroup (GT EF<40).
  All three arms predict **without any GT at test time** (a fair comparison).
Additional reference (not a student, requires no training):
  prop_fwd = the EF of two anchors (GT injected) + fwd interior filling (reading the pf pngs
             exported by 73), i.e. the upstream figure for "our method without distillation",
             on the same footing as the fwd of 72 and now moved to test. At inference it enjoys
             two slices of GT, so it serves as a reference only and no conclusion places it
             alongside the students.

How to read the comparison (load-bearing):
  - seg: B's overall/far-end Dice ~ A and >> C  => "2-anchor propagation ~ dense annotation,
    and propagation is not dispensable".
  - EF : B's ICC/MAE ~ A                        => "2 annotated slices approach the clinical
    measurement of full supervision".
  - A's absolute LV Dice matching the fully supervised ACDC literature (~0.94-0.96) => the ceiling
    is credible, guarding against a "weak baseline" objection.

Usage:
    cd /root/autodl-tmp/sam3-main
    python 75_eval_seg_and_ef.py \
        --mroot /root/autodl-tmp/acdc/seg_lv \
        --data_dir /root/autodl-tmp/acdc/test \
        --ckpt_dir /root/autodl-tmp/acdc/seg_lv/ckpt \
        --arms A,B,C --encoder resnet34 \
        --out_dir /root/autodl-tmp/acdc/seg_lv/eval
Products: {out_dir}/test_per_slice.csv, test_ef_per_case.csv + all comparison tables printed at the end.
================================================================================
"""
import argparse, os
import cv2, numpy as np, pandas as pd, nibabel as nib
import torch, torch.nn as nn, torch.nn.functional as F
import timm


def log(m): print(m, flush=True)


# ---------- model (same as 74) ----------
class ConvBNAct(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNAct(out_ch, out_ch)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv2(self.conv1(torch.cat([x, skip], 1)))


class UNet(nn.Module):
    def __init__(self, encoder_name="resnet34", in_chans=1, num_classes=1, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, features_only=True,
                                          pretrained=pretrained, in_chans=in_chans)
        enc_chs = self.encoder.feature_info.channels(); n_skip = len(enc_chs) - 1
        dec_chs = [256, 128, 64, 32, 16][:n_skip]; skips = enc_chs[:-1][::-1]
        self.blocks = nn.ModuleList(); in_ch = enc_chs[-1]
        for skip_ch, out_ch in zip(skips, dec_chs):
            self.blocks.append(DecoderBlock(in_ch, skip_ch, out_ch)); in_ch = out_ch
        self.head = nn.Conv2d(in_ch, num_classes, 1)
    def forward(self, x):
        H, W = x.shape[-2:]; feats = self.encoder(x)
        f = feats[-1]; skips = feats[:-1][::-1]
        for blk, skip in zip(self.blocks, skips): f = blk(f, skip)
        f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
        return self.head(f)


def dice_np(pred, gt):
    pred = (pred > 0).astype(np.float32); gt = (gt > 0).astype(np.float32)
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else float(2.0 * (pred * gt).sum() / s)


def icc_2_1(gt, pred):
    """ICC(2,1): two-way random, single measurement, absolute agreement. k=2 raters."""
    Y = np.column_stack([np.asarray(gt, float), np.asarray(pred, float)])
    n, k = Y.shape
    if n < 3: return float("nan")
    grand = Y.mean()
    SSR = k * ((Y.mean(1) - grand) ** 2).sum()
    SSC = n * ((Y.mean(0) - grand) ** 2).sum()
    SST = ((Y - grand) ** 2).sum()
    SSE = SST - SSR - SSC
    MSR = SSR / (n - 1); MSC = SSC / (k - 1); MSE = SSE / ((n - 1) * (k - 1))
    den = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    return float((MSR - MSE) / den) if den != 0 else float("nan")


@torch.no_grad()
def predict_all(model, df, root_split, device):
    """Return dict[(key,phase,z)] = pred_px, together with the per-slice dice (against gt)."""
    model.eval()
    pred_px, dice_row = {}, []
    for _, r in df.iterrows():
        img = cv2.imread(os.path.join(root_split, r["img"]), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(root_split, r["gt"]), cv2.IMREAD_GRAYSCALE) > 127
        x = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None].to(device)
        p = (torch.sigmoid(model(x))[0, 0].cpu().numpy() > 0.5)
        pred_px[(r["key"], r["phase"], int(r["z"]))] = int(p.sum())
        dice_row.append(dice_np(p, gt))
    return pred_px, np.array(dice_row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mroot", default="/root/autodl-tmp/acdc/seg_lv")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/acdc/test")
    ap.add_argument("--ckpt_dir", default="/root/autodl-tmp/acdc/seg_lv/ckpt")
    ap.add_argument("--arms", default="A,B,C")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/acdc/seg_lv/eval")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_split = os.path.join(args.mroot, "test")
    arms = args.arms.split(",")

    df = pd.read_csv(os.path.join(args.mroot, "manifest_test.csv"))
    df["pf"] = df["pf"].fillna("")

    # ---- 1) per-arm prediction + per-slice Dice ----
    per_slice = df[["key", "phase", "z", "dist", "in_lvspan", "is_anchor", "gt_px"]].copy()
    pred_px_by_arm = {}
    for arm in arms:
        ckpt = os.path.join(args.ckpt_dir, f"student_{arm}.pt")
        if not os.path.exists(ckpt):
            log(f"[skip] missing {ckpt}"); continue
        st = torch.load(ckpt, map_location=device)
        model = UNet(st.get("encoder", args.encoder), 1, 1, pretrained=False).to(device)
        model.load_state_dict(st["model"])
        pp, dr = predict_all(model, df, root_split, device)
        pred_px_by_arm[arm] = pp
        per_slice[f"dice_{arm}"] = dr
        per_slice[f"predpx_{arm}"] = [pp[(r.key, r.phase, int(r.z))]
                                      for r in df.itertuples()]
        del model; torch.cuda.empty_cache()
        log(f"[arm {arm}] prediction complete (best-ckpt val_dice={st.get('val_dice'):.4f})")

    arms = [a for a in arms if a in pred_px_by_arm]     # keep only the arms that loaded successfully
    if not arms:
        log("[!] no usable arm (all ckpts missing); exiting."); return

    per_slice.to_csv(os.path.join(args.out_dir, "test_per_slice.csv"), index=False)

    # stratified Dice (LV-present only: gt_px>0; and in_lvspan==1 so that dist exists)
    pres = per_slice[per_slice["gt_px"] > 0]
    log("\n" + "=" * 74)
    log("  test per-slice Dice (LV-present) stratified by dist  [arms: " + " ".join(arms) + "]")
    log("=" * 74)
    log("  dist |    n | " + " | ".join(f"{a:>6s}" for a in arms))
    for dist in sorted(pres[pres["in_lvspan"] == 1]["dist"].unique()):
        sub = pres[(pres["in_lvspan"] == 1) & (pres["dist"] == dist)]
        cells = [f"{sub[f'dice_{a}'].mean():.3f}" if f"dice_{a}" in sub else "NA" for a in arms]
        log(f"  {int(dist):>4} | {len(sub):>4} | " + " | ".join(f"{c:>6s}" for c in cells))
    log("  " + "-" * 60)
    ov = [f"{pres[f'dice_{a}'].mean():.3f}" if f"dice_{a}" in pres else "NA" for a in arms]
    log(f"  overall |{len(pres):>5} | " + " | ".join(f"{c:>6s}" for c in ov))
    log("  (expected: B~A and B>>C at the far end; A in absolute terms should be ~the fully supervised LV of the ACDC literature, ~0.94-0.96)")

    # ---- 2) the EF loop ----
    # budget prop_fwd pixels per slice: anchors=gt_px, interior in-span=pf pixels, outside the span=0
    def pf_px(r):
        if int(r["is_anchor"]) == 1:
            return int(r["gt_px"])
        if int(r["in_lvspan"]) == 1 and str(r["pf"]):
            m = cv2.imread(os.path.join(root_split, r["pf"]), cv2.IMREAD_GRAYSCALE)
            return int((m > 127).sum()) if m is not None else 0
        return 0
    df["_propfwd_px"] = df.apply(pf_px, axis=1)

    keys = sorted(df["key"].unique())
    sources = ["gt", "propfwd"] + arms
    rows = []
    for key in keys:
        vox, vol = {}, {s: {} for s in sources}   # vol[src][phase] = mL
        ok = True
        for phase in ("ed", "es"):
            sub = df[(df["key"] == key) & (df["phase"] == phase)]
            if sub.empty: ok = False; break
            gp = os.path.join(args.data_dir, key, f"{key}_sax_{phase}_gt.nii.gz")
            if not os.path.exists(gp): ok = False; break
            vox_ml = abs(np.linalg.det(nib.load(gp).affine[:3, :3])) / 1000.0
            vol["gt"][phase] = float(sub["gt_px"].sum()) * vox_ml
            vol["propfwd"][phase] = float(sub["_propfwd_px"].sum()) * vox_ml
            for a in arms:
                pxs = sum(pred_px_by_arm[a][(key, phase, int(z))] for z in sub["z"])
                vol[a][phase] = float(pxs) * vox_ml
        if not ok: continue
        row = dict(key=key)
        for s in sources:
            edv, esv = vol[s]["ed"], vol[s]["es"]
            ef = 100.0 * (edv - esv) / edv if edv > 1e-6 else np.nan
            row[f"edv_{s}"] = round(edv, 2); row[f"esv_{s}"] = round(esv, 2)
            row[f"ef_{s}"] = round(ef, 2)
        rows.append(row)

    ef = pd.DataFrame(rows)
    ef.to_csv(os.path.join(args.out_dir, "test_ef_per_case.csv"), index=False)
    log(f"\n[EF] {len(ef)} cases -> test_ef_per_case.csv")
    if ef.empty: return

    gt = ef["ef_gt"].values
    log("\n" + "=" * 74)
    log("  test EF vs the internally computed true-GT EF  (bias=pred-gt %, LoA=bias+/-1.96 sigma)")
    log("=" * 74)
    log(f"  {'source':10s} | {'bias':>6s} {'MAE':>6s} {'LoA width':>9s} {'r':>6s} {'ICC':>6s}")
    for s in ["propfwd"] + arms:
        pe = ef[f"ef_{s}"].values; d = pe - gt; m = np.isfinite(d)
        r = np.corrcoef(pe[m], gt[m])[0, 1] if m.sum() > 3 else np.nan
        icc = icc_2_1(gt[m], pe[m])
        tag = s + ("(reference, uses 2 GT slices)" if s == "propfwd" else "")
        log(f"  {tag:10s} | {d[m].mean():6.2f} {np.abs(d[m]).mean():6.2f} "
            f"{2*1.96*d[m].std():7.2f} {r:6.3f} {icc:6.3f}")

    hf = ef[ef["ef_gt"] < 40]
    log(f"\n  HFrEF subgroup (GT EF<40, n={len(hf)}):")
    if len(hf) > 3:
        for s in ["propfwd"] + arms:
            d = hf[f"ef_{s}"].values - hf["ef_gt"].values; m = np.isfinite(d)
            log(f"    {s:8s} bias={d[m].mean():+.2f} MAE={np.abs(d[m]).mean():.2f}")
    log("\n[reading] look at column B: if seg Dice ~ A with the far end >> C, and EF ICC/MAE ~ A, the load-bearing claim holds (2 slices approach full supervision).")
    log("       If A's absolute Dice is ~0.94+, the ceiling is credible; propfwd is a reference only (it uses 2 GT slices at inference).")


if __name__ == "__main__":
    main()
