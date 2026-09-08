#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
75_eval_seg_and_ef.py —— test 上评估 A/B/C 三臂学生: 分层Dice + EF闭环
================================================================================
在 held-out test(从未参与训练)上, 对每个学生臂:
  (1) 逐层 Dice vs 真 GT, 按到最近锚点距离 dist 分层(看 B 远端是否守住, C 是否崩)。
  (2) EF: method-of-disks(逐层预测LV像素 × 体素体积求和 -> EDV/ESV -> EF),
      对自算真 GT EF 出一致性: bias/MAE/LoA/Pearson/ICC(2,1) + HFrEF(GT EF<40)亚组。
  三臂 test 时**都零GT纯预测**(公平)。
额外参照(非学生, 无需训练):
  prop_fwd = 两锚点(注GT) + fwd 内部填充(读 73 导出的 pf png) 的 EF —— 即"我们方法不蒸馏"的上游数,
             与 72 的 fwd 同口径, 现在搬到 test。它推理时占了2层GT便宜, 故只作参照, 不与学生同列下结论。

对照读法(承重):
  - seg: B 的整体/远端 Dice ≈ A 且 ≫ C  => "2锚点传播≈稠密标注, 传播不可省"。
  - EF : B 的 ICC/MAE ≈ A                => "2层标注逼近全监督的临床测量"。
  - A 的绝对 LV Dice 对齐 ACDC 文献全监督(~0.94-0.96) => 天花板可信, 防"基线弱"。

用法:
    cd /root/autodl-tmp/sam3-main
    python 75_eval_seg_and_ef.py \
        --mroot /root/autodl-tmp/acdc/seg_lv \
        --data_dir /root/autodl-tmp/acdc/test \
        --ckpt_dir /root/autodl-tmp/acdc/seg_lv/ckpt \
        --arms A,B,C --encoder resnet34 \
        --out_dir /root/autodl-tmp/acdc/seg_lv/eval
产物: {out_dir}/test_per_slice.csv, test_ef_per_case.csv + 结尾打印全部对照表。
================================================================================
"""
import argparse, os
import cv2, numpy as np, pandas as pd, nibabel as nib
import torch, torch.nn as nn, torch.nn.functional as F
import timm


def log(m): print(m, flush=True)


# ---------- 模型(与 74 一致) ----------
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
    """ICC(2,1): 双向随机, 单测量, 绝对一致。k=2 raters。"""
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
    """返回 dict[(key,phase,z)] = pred_px, 以及逐层 dice(对gt)。"""
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

    # ---- 1) 各臂预测 + 逐层 Dice ----
    per_slice = df[["key", "phase", "z", "dist", "in_lvspan", "is_anchor", "gt_px"]].copy()
    pred_px_by_arm = {}
    for arm in arms:
        ckpt = os.path.join(args.ckpt_dir, f"student_{arm}.pt")
        if not os.path.exists(ckpt):
            log(f"[skip] 缺 {ckpt}"); continue
        st = torch.load(ckpt, map_location=device)
        model = UNet(st.get("encoder", args.encoder), 1, 1, pretrained=False).to(device)
        model.load_state_dict(st["model"])
        pp, dr = predict_all(model, df, root_split, device)
        pred_px_by_arm[arm] = pp
        per_slice[f"dice_{arm}"] = dr
        per_slice[f"predpx_{arm}"] = [pp[(r.key, r.phase, int(r.z))]
                                      for r in df.itertuples()]
        del model; torch.cuda.empty_cache()
        log(f"[arm {arm}] 预测完毕 (best-ckpt val_dice={st.get('val_dice'):.4f})")

    arms = [a for a in arms if a in pred_px_by_arm]     # 只保留成功加载的臂
    if not arms:
        log("[!] 无可用臂(ckpt全缺), 退出。"); return

    per_slice.to_csv(os.path.join(args.out_dir, "test_per_slice.csv"), index=False)

    # 分层 Dice(只在 LV-present: gt_px>0; 且 in_lvspan==1 有 dist)
    pres = per_slice[per_slice["gt_px"] > 0]
    log("\n" + "=" * 74)
    log("  test 逐层 Dice(LV-present) 按 dist 分层  [臂: " + " ".join(arms) + "]")
    log("=" * 74)
    log("  dist |    n | " + " | ".join(f"{a:>6s}" for a in arms))
    for dist in sorted(pres[pres["in_lvspan"] == 1]["dist"].unique()):
        sub = pres[(pres["in_lvspan"] == 1) & (pres["dist"] == dist)]
        cells = [f"{sub[f'dice_{a}'].mean():.3f}" if f"dice_{a}" in sub else "NA" for a in arms]
        log(f"  {int(dist):>4} | {len(sub):>4} | " + " | ".join(f"{c:>6s}" for c in cells))
    log("  " + "-" * 60)
    ov = [f"{pres[f'dice_{a}'].mean():.3f}" if f"dice_{a}" in pres else "NA" for a in arms]
    log(f"  整体 |{len(pres):>5} | " + " | ".join(f"{c:>6s}" for c in ov))
    log("  (期望: B≈A 且远端 B≫C; A 绝对值应≈ACDC文献全监督 LV~0.94-0.96)")

    # ---- 2) EF 闭环 ----
    # 预算 prop_fwd 每层像素: 锚点=gt_px, 内部in-span=pf像素, 跨度外=0
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
    log(f"\n[EF] {len(ef)} 例 -> test_ef_per_case.csv")
    if ef.empty: return

    gt = ef["ef_gt"].values
    log("\n" + "=" * 74)
    log("  test EF vs 自算真GT EF  (bias=pred-gt %, LoA=bias±1.96σ)")
    log("=" * 74)
    log(f"  {'源':10s} | {'bias':>6s} {'MAE':>6s} {'LoA宽':>7s} {'r':>6s} {'ICC':>6s}")
    for s in ["propfwd"] + arms:
        pe = ef[f"ef_{s}"].values; d = pe - gt; m = np.isfinite(d)
        r = np.corrcoef(pe[m], gt[m])[0, 1] if m.sum() > 3 else np.nan
        icc = icc_2_1(gt[m], pe[m])
        tag = s + ("(参照,占2层GT)" if s == "propfwd" else "")
        log(f"  {tag:10s} | {d[m].mean():6.2f} {np.abs(d[m]).mean():6.2f} "
            f"{2*1.96*d[m].std():7.2f} {r:6.3f} {icc:6.3f}")

    hf = ef[ef["ef_gt"] < 40]
    log(f"\n  HFrEF 亚组 (GT EF<40, n={len(hf)}):")
    if len(hf) > 3:
        for s in ["propfwd"] + arms:
            d = hf[f"ef_{s}"].values - hf["ef_gt"].values; m = np.isfinite(d)
            log(f"    {s:8s} bias={d[m].mean():+.2f} MAE={np.abs(d[m]).mean():.2f}")
    log("\n[判读] 看 B 列: seg Dice≈A且远端≫C, EF ICC/MAE≈A => 承重成立(2层逼近全监督)。")
    log("       A 绝对 Dice 若≈0.94+ => 天花板可信; propfwd 仅参照(推理占2层GT)。")


if __name__ == "__main__":
    main()
