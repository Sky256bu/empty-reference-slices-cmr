#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
73b_export_interp_pseudo.py —— 两锚点 SDF 形状插值伪标签(非SAM densification 对照)
================================================================================
关②对照: 证明低预算下赢的是"SAM传播"而非"随便把中间层填满"。
  仅用 C/B 同样的两个锚点层 GT(lo,hi), 对区间内每层做**符号距离场(SDF)线性插值**填充:
    sdf(m) = edt(m) - edt(~m)   (内部>0)
    s_z    = (1-t)*sdf_lo + t*sdf_hi ,  t=(z-lo)/(hi-lo)
    mask_z = s_z > 0
  锚点层 pi=GT; LV跨度外空。写到 {out}/{split}/pi/{key}_{phase}_z{z:02d}.png (与73同名约定)。
  74 的 arm I 按同名从 imgs/->pi/ 取标签。零SAM、纯scipy、秒级。

用法(与73同参, train必须; test可选):
    python 73b_export_interp_pseudo.py --data_dir /root/autodl-tmp/acdc/train \
        --out_dir /root/autodl-tmp/acdc/seg_lv --split train --lv_label 3 --min_lv_px 15
产物: {out}/{split}/pi/*.png (+打印 插值 self/相邻sanity)
注: out_dir/split 必须与 73 导出的 seg_lv 一致(共用 imgs/gts, 只补 pi)。
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
    log(f"[interp] split={args.split} {len(pats)}例 -> {pi_dir}")

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
                    # sanity: 与真gt的dice(仅统计, 不用于训练)
                    if lv[z].sum() > 0:
                        inter = np.logical_and(m, lv[z]).sum()
                        dsc.append(2 * inter / (m.sum() + lv[z].sum() + 1e-6))
                cv2.imwrite(os.path.join(pi_dir, base), (m.astype(np.uint8) * 255))
                n += 1
    log(f"[interp] 写 {n} 层。插值内部层 vs GT dice: "
        f"mean={np.mean(dsc):.3f} (n={len(dsc)})  <- 对比SAM传播质量")


if __name__ == "__main__":
    main()
