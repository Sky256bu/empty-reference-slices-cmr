#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
81b_export_mnms2_interp.py —— M&Ms-2 两锚点 SDF 插值伪标签(I 臂, 跨厂商对照)
================================================================================
关②跨厂商版: 证明 82 里低预算下赢的是"SAM 层间传播(B)"而非"随便把中间层填满(I)"。
  与 73b 同一套 SDF 形状插值, 只把数据源换成 M&Ms-2、标签约定换成 LV=1:
    仅用两锚点层 GT(lo,hi), 区间内每层线性插 SDF -> mask_z=(s_z>0); 锚点层=GT。
  写到 {out_dir}/all/pi/{key}_{phase}_z{z:02d}.png (与 81 的 imgs/ 同名约定)。
  74 的 arm I 按 imgs/->pi/ 取标签, 不读 manifest 列, 故本脚本不动 manifest。
  零 SAM、纯 scipy、CPU、秒级。

关键: 只有 SIEMENS 训练池的**内部层**在 82 arm I 训练时会被读; 但为鲁棒(将来换训练厂商
      也不炸), 这里对**所有** LV跨度≥3 的 stack 都导 pi。测试域评估用真 GT, 不碰 pi。

用法:
    cd /root/autodl-tmp/sam3-main
    python 81b_export_mnms2_interp.py --root /root/autodl-tmp/mnms/MnM2 \
        --out_dir /root/autodl-tmp/mnms/seg_lv --lv_label 1 --min_lv_px 15
产物: {out_dir}/all/pi/*.png
自检: 结尾打印 **插值内部层 vs GT 的 dice(整体+按厂商)** —— 与 81 的 SAM self-dice 对比,
      若插值 dice 明显低(如 ACDC 上 LV~0.645), 就预告 I 臂会被毒化(densification 有害)。
跑完接: for N in 10 25 50 0; do for S in 0 1 2; do
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

    # 复用 81 的 vendor 映射(仅用于 sanity 分厂商统计)
    code_dir = os.path.abspath(args.code_dir)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    try:
        E81 = importlib.import_module("81_export_mnms2_pseudo")
        vmap = E81.load_vendor_map(args.root)
    except Exception as e:
        log(f"[!] 复用 81.load_vendor_map 失败({type(e).__name__}), 厂商统计将为空: {e}")
        vmap = {}

    ds = os.path.join(args.root, "dataset")
    if not os.path.isdir(ds):
        ds = args.root
    cases = sorted([d for d in glob.glob(os.path.join(ds, "*")) if os.path.isdir(d)])
    if args.limit > 0:
        cases = cases[:args.limit]
    log(f"[interp] {len(cases)} 例 -> {pi_dir}  (LV label={args.lv_label})")

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
                    if lv[z].sum() > 0:                    # 仅统计, 不入训练
                        dd = dice(m, lv[z])
                        dsc_all.append(dd)
                        dsc_by_v.setdefault(vend, []).append(dd)
                cv2.imwrite(os.path.join(pi_dir, base), (m.astype(np.uint8) * 255))
                n_png += 1
        if ci % 50 == 0:
            log(f"  ...{ci}/{len(cases)}")

    log(f"\n[out] 写 {n_png} 层 pi -> {pi_dir}")
    if dsc_all:
        a = np.array(dsc_all)
        log(f"[interp vs GT] 内部层 dice: 整体 mean={a.mean():.3f} (n={len(a)})")
        for v in ["SIEMENS", "Philips", "GE"]:
            if v in dsc_by_v:
                b = np.array(dsc_by_v[v])
                log(f"    {v:>8s}: mean={b.mean():.3f}  (n={len(b)})")
        log("  [判读] 与 81 的 SAM self/传播质量对比: 插值 dice 越低 => I 臂越会被毒化,")
        log("         '低预算下赢的是 SAM 传播、不是随便填满' 的跨厂商反证越硬。")
    else:
        log("[!] 无内部层可统计(检查 lv_label/min_lv_px 或数据路径)")


if __name__ == "__main__":
    main()
