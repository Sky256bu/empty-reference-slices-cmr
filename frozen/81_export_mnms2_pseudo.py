#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
81_export_mnms2_pseudo.py —— M&Ms-2 SA 短轴切片导出 + 层间两锚点传播(仿 73)
================================================================================
目的: 为跨厂商 OOD 实验(82) 提供切片级 png 数据集 + manifest(带 vendor/disease)。
  与 73_export_lv_pseudo.py **同一套 SAM 原语**(29_baseline_50 + probe_ted_03_bestof2),
  只把数据源换成 M&Ms-2, 并加两条关键改动:
    (1) 标签约定: M&Ms LV=1/MYO=2/RV=3 (与 ACDC LV=3 相反!) -> --lv_label 默认 1。
    (2) 只对 --prop_vendors 里的厂商跑 SAM 层间传播(出 pf/pb 伪标签);
        其余厂商(测试域)仅导 img+gt(+锚点/dist/in_span 元信息), 省传播算力。
        测试域评估用真 GT, 不需要伪标签。

数据布局(探测确认):
  {root}/dataset/<code>/<code>_SA_ED.nii.gz     (H,W,Z) 3D, ED 相位
                        <code>_SA_ED_gt.nii.gz   同形 GT(1=LV,2=MYO,3=RV)
                        <code>_SA_ES(.._gt).nii.gz
  {root}/dataset_information.csv  列: SUBJECT_CODE(浮点1.0..), DISEASE, VENDOR, SCANNER, FIELD
    -> SUBJECT_CODE 用 int(float(x)).zfill(3) 对齐文件夹名; 有~1e6 空行, dropna。

每 stack(一个 code 的一个相位) 逻辑同 73:
  含目标结构的层 present, 底/顶 lo/hi 为两锚点; fwd 从 lo 前向、bwd 从 hi 反向层间传播。
  逐层导 img.png+gt.png; 锚点层伪标=GT; 内部层 pf=fwd/pb=bwd + disagree=1-dice(fwd,bwd)。

用法:
    cd /root/autodl-tmp/sam3-main
    # 1) 先 smoke(只 SIEMENS 前3例), 看结尾 self-dice≈1.000:
    python 81_export_mnms2_pseudo.py --root /root/autodl-tmp/mnms/MnM2 \
        --out_dir /root/autodl-tmp/mnms/seg_lv --lv_label 1 \
        --prop_vendors SIEMENS --limit 3
    # 2) 对上后全量(SIEMENS 传播 + 全厂商导切片):
    python 81_export_mnms2_pseudo.py --root /root/autodl-tmp/mnms/MnM2 \
        --out_dir /root/autodl-tmp/mnms/seg_lv --lv_label 1 \
        --prop_vendors SIEMENS --limit 0
产物: {out_dir}/all/{imgs,gts,pf,pb}/*.png + {out_dir}/manifest_mnms2.csv
      manifest 列: key,phase,z,vendor,disease,is_anchor,dist,in_lvspan,do_prop,disagree,gt_px,img,gt,pf,pb
自检: 结尾打印锚点 self-dice(应≈1.000); 偏低=注入/分辨率问题, 停下别进 82。
================================================================================
"""
import argparse, glob, importlib, os, sys, csv
import cv2, numpy as np, nibabel as nib, pandas as pd


def log(m): print(m, flush=True)


def vol_to_u8_frames(vol_hwz, plo=1.0, phi=99.0):
    """(H,W,Z) MRI -> uint8 [Z,H,W], 1/99 百分位拉伸(同 73)。"""
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
    """读 dataset_information.csv -> {code(zfill3): (vendor, disease)}。"""
    cpath = os.path.join(root, "dataset_information.csv")
    if not os.path.exists(cpath):
        cpath = next(iter(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)), None)
    m = {}
    if cpath is None:
        log("[!] 找不到 dataset_information.csv, vendor 列将为空"); return m
    df = None
    for sep in (",", ";", "\t"):
        try:
            t = pd.read_csv(cpath, sep=sep, low_memory=False)
            if t.shape[1] > 1: df = t; break
        except Exception:
            pass
    if df is None:
        log(f"[!] 读不了 {cpath}"); return m
    cl = {str(x).strip().lower(): x for x in df.columns}
    scol = next((cl[k] for k in cl if k in ("subject_code", "code", "subject")), None)
    vcol = next((cl[k] for k in cl if "vendor" in k), None)
    dcol = next((cl[k] for k in cl if "disease" in k or "patholog" in k), None)
    if scol is None:
        log("[!] csv 无 SUBJECT_CODE 列"); return m
    df = df.dropna(subset=[scol])
    for _, r in df.iterrows():
        try:
            code = str(int(float(r[scol]))).zfill(3)
        except Exception:
            code = str(r[scol]).strip().zfill(3)
        vend = str(r[vcol]).strip() if vcol else ""
        dis = str(r[dcol]).strip() if dcol else ""
        m[code] = (vend, dis)
    log(f"[vendor] 读到 {len(m)} 例的 vendor/disease")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/mnms/MnM2")
    ap.add_argument("--code_dir", default="/root/autodl-tmp/sam3-main")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/mnms/seg_lv")
    ap.add_argument("--lv_label", type=int, default=1, help="M&Ms: LV=1,MYO=2,RV=3")
    ap.add_argument("--min_lv_px", type=int, default=15)
    ap.add_argument("--prop_vendors", default="SIEMENS",
                    help="逗号分隔; 只对这些厂商跑SAM传播(出伪标签)。子串匹配, 大小写不敏感。")
    ap.add_argument("--limit", type=int, default=0, help="0=全部; 调试用小数")
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

    # ---- SAM 原语(同 73) ----
    code_dir = os.path.abspath(args.code_dir)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    os.chdir(code_dir)
    B = importlib.import_module("29_baseline_50")
    P = importlib.import_module("probe_ted_03_bestof2")
    import torch
    assert torch.cuda.is_available(), "需要 CUDA"
    from types import SimpleNamespace
    predictor = B.build(SimpleNamespace(ckpt=B.CKPT, bpe=B.BPE))
    B.patch_load_video_frames()
    inner = predictor.model.tracker.model
    B.graft_backbone(predictor, inner); B.shim_backbone_forward(inner)
    B.guard_highres_convs(inner); B.patch_preflight(inner)

    work = os.path.join(args.out_dir, "_w"); os.makedirs(work, exist_ok=True)
    cases = sorted([d for d in glob.glob(os.path.join(ds, "*")) if os.path.isdir(d)])
    if args.limit > 0:
        # smoke: 优先取需要传播的厂商, 便于验 self-dice
        cases = sorted(cases, key=lambda c: 0 if need_prop(
            vmap.get(os.path.basename(c), ("", ""))[0]) else 1)[:args.limit]
    log(f"[data] {len(cases)} 例  (LV label={args.lv_label}, 传播厂商={args.prop_vendors})")

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
                log(f"  [skip] {code}/{phase} SA 非3D shape={vol.shape}"); continue
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
                    log(f"  {code}/{phase} fwd 异常 {type(e).__name__}: {e}")
                try:
                    st = B.fresh_inject(inner, mp4, hi, lv[hi], 1)
                    bwd = B.propagate_dir(inner, st, hi, reverse=True, span=Z)
                    del st; torch.cuda.empty_cache()
                except Exception as e:
                    log(f"  {code}/{phase} bwd 异常 {type(e).__name__}: {e}")
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
        if ci % 20 == 0: log(f"  ...{ci}/{len(cases)}  (已传播 stack={n_prop})")

    fo.close()
    log(f"\n[out] {n_rows} 层 (来自 {n_stack} stack, 其中 {n_prop} 个跑了传播) -> {man_path}")
    if self_dices:
        sd = np.array([x for x in self_dices if x == x])
        log(f"[self-dice] 锚点自身 mean={sd.mean():.3f} min={sd.min():.3f} (应≈1.000)")
        if sd.mean() < 0.95:
            log("  [!] self-dice 偏低 -> 注入/分辨率问题, 先别进82, 查传播。")
    else:
        log("[self-dice] 无(本次未传播任何 stack, 检查 --prop_vendors 是否匹配)")


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
