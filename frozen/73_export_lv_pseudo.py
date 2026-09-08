#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
73_export_lv_pseudo.py —— 把 ACDC 切片导成"学生数据集" + 两锚点传播伪标签
================================================================================
目的(承重实验的地基):
  为 A/B/C 三臂学生(full-dense / 2-anchor伪标签 / sparse-only)提供**同一份**切片级
  png 数据 + 一份 manifest.csv。三臂只是对 manifest 做不同的行/标签选择, 保证
  "同架构同recipe, 只换标签来源"的受控对照。

对每个 patient 的每个相位(ed/es):
  - 读 img(百分位拉伸u8) + gt(0=bg 1=RV 2=MYO 3=LV, 本脚本只取 LV=3)。
  - 找含 LV 的层, 底/顶两层 lo/hi 作锚点(临床两片标注); SAM3.1 双向层间传播:
      fwd = 从底锚 lo 前向;  bwd = 从顶锚 hi 反向。
  - **每一层** z 都导出: img.png + gt.png(0/255)。
    * 锚点层(lo,hi): 伪标签 = GT(临床已标)。
    * 内部层: pseudo_fwd = fwd[z] , pseudo_bwd = bwd[z] , disagree = 1-dice(fwd,bwd)。
    * LV 跨度外的层(apex/base 无 LV): gt 空掩码, 无伪标签(pf/pb 留空白png)。
  - manifest 每层一行, 记 split/key/phase/z/is_anchor/dist/in_lvspan/disagree/gt_px + 4张png相对路径。

三臂读法(下游 74 训练脚本按此过滤):
  A(full-dense): 用 gt.png, 所有层(含LV跨度外的空层, 学生学会"这里没有LV")。
  B(2-anchor伪标签): 锚点层用 gt.png; LV跨度内内部层用 pseudo_fwd.png; 跨度外用空 gt。
  C(sparse-only): 只用 is_anchor==1 的层的 gt.png。
  B+(QC加权, 可选后续): 同 B, 但内部层按 disagree 加权/筛选(本脚本已存 disagree, 零额外传播)。

复用 29_baseline_50 加载链 + propagate_dir + probe_ted_03_bestof2.frames_to_mp4,
与 70/72 **完全同一套原语**, 不新摸 API。

用法:
    cd /root/autodl-tmp/sam3-main
    # train(必须, 出伪标签):
    python 73_export_lv_pseudo.py \
        --data_dir /root/autodl-tmp/acdc/train \
        --out_dir  /root/autodl-tmp/acdc/seg_lv \
        --split train --limit 5          # 先 smoke, 对上 self-dice≈1 再 --limit 0
    # test(eval用, 也跑传播只为存 disagree; 不需要可无所谓):
    python 73_export_lv_pseudo.py \
        --data_dir /root/autodl-tmp/acdc/test \
        --out_dir  /root/autodl-tmp/acdc/seg_lv \
        --split test --limit 0
产物: {out_dir}/{split}/imgs|gts|pf|pb/*.png  +  {out_dir}/manifest_{split}.csv
自检: 结尾打印 self-dice(锚点自身应≈1.000)。对不上=注入/分辨率问题, 停。
================================================================================
"""
import argparse, glob, importlib, os, sys, csv
import cv2, numpy as np, nibabel as nib

try:
    from scipy.ndimage import distance_transform_edt  # noqa (与70/72同环境)
except Exception:
    pass


def log(m): print(m, flush=True)


def vol_to_u8_frames(vol_hwz, plo=1.0, phi=99.0):
    """(H,W,Z) MRI -> uint8 [Z,H,W], 1/99 百分位拉伸(不能 /256)。"""
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
    ap.add_argument("--data_dir", required=True, help="acdc/train 或 acdc/test")
    ap.add_argument("--code_dir", default="/root/autodl-tmp/sam3-main")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/acdc/seg_lv")
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--lv_label", type=int, default=3)
    ap.add_argument("--min_lv_px", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0, help="0=全部")
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
    assert torch.cuda.is_available(), "需要 CUDA"

    from types import SimpleNamespace
    predictor = B.build(SimpleNamespace(ckpt=B.CKPT, bpe=B.BPE))
    B.patch_load_video_frames()
    inner = predictor.model.tracker.model
    B.graft_backbone(predictor, inner); B.shim_backbone_forward(inner)
    B.guard_highres_convs(inner); B.patch_preflight(inner)

    work = os.path.join(args.out_dir, "_w"); os.makedirs(work, exist_ok=True)
    pats = sorted(glob.glob(os.path.join(args.data_dir, "patient*")))
    if args.limit > 0: pats = pats[: args.limit]
    log(f"[data] split={args.split}  {len(pats)} 例  (LV label={args.lv_label})")

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
                # LV 太少: 仍导 img/gt(全空或近空), 但无锚/无伪标签, 只进 A 的负样本
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
                log(f"  {key}/{phase} fwd 异常 {type(e).__name__}: {e}"); fwd = {}
            try:
                st = B.fresh_inject(inner, mp4, hi, lv[hi], 1)
                bwd = B.propagate_dir(inner, st, hi, reverse=True, span=Z)
                del st; torch.cuda.empty_cache()
            except Exception as e:
                log(f"  {key}/{phase} bwd 异常 {type(e).__name__}: {e}"); bwd = {}

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
                    pf = pb = g            # 锚点伪标签=GT
                    disagree = 0.0
                _dump_row(wr, args, key, phase, z, frames[z], g, pf, pb,
                          is_anchor, dist, in_span, disagree)
                n_rows += 1
            n_stack += 1
        if pi % 10 == 0: log(f"  ...{pi}/{len(pats)}")

    fo.close()
    log(f"\n[out] {n_rows} 层 (来自 {n_stack} 个可传播 stack) -> {man_path}")
    if self_dices:
        sd = np.array([x for x in self_dices if x == x])
        log(f"[self-dice] 锚点自身 mean={sd.mean():.3f} min={sd.min():.3f} (应≈1.000)")
        if sd.mean() < 0.95:
            log("  [!] self-dice 偏低 -> 注入/分辨率问题, 先别训学生, 查传播。")


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
