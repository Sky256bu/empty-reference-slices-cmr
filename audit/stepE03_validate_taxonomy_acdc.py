#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stepE03_validate_taxonomy_acdc.py
=================================
把 stepE02 的空层假阳分类接到 **ACDC 真实掩膜** 上，做两件事：

  (A) 门控验证 —— 逐病例复现冻结的 `stack_dice_acdc_clean.csv:empty_fp` 列。
      对不上就中止，不产出任何设计期读数。
  (B) 设计期读数 —— 仅在 (A) 通过后运行：connected / isolated / beyond /
      interior 与 lcc_removable_frac 的分布，26 与 6 邻域各一份。

身份
----
(A) 为**代码正确性自证**，与 stepE01 的 S1/S2 同性质。
(B) 为 **design-phase**（发现集 ACDC，修正案 §1.1 授权），
    禁止进入当前手稿，禁止与任何冻结结果做统计比较。

⚠️ (B) 一旦运行，"LCC 能否清除多数空层假阳"在 ACDC 上的答案即被看见。
   ACDC 已按修正案 D-E6 降为发现集，故此为允许；但该答案**此后不得**
   被表述为对该问题的确认性证据。

只读输入，只往 --out-dir 写。不训练，不改任何冻结文件。

依赖：numpy scipy pandas nibabel

用法
----
  # 第一步：只探路，不算数（强烈建议先跑这个）
  python3 stepE03_validate_taxonomy_acdc.py --discover \
      --gt-root  /root/autodl-tmp/acdc/... \
      --pred-root /root/autodl-tmp/acdc/...

  # 第二步：门控验证（可先用 --limit 20 试跑）
  python3 stepE03_validate_taxonomy_acdc.py \
      --gt-root /path --pred-root /path \
      --frozen-csv /path/stack_dice_acdc_clean.csv \
      --out-dir /root/autodl-tmp/design_phase_quarantine --limit 20

  # 第三步：全量 + 设计期读数
  python3 ... （去掉 --limit，加 --readout）
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError:
    nib = None


# ------------------------------------------------------------------ 工具
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_taxonomy(path):
    if not os.path.exists(path):
        sys.exit(f"[ABORT] 找不到 stepE02_fp_taxonomy.py：{path}")
    spec = importlib.util.spec_from_file_location("taxo", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, sha256(path)


def find_niftis(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith((".nii", ".nii.gz")):
                out.append(os.path.join(dp, fn))
    return sorted(out)


PAT_KEY = re.compile(r"(patient\d{3})", re.I)
PAT_FRAME = re.compile(r"frame(\d+)", re.I)
PAT_ARM = re.compile(r"(?:^|[_/])arm[_-]?([A-Z])(?:[_/]|$)")
PAT_N = re.compile(r"(?:^|[_/])N[_-]?(\d+)(?:[_/]|$)")
PAT_SEED = re.compile(r"(?:^|[_/])s(?:eed)?[_-]?(\d+)(?:[_/]|$)")


def parse_meta(path):
    d = {}
    for pat, k, cast in [(PAT_KEY, "key", str), (PAT_FRAME, "frame", int),
                         (PAT_ARM, "arm", str), (PAT_N, "N", int),
                         (PAT_SEED, "seed", int)]:
        m = pat.search(path)
        if m:
            d[k] = cast(m.group(1))
    return d


# ------------------------------------------------------------------ 探路模式
def discover(args):
    print("=" * 70)
    print("DISCOVER 模式：只列清单，不做任何计算")
    print("=" * 70)
    for name, root in [("GT", args.gt_root), ("PRED", args.pred_root)]:
        if not root:
            print(f"\n[{name}] 未提供 root，跳过")
            continue
        if not os.path.isdir(root):
            print(f"\n[{name}] 目录不存在：{root}")
            continue
        fs = find_niftis(root)
        print(f"\n[{name}] {root}")
        print(f"  NIfTI 文件数：{len(fs)}")
        for f in fs[:8]:
            print("   ", os.path.relpath(f, root))
        if len(fs) > 8:
            print(f"    ... 另有 {len(fs)-8} 个")
        if fs and nib is not None:
            im = nib.load(fs[0])
            arr = np.asanyarray(im.dataobj)
            print(f"  样例 shape={arr.shape} dtype={arr.dtype}")
            print(f"  样例 唯一值={np.unique(arr)[:12]}")
            print(f"  样例 zooms={im.header.get_zooms()}")
            print(f"  样例 解析元信息={parse_meta(fs[0])}")
    if args.frozen_csv and os.path.exists(args.frozen_csv):
        d = pd.read_csv(args.frozen_csv)
        print(f"\n[FROZEN] {args.frozen_csv}")
        print(f"  sha256={sha256(args.frozen_csv)}")
        print(f"  shape={d.shape}  arms={sorted(d['arm'].unique())}")
        print(f"  N={sorted(d['N'].unique())}  seeds={sorted(d['seed'].unique())}")
        print(f"  phase={sorted(d['phase'].unique())}")
        print(f"  key 样例={sorted(d['key'].unique())[:5]}")
        print(f"  empty_fp 统计: >0 比例={float((d['empty_fp']>0).mean()):.3f}"
              f"  max={float(d['empty_fp'].max()):.0f}")
    print("\n把上面的输出发回，我据此写死路径与标签映射规则。")


# ------------------------------------------------------------------ 主验证
def axis_probe(taxo, pred, gt, spacing):
    """三个轴各试一次，返回每个轴的 empty_fp 体素数。用于诊断切片轴方向。"""
    res = {}
    for ax in range(3):
        p = np.moveaxis(pred, ax, 0)
        g = np.moveaxis(gt, ax, 0)
        try:
            r = taxo.decompose_empty_slice_fp(p, g, (1.0, 1.0, 1.0), 26)
            res[ax] = (np.nan if np.isnan(r["empty_fp_total_mm3"])
                       else int(round(r["empty_fp_total_mm3"])))
        except Exception as e:
            res[ax] = f"ERR:{e}"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "stepE02_fp_taxonomy.py"))
    ap.add_argument("--gt-root")
    ap.add_argument("--pred-root")
    ap.add_argument("--frozen-csv")
    ap.add_argument("--out-dir")
    ap.add_argument("--gt-label", type=int, default=3,
                    help="ACDC 原始标签：1=RV 2=MYO 3=LV；若掩膜已二值化则用 --binary")
    ap.add_argument("--binary", action="store_true",
                    help="掩膜已二值化，>0 即前景")
    ap.add_argument("--slice-axis", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--readout", action="store_true",
                    help="(A) 通过后追加设计期读数 (B)")
    a = ap.parse_args()

    if a.discover:
        return discover(a)

    for req in ("gt_root", "pred_root", "frozen_csv", "out_dir"):
        if not getattr(a, req):
            sys.exit(f"[ABORT] 缺少 --{req.replace('_','-')}")
    if nib is None:
        sys.exit("[ABORT] 需要 nibabel：pip install nibabel")
    os.makedirs(a.out_dir, exist_ok=True)

    taxo, taxo_hash = load_taxonomy(a.taxonomy)
    log = {"taxonomy_sha256": taxo_hash,
           "frozen_csv_sha256": sha256(a.frozen_csv),
           "gt_label": a.gt_label, "binary": a.binary,
           "slice_axis": a.slice_axis}
    print(f"[info] taxonomy sha256 = {taxo_hash}")

    frozen = pd.read_csv(a.frozen_csv)
    gt_files = find_niftis(a.gt_root)
    pred_files = find_niftis(a.pred_root)
    print(f"[info] GT {len(gt_files)}  PRED {len(pred_files)}")
    if not gt_files or not pred_files:
        sys.exit("[ABORT] 未找到掩膜文件，请先用 --discover 探路")

    # ---- 建索引
    gt_idx = {}
    for f in gt_files:
        m = parse_meta(f)
        if "key" in m and "frame" in m:
            gt_idx[(m["key"], m["frame"])] = f
    pred_idx = defaultdict(dict)
    for f in pred_files:
        m = parse_meta(f)
        if all(k in m for k in ("key", "frame", "arm", "N", "seed")):
            pred_idx[(m["arm"], m["N"], m["seed"])][(m["key"], m["frame"])] = f
    print(f"[info] GT 索引 {len(gt_idx)}  PRED run 数 {len(pred_idx)}")
    if not pred_idx:
        sys.exit("[ABORT] 预测文件名未能解析出 arm/N/seed，请用 --discover "
                 "把文件名样例发回，我改解析规则")

    # ---- 逐条比对
    rows, probes = [], []
    todo = list(frozen.itertuples(index=False))
    if a.limit:
        todo = todo[:a.limit]

    for i, r in enumerate(todo):
        runkey = (r.arm, int(r.N), int(r.seed))
        cand = [(k, f) for (k, fr), f in pred_idx.get(runkey, {}).items()
                if k == r.key] if False else None
        # frame 号从 GT 索引反查（frozen 表只有 phase，不含 frame）
        frames = sorted([fr for (k, fr) in gt_idx if k == r.key])
        if len(frames) != 2:
            rows.append({"key": r.key, "phase": r.phase, "status": "GT_FRAME_AMBIG",
                         "n_frames_found": len(frames)})
            continue
        frame = frames[0] if r.phase == "ed" else frames[1]
        gf = gt_idx.get((r.key, frame))
        pf = pred_idx.get(runkey, {}).get((r.key, frame))
        if gf is None or pf is None:
            rows.append({"arm": r.arm, "N": r.N, "seed": r.seed, "key": r.key,
                         "phase": r.phase, "status": "MASK_MISSING"})
            continue

        gim = nib.load(gf); pim = nib.load(pf)
        g = np.asanyarray(gim.dataobj); p = np.asanyarray(pim.dataobj)
        gb = (g > 0) if a.binary else (g == a.gt_label)
        pb = (p > 0) if a.binary else (p == a.gt_label)
        if gb.shape != pb.shape:
            rows.append({"arm": r.arm, "N": r.N, "seed": r.seed, "key": r.key,
                         "phase": r.phase, "status": "SHAPE_MISMATCH",
                         "gt_shape": str(gb.shape), "pred_shape": str(pb.shape)})
            continue

        gb = np.moveaxis(gb, a.slice_axis, 0)
        pb = np.moveaxis(pb, a.slice_axis, 0)
        zoom = list(gim.header.get_zooms()[:3])
        spacing = tuple([zoom[a.slice_axis]] +
                        [z for j, z in enumerate(zoom) if j != a.slice_axis])

        out = taxo.decompose_empty_slice_fp(pb, gb, (1.0, 1.0, 1.0), 26)
        got_vox = (np.nan if np.isnan(out["empty_fp_total_mm3"])
                   else int(round(out["empty_fp_total_mm3"])))
        exp_vox = float(r.empty_fp)
        ok = (not np.isnan(got_vox)) and abs(got_vox - exp_vox) < 0.5

        rec = {"arm": r.arm, "N": r.N, "seed": r.seed, "key": r.key,
               "phase": r.phase, "status": "OK" if ok else "MISMATCH",
               "empty_fp_frozen": exp_vox, "empty_fp_recomputed": got_vox,
               "gt_px_frozen": float(r.gt), "gt_px_recomputed": int(gb.sum()),
               "pred_px_frozen": float(r.pred), "pred_px_recomputed": int(pb.sum()),
               "spacing_mm": str(spacing)}
        rows.append(rec)

        if not ok and len(probes) < 10:
            probes.append({"key": r.key, "phase": r.phase,
                           "axis_probe": axis_probe(taxo, pb if a.slice_axis == 0
                                                    else np.moveaxis(pb, 0, a.slice_axis),
                                                    gb if a.slice_axis == 0
                                                    else np.moveaxis(gb, 0, a.slice_axis),
                                                    None),
                           "expected": exp_vox})
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(todo)}")

    df = pd.DataFrame(rows)
    p_val = os.path.join(a.out_dir, "stepE03_validation.csv")
    df.to_csv(p_val, index=False)

    n = len(df)
    n_ok = int((df.get("status") == "OK").sum())
    n_mis = int((df.get("status") == "MISMATCH").sum())
    n_other = n - n_ok - n_mis
    print("\n" + "=" * 70)
    print(f"(A) 门控验证：总 {n}  OK {n_ok}  MISMATCH {n_mis}  其他 {n_other}")
    if "status" in df:
        print(df["status"].value_counts().to_string())
    if probes:
        print("\n轴向诊断（前若干条不匹配样例，三个轴分别的 empty_fp 体素数）：")
        for pr in probes:
            print("  ", pr)
        print("  若某个非 0 轴恰好等于 expected，则 --slice-axis 设错了。")
    log.update({"n": n, "n_ok": n_ok, "n_mismatch": n_mis, "n_other": n_other,
                "probes": probes})
    with open(os.path.join(a.out_dir, "stepE03_log.json"), "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False, default=str)

    if n_ok != n:
        print("\n[GATE FAILED] 未逐病例复现冻结 empty_fp。")
        print("            不产出任何设计期读数。请把上面的诊断发回。")
        sys.exit(3)
    print("[GATE PASSED] 逐病例复现冻结 empty_fp")

    if not a.readout:
        print("加 --readout 以运行 (B) 设计期读数")
        return

    # ---------------------------------------------------------- (B) 设计期读数
    print("\n" + "=" * 70)
    print("(B) DESIGN-PHASE 读数（发现集 ACDC；禁止进入当前手稿）")
    out_rows = []
    for r in todo:
        frames = sorted([fr for (k, fr) in gt_idx if k == r.key])
        frame = frames[0] if r.phase == "ed" else frames[1]
        gim = nib.load(gt_idx[(r.key, frame)])
        pim = nib.load(pred_idx[(r.arm, int(r.N), int(r.seed))][(r.key, frame)])
        g = np.asanyarray(gim.dataobj); p = np.asanyarray(pim.dataobj)
        gb = np.moveaxis((g > 0) if a.binary else (g == a.gt_label), a.slice_axis, 0)
        pb = np.moveaxis((p > 0) if a.binary else (p == a.gt_label), a.slice_axis, 0)
        zoom = list(gim.header.get_zooms()[:3])
        spacing = tuple([zoom[a.slice_axis]] +
                        [z for j, z in enumerate(zoom) if j != a.slice_axis])
        for conn in (26, 6):
            o = taxo.decompose_empty_slice_fp(pb, gb, spacing, conn)
            o.update({"arm": r.arm, "N": r.N, "seed": r.seed,
                      "key": r.key, "phase": r.phase})
            out_rows.append(o)

    rd = pd.DataFrame(out_rows)
    p_rd = os.path.join(a.out_dir, "DESIGNPHASE_stepE03_fp_taxonomy_acdc.csv")
    rd.to_csv(p_rd, index=False)

    for conn in (26, 6):
        s = rd[rd["connectivity"] == conn]
        pos = s[s["empty_fp_total_mm3"] > 0]
        print(f"\n--- {conn}-邻域 ---")
        print(f"  有空层假阳的单位数 {len(pos)} / {len(s)}")
        if len(pos):
            frac_conn = pos["empty_fp_connected_mm3"] / pos["empty_fp_total_mm3"]
            print(f"  connected 占比   中位 {frac_conn.median():.3f}"
                  f"  IQR [{frac_conn.quantile(.25):.3f}, {frac_conn.quantile(.75):.3f}]")
            print(f"  lcc_removable    中位 {pos['lcc_removable_frac'].median():.3f}"
                  f"  IQR [{pos['lcc_removable_frac'].quantile(.25):.3f},"
                  f" {pos['lcc_removable_frac'].quantile(.75):.3f}]")
            byd = pos["empty_fp_beyond_mm3"].sum()
            itr = pos["empty_fp_interior_mm3"].sum()
            print(f"  beyond / interior 体积比 = {byd:.0f} : {itr:.0f}")
        for ph in ("ed", "es"):
            q = pos[pos["phase"] == ph]
            if len(q):
                print(f"    [{ph}] n={len(q)}  lcc_removable 中位"
                      f" {q['lcc_removable_frac'].median():.3f}")
    print(f"\n[written]\n  {p_val}\n  {p_rd}")
    print("\n⚠ (B) 全部为 design-phase：禁止进入当前手稿，"
          "禁止与任何冻结结果做统计比较。")


if __name__ == "__main__":
    main()
