#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stepE04_asset_audit_and_gt_gate.py
==================================
用途：在"63 个正式 run 的预测掩膜未落盘"这一既成事实下，
      为 stepE02 的空层假阳分类提供**可执行的替代闸门**，
      并把资产存活状态查清（决定这件事是"不可逆"还是"贵"）。

上游：EVAL_VALIDITY_EXTENSION_v0_3_DRAFT_FP_TAXONOMY.md §7
      （原第一项闸门"在 ACDC 真实掩膜上核对 empty_fp 逐病例一致"已不可执行）

四个模式，互相独立，全部**只读 + 纯 CPU**：

  --audit    资产存活审计。回答三个问题：
             (1) 预测掩膜是否真的不存在（比 GPT 的文件计数论证更硬的证据）
             (2) checkpoint 是否还在、哈希是否仍等于 per-case 表的声明值
                 → 若在，掩膜是"可重新生成但需 GPU"，不是"永久丢失"
             (3) 是否存在任何非 NIfTI 形式的数组落盘（.npy/.npz/.h5/.mha/...）

  --discover ACDC GT 布局探路。**不产出任何判定**，只打印文件命名、
             轴序、affine spacing、标签取值，供 --gate 填参数。

  --gate     GT 侧闸门（**真实掩膜、真实闸门**）。由公开 ACDC GT 重算
             每个 (patient, phase) 的前景体素数与空层结构，
             与冻结 stack_dice_acdc_clean.csv 的 `gt` 列逐病例比对。
             验证的是 stepE02 中**只依赖 GT 的那一半**：
             轴 0 是否真为切片轴、spacing 是否取自 affine、Z_present 的定义。

  --perturb  真解剖扰动闸门。由真实 GT 构造答案已知的伪预测
             （越界融合 / 孤立块 / 内部空洞 / 各向异性），
             对 stepE02 的 decompose_empty_slice_fp 做验证。
             比 T1–T9 合成体模强（真实形状、真实各向异性、真实 Z），
             但**仍不能替代对真实模型输出的验证** —— 该缺口按协议须作为
             已知限制显式写入，不得当作已闭合。

身份：design-phase（发现集 ACDC，修正案 v0.2 §1.1 授权）。
      本脚本任何输出**禁止进入当前手稿**，禁止与任何冻结结果做统计比较。
      --gate 为代码正确性自证（与 stepE01 的 S1/S2 同性质），不是新统计量。

依赖：numpy scipy pandas nibabel

用法
----
  python3 stepE04_asset_audit_and_gt_gate.py --audit \
      --roots /root/autodl-tmp /root \
      --percase-csv /path/mnms2_scope_control_v3_R_aggregate_averaged_per_case.csv \
      --out-dir /root/autodl-tmp/design_phase_quarantine

  python3 stepE04_asset_audit_and_gt_gate.py --discover --acdc-root /root/autodl-tmp/acdc

  python3 stepE04_asset_audit_and_gt_gate.py --gate \
      --acdc-root /root/autodl-tmp/acdc --label 3 \
      --frozen-csv /path/stack_dice_acdc_clean.csv \
      --out-dir /root/autodl-tmp/design_phase_quarantine

  python3 stepE04_asset_audit_and_gt_gate.py --perturb \
      --acdc-root /root/autodl-tmp/acdc --label 3 --taxonomy /path/stepE02_fp_taxonomy.py \
      --out-dir /root/autodl-tmp/design_phase_quarantine
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import tarfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError:
    nib = None

ARRAY_EXT = (".npy", ".npz", ".h5", ".hdf5", ".mha", ".mhd", ".nrrd",
             ".nii", ".nii.gz", ".pkl", ".pickle", ".raw")
CKPT_EXT = (".pt", ".pth", ".ckpt", ".safetensors")
PRED_HINT = re.compile(r"(pred|prob|logit|infer|seg_out|output|mask)", re.I)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def ensure_out(d):
    os.makedirs(d, exist_ok=True)
    return d


# ==================================================================== AUDIT
def cmd_audit(a):
    out = ensure_out(a.out_dir)
    report = {"mode": "audit", "roots": a.roots}

    ext_counter = Counter()
    pred_hits, ckpt_files, tars = [], [], []
    for root in a.roots:
        if not os.path.isdir(root):
            print(f"[WARN] 不存在：{root}")
            continue
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in (".git", "__pycache__")]
            for fn in fns:
                p = os.path.join(dp, fn)
                low = fn.lower()
                for e in ARRAY_EXT:
                    if low.endswith(e):
                        ext_counter[e] += 1
                        if PRED_HINT.search(p):
                            pred_hits.append(p)
                        break
                if low.endswith(CKPT_EXT):
                    ckpt_files.append(p)
                if low.endswith((".tar", ".tar.gz", ".tgz")):
                    tars.append(p)

    report["array_file_counts"] = dict(ext_counter)
    report["pred_like_array_files_n"] = len(pred_hits)
    report["pred_like_array_files_sample"] = pred_hits[:50]
    report["checkpoint_files_n"] = len(ckpt_files)

    # ---- tar 内部清单（只列表，不解包）
    tar_inventory = {}
    for t in tars:
        try:
            with tarfile.open(t, "r:*") as tf:
                names = tf.getnames()
            c = Counter()
            for n in names:
                nl = n.lower()
                for e in ARRAY_EXT + CKPT_EXT:
                    if nl.endswith(e):
                        c[e] += 1
                        break
            tar_inventory[t] = {"n_members": len(names), "by_ext": dict(c),
                                "pred_like": sum(1 for n in names if PRED_HINT.search(n))}
        except Exception as ex:
            tar_inventory[t] = {"error": repr(ex)}
    report["tar_inventory"] = tar_inventory

    # ---- checkpoint 存活 + 哈希核对（决定"不可逆 or 只是贵"）
    ck = {"declared": 0, "exists": 0, "hash_match": 0, "hash_mismatch": [], "missing": []}
    if a.percase_csv and os.path.exists(a.percase_csv):
        df = pd.read_csv(a.percase_csv)
        if {"checkpoint", "checkpoint_sha256"} <= set(df.columns):
            decl = df[["checkpoint", "checkpoint_sha256"]].drop_duplicates()
            ck["declared"] = len(decl)
            for p, h in decl.itertuples(index=False):
                if os.path.exists(p):
                    ck["exists"] += 1
                    got = sha256(p)
                    if got == h:
                        ck["hash_match"] += 1
                    else:
                        ck["hash_mismatch"].append({"path": p, "declared": h, "actual": got})
                else:
                    ck["missing"].append(p)
    report["checkpoint_survival"] = ck

    with open(os.path.join(out, "stepE04_audit.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n===== 资产存活审计 =====")
    print("数组类文件计数：", dict(ext_counter))
    print("疑似预测数组文件：", len(pred_hits))
    print("checkpoint 文件：", len(ckpt_files))
    print(f"per-case 表声明 checkpoint {ck['declared']} 个："
          f"存在 {ck['exists']}，哈希吻合 {ck['hash_match']}，"
          f"不吻合 {len(ck['hash_mismatch'])}，缺失 {len(ck['missing'])}")
    print("\n判读规则（写在这里，避免事后解释）：")
    print("  A. 疑似预测数组 = 0 且 tar 内 pred_like = 0  → 掩膜确未落盘（证据强于文件计数比较）")
    print("  B. checkpoint 哈希全部吻合                   → 掩膜为『需 GPU 重生成』，非永久丢失")
    print("  C. checkpoint 缺失或哈希不符                 → 该 run 的掩膜为永久丢失，须记入台账")
    print(f"\n写出：{os.path.join(out, 'stepE04_audit.json')}")


# ================================================================= DISCOVER
def _nifti_list(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith((".nii", ".nii.gz")):
                out.append(os.path.join(dp, fn))
    return sorted(out)


PAT_KEY = re.compile(r"(patient\d{3})", re.I)
PAT_FRAME = re.compile(r"frame(\d+)", re.I)


def cmd_discover(a):
    if nib is None:
        sys.exit("[ABORT] 需要 nibabel")
    files = _nifti_list(a.acdc_root)
    print(f"NIfTI 总数：{len(files)}")
    gts = [f for f in files if "_gt" in os.path.basename(f).lower()]
    print(f"含 _gt 的文件：{len(gts)}")
    keys = sorted({m.group(1).lower() for f in files if (m := PAT_KEY.search(f))})
    print(f"患者键：{len(keys)} 个，示例 {keys[:5]} … {keys[-3:] if keys else ''}")
    cfgs = []
    for dp, _, fns in os.walk(a.acdc_root):
        for fn in fns:
            if fn.lower() == "info.cfg":
                cfgs.append(os.path.join(dp, fn))
    print(f"Info.cfg：{len(cfgs)} 个 —— ED/ES 帧号来源，缺失则无法做相位对齐")
    for f in (gts or files)[:5]:
        img = nib.load(f)
        zoom = img.header.get_zooms()
        data = np.asanyarray(img.dataobj)
        vals = np.unique(data)
        print(f"\n{f}\n  shape={data.shape} zooms={zoom}")
        print(f"  affine 对角={np.round(np.abs(np.diag(img.affine))[:3], 4)}")
        print(f"  标签取值={vals[:10]}")
        if data.ndim == 3:
            nz = [int(data[..., z].any()) for z in range(data.shape[2])]
            print(f"  末轴非空层 {sum(nz)}/{data.shape[2]}（若末轴是切片轴，此数应≈心尖到心底层数）")
    print("\n⚠️ ACDC 原生 NIfTI 通常为 (X, Y, Z)，切片轴是**末轴**；"
          "stepE02 约定轴 0 为切片轴。--gate/--perturb 会按 --slice-axis 转置。"
          "该约定必须逐数据集分别确认，不得假定相同（spec §7）。")


# ===================================================================== GATE
PAT_PHASE_NAME = re.compile(r"[_\-](ed|es)[_\-]", re.I)


def _phase_from_name(base, key, ed_es):
    """两种命名约定：
       (a) 官方 ACDC：patientXXX_frameNN_gt.nii.gz + Info.cfg 给 ED/ES 帧号
       (b) 本项目预处理副本：patientXXX_sax_ed_gt.nii.gz（相位写在文件名里，无 Info.cfg）
       只认这两种，认不出返回 None——不猜。"""
    m = PAT_PHASE_NAME.search(base)
    if m:
        return m.group(1).lower()
    fr = PAT_FRAME.search(base)
    if fr and key in ed_es:
        ed, es = ed_es[key]
        f = int(fr.group(1))
        return "ed" if f == ed else ("es" if f == es else None)
    return None


def _load_acdc_pairs(root, label, slice_axis):
    """返回 {(key, phase): (mask3d[Z,Y,X], zooms_zyx)}。"""
    if nib is None:
        sys.exit("[ABORT] 需要 nibabel")
    # 1) Info.cfg → ED/ES 帧号（命名约定 (a) 才需要；不存在则留空）
    ed_es = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower() == "info.cfg":
                k = PAT_KEY.search(dp)
                if not k:
                    continue
                cfg = {}
                for line in open(os.path.join(dp, fn)):
                    if ":" in line:
                        a_, b_ = line.split(":", 1)
                        cfg[a_.strip().upper()] = b_.strip()
                try:
                    ed_es[k.group(1).lower()] = (int(cfg["ED"]), int(cfg["ES"]))
                except Exception:
                    pass
    out = {}
    for f in _nifti_list(root):
        base = os.path.basename(f).lower()
        if "_gt" not in base:
            continue
        k = PAT_KEY.search(base)
        if not k:
            continue
        key = k.group(1).lower()
        phase = _phase_from_name(base, key, ed_es)
        if phase is None:
            continue
        img = nib.load(f)
        d = np.asanyarray(img.dataobj)
        if d.ndim != 3:
            continue
        m = (d == label)
        z = list(img.header.get_zooms()[:3])
        m = np.moveaxis(m, slice_axis, 0)
        z = [z[slice_axis]] + [z[i] for i in range(3) if i != slice_axis]
        out[(key, phase)] = (m, tuple(float(x) for x in z))
    return out


def cmd_gate(a):
    out = ensure_out(a.out_dir)
    fro = pd.read_csv(a.frozen_csv)
    # 冻结表中同一 (key, phase) 的 gt 列跨 63 run 应为常数——先自查
    const = fro.groupby(["key", "phase"])["gt"].nunique()
    n_nonconst = int((const > 1).sum())
    print(f"[前置] 冻结表 gt 列跨 run 非常数的 (key,phase) 数：{n_nonconst}（应为 0）")

    ref = fro.groupby(["key", "phase"])["gt"].first().to_dict()
    pairs = _load_acdc_pairs(a.acdc_root, a.label, a.slice_axis)
    print(f"[前置] 从 GT 解析出的 (key,phase) 数：{len(pairs)}；冻结表中：{len(ref)}")
    if not pairs:
        sys.exit("[ABORT] 未解析出任何 GT 单元 —— 这是解析失败，不是闸门 FAIL。"
                 "先跑 --discover 核对命名与目录。0/0 不得记为任何判定。")

    rows, miss = [], []
    for (k, ph), gtv in ref.items():
        if (k, ph) not in pairs:
            miss.append((k, ph))
            continue
        m, zoom = pairs[(k, ph)]
        zpres = np.array([m[z].any() for z in range(m.shape[0])])
        rows.append({
            "key": k, "phase": ph,
            "gt_frozen": int(gtv), "gt_recomputed": int(m.sum()),
            "diff": int(m.sum()) - int(gtv),
            "n_slices": int(m.shape[0]),
            "n_gt_present": int(zpres.sum()),
            "n_gt_empty": int((~zpres).sum()),
            "gt_support_contiguous": bool(
                zpres.sum() == 0 or
                (np.flatnonzero(zpres)[-1] - np.flatnonzero(zpres)[0] + 1) == zpres.sum()),
            "dz_mm": zoom[0], "dy_mm": zoom[1], "dx_mm": zoom[2],
        })
    df = pd.DataFrame(rows)
    p = os.path.join(out, "stepE04_gt_gate.csv")
    df.to_csv(p, index=False)

    n = len(df)
    exact = int((df["diff"] == 0).sum()) if n else 0
    print("\n===== GT 侧闸门 =====")
    print(f"可比对单元：{n}；缺失：{len(miss)}")
    print(f"`gt` 列逐病例精确一致：{exact}/{n}")
    if n and exact < n:
        print("不一致样例：")
        print(df[df["diff"] != 0].head(10).to_string(index=False))
    if n:
        print(f"空层数分布：中位 {df['n_gt_empty'].median():.1f}，"
              f"范围 {df['n_gt_empty'].min()}–{df['n_gt_empty'].max()}")
        print(f"GT 轴向支撑连续的单元：{int(df['gt_support_contiguous'].sum())}/{n} "
              "（不连续者会使『beyond / interior』划分产生歧义，须在协议中处理）")
    print("\n判读（预先写死，不得事后解释）：")
    print("  PASS  = exact/n 为 1.00 → 轴序、标签、spacing、相位映射与冻结管线一致")
    print("  FAIL  = 任一不一致 → 先查 --slice-axis 与 --label，再查 GT 版本；"
          "在 FAIL 状态下不得运行 --perturb 之外的任何读数")
    print(f"\n写出：{p}")


# ================================================================== PERTURB
def _load_taxonomy(path):
    spec = importlib.util.spec_from_file_location("taxo", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, sha256(path)


def _make_cases(gt):
    """由真实 GT 构造答案已知的伪预测。返回 {名称: (pred, 期望性质)}。"""
    from scipy import ndimage
    Z = gt.shape[0]
    zp = np.array([gt[z].any() for z in range(Z)])
    idx = np.flatnonzero(zp)
    cases = {}
    # P0 完美预测：空层假阳应为 0
    cases["P0_perfect"] = (gt.copy(), {"empty_fp_total_mm3": 0.0})
    if idx.size == 0:
        return cases
    lo, hi = int(idx[0]), int(idx[-1])
    # P1 越界一层且与主体连通（LCC 清不掉）→ connected + beyond
    p1 = gt.copy()
    if hi + 1 < Z:
        p1[hi + 1] = ndimage.binary_dilation(gt[hi], iterations=1)
        cases["P1_beyond_connected"] = (p1, {"expect": "connected>0, lcc_removable_frac==0"})
    # P2 孤立块落在空层（LCC 清得掉）→ isolated
    p2 = gt.copy()
    tgt = None
    for z in range(Z):
        if not zp[z] and (z == 0 or z >= hi + 2):
            tgt = z
            break
    if tgt is not None:
        yy, xx = gt.shape[1] // 4, gt.shape[2] // 4
        p2[tgt, yy:yy + 3, xx:xx + 3] = True
        cases["P2_isolated"] = (p2, {"expect": "isolated>0, lcc_removable_frac==1"})
    # P3 两者并存 → LCC 只清得掉 isolated 的一份
    if "P1_beyond_connected" in cases and "P2_isolated" in cases:
        cases["P3_mixed"] = (p1 | p2, {"expect": "0 < lcc_removable_frac < 1"})
    # P4 只在 present 层加杂块 → 不得计入空层 FP（分类不越界）
    p4 = gt.copy()
    yy, xx = gt.shape[1] // 4, gt.shape[2] // 4
    p4[lo, yy:yy + 3, xx:xx + 3] = True
    cases["P4_present_only_blob"] = (p4, {"empty_fp_total_mm3": 0.0})
    # P5 角接触（真实解剖版的 T8）：26 下 connected、6 下 isolated
    #    这一例决定 spec §4 的 26/6 敏感性在真实数据上是否**实质**
    if hi + 1 < Z:
        ys, xs = np.nonzero(gt[hi])
        if ys.size:
            j = int(np.argmax(ys + xs))
            y0, x0 = int(ys[j]) + 1, int(xs[j]) + 1
            if y0 + 2 <= gt.shape[1] and x0 + 2 <= gt.shape[2]:
                p5 = gt.copy()
                blob = np.zeros_like(gt)
                blob[hi + 1, y0:y0 + 2, x0:x0 + 2] = True
                # 守卫：必须与 GT 无交、且无 6-邻接（否则该例不成立，跳过）
                from scipy import ndimage as _nd
                face = _nd.binary_dilation(blob, structure=_nd.generate_binary_structure(3, 1))
                if not (blob & gt).any() and not (face & gt).any():
                    p5 |= blob
                    cases["P5_corner_contact"] = (
                        p5, {"expect": "26: connected(lcc_removable=0) / 6: isolated(=1)"})
    return cases


def cmd_perturb(a):
    out = ensure_out(a.out_dir)
    taxo, taxo_hash = _load_taxonomy(a.taxonomy)
    print(f"stepE02 SHA256 = {taxo_hash}")
    pairs = _load_acdc_pairs(a.acdc_root, a.label, a.slice_axis)
    if not pairs:
        sys.exit("[ABORT] 未解析出任何 (key, phase) GT 单元。先跑 --discover 核对命名与目录，"
                 "不要在空输入上产出任何读数。")
    keys = sorted(pairs.keys())[: a.limit] if a.limit else sorted(pairs.keys())
    rows = []
    for (k, ph) in keys:
        gt, zoom = pairs[(k, ph)]
        for cname, (pred, exp) in _make_cases(gt).items():
            for conn in (26, 6):
                r = taxo.decompose_empty_slice_fp(pred, gt, spacing=zoom, connectivity=conn)
                r.update({"key": k, "phase": ph, "case": cname,
                          "expect": json.dumps(exp, ensure_ascii=False)})
                rows.append(r)
    df = pd.DataFrame(rows)
    p = os.path.join(out, "stepE04_perturb.csv")
    df.to_csv(p, index=False)

    print("\n===== 真解剖扰动闸门 =====")
    ok = True
    for cname, expected_zero in (("P0_perfect", True), ("P4_present_only_blob", True)):
        sub = df[(df["case"] == cname)]
        if len(sub):
            bad = int((sub["empty_fp_total_mm3"].fillna(0) > 0).sum())
            print(f"{cname}: 空层假阳应为 0 → 违例 {bad}/{len(sub)}")
            ok &= (bad == 0)
    for cname, col, want in (("P1_beyond_connected", "empty_fp_connected_mm3", ">0"),
                             ("P2_isolated", "empty_fp_isolated_mm3", ">0")):
        sub = df[(df["case"] == cname) & (df["connectivity"] == 26)]
        if len(sub) and col in sub:
            good = int((sub[col].fillna(0) > 0).sum())
            print(f"{cname}: {col} {want} → 满足 {good}/{len(sub)}")
            ok &= (good == len(sub))
    sub5 = df[df["case"] == "P5_corner_contact"]
    if len(sub5):
        a26 = sub5[sub5["connectivity"] == 26]["lcc_removable_frac"].fillna(-1)
        a6 = sub5[sub5["connectivity"] == 6]["lcc_removable_frac"].fillna(-1)
        print(f"P5_corner_contact: 26 邻域中位 {a26.median():.2f}（应 0）, "
              f"6 邻域中位 {a6.median():.2f}（应 1）—— 构造成功 {len(a26)} 例")
    if "lcc_removable_frac" in df:
        s26 = df[df["connectivity"] == 26].set_index(["key", "phase", "case"])["lcc_removable_frac"]
        s6 = df[df["connectivity"] == 6].set_index(["key", "phase", "case"])["lcc_removable_frac"]
        j = pd.concat([s26.rename("c26"), s6.rename("c6")], axis=1).dropna()
        if len(j):
            flip = float((np.abs(j["c26"] - j["c6"]) > 0.5).mean())
            print(f"26 vs 6 邻域的 lcc_removable_frac 大幅分歧比例：{flip:.3f}")
            print("  → 该比例即 spec §4 敏感性分析的实质性证据；"
                  "若在真实各向异性下接近 0，须如实记录『敏感性在本数据上不实质』")
    print(f"\n总体：{'PASS' if ok else 'FAIL'}")
    print("⚠️ 本闸门用真实解剖 + 真实各向异性 spacing，强于 T1–T9 合成体模，"
          "但**不含任何真实模型输出**。该缺口不得表述为已闭合。")
    print(f"\n写出：{p}")


# ====================================================================== CLI
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audit", action="store_true")
    g.add_argument("--discover", action="store_true")
    g.add_argument("--gate", action="store_true")
    g.add_argument("--perturb", action="store_true")
    ap.add_argument("--roots", nargs="*", default=["/root/autodl-tmp"])
    ap.add_argument("--percase-csv", default=None)
    ap.add_argument("--acdc-root", default=None)
    ap.add_argument("--frozen-csv", default=None)
    ap.add_argument("--taxonomy", default="stepE02_fp_taxonomy.py")
    ap.add_argument("--label", type=int, default=3, help="ACDC: 1=RV 2=MYO 3=LV")
    ap.add_argument("--slice-axis", type=int, default=2, help="原生 NIfTI 中的切片轴，默认末轴")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="./design_phase_quarantine")
    a = ap.parse_args()

    if a.audit:
        cmd_audit(a)
    elif a.discover:
        if not a.acdc_root:
            sys.exit("[ABORT] --discover 需要 --acdc-root")
        cmd_discover(a)
    elif a.gate:
        for req in ("acdc_root", "frozen_csv"):
            if not getattr(a, req):
                sys.exit(f"[ABORT] --gate 需要 --{req.replace('_', '-')}")
        cmd_gate(a)
    else:
        if not a.acdc_root:
            sys.exit("[ABORT] --perturb 需要 --acdc-root")
        cmd_perturb(a)


if __name__ == "__main__":
    main()
