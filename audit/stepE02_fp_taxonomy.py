#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stepE02_fp_taxonomy.py
======================
空层假阳分类（connected / isolated）的**参考实现 + 自检**

上游：EVAL_VALIDITY_EXTENSION_v0_3_DRAFT_FP_TAXONOMY.md
身份：设计期实现；**尚未在任何真实分割掩膜上验证**。
      本文件的自检全部基于**合成体模**（答案由构造已知），
      锁定前必须补做 ACDC 真实掩膜上的验证（见 spec §7）。

不读取任何冻结数据，不产生任何与手稿相关的数字。

用法：
  python3 stepE02_fp_taxonomy.py --selftest
"""

import argparse
import sys

import numpy as np
from scipy import ndimage

# --------------------------------------------------------------- 连通性结构元
STRUCT = {
    26: np.ones((3, 3, 3), dtype=bool),
    6: ndimage.generate_binary_structure(3, 1),
}


def _label(mask, connectivity):
    if connectivity not in STRUCT:
        raise ValueError("connectivity 必须为 26 或 6")
    lab, n = ndimage.label(mask, structure=STRUCT[connectivity])
    return lab, n


def decompose_empty_slice_fp(pred, gt, spacing=(1.0, 1.0, 1.0), connectivity=26):
    """
    输入
    ----
    pred, gt : (Z, Y, X) 布尔数组，**单结构、单相位、单病例**
               轴 0 为切片轴（短轴 stack 的层方向 / CT 的 z）
    spacing  : (dz, dy, dx)，单位 mm
    connectivity : 26（主分析）或 6（敏感性分析）

    输出
    ----
    dict，体积单位 mm^3
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if pred.shape != gt.shape or pred.ndim != 3:
        raise ValueError("pred/gt 必须同形且为 3D")

    vox_mm3 = float(np.prod(spacing))
    Z = pred.shape[0]

    # ---- GT 轴向支撑
    z_present = np.array([gt[z].any() for z in range(Z)], dtype=bool)
    out = {
        "n_slices": Z,
        "n_gt_present_slices": int(z_present.sum()),
        "gt_volume_mm3": float(gt.sum() * vox_mm3),
        "pred_volume_mm3": float(pred.sum() * vox_mm3),
        "connectivity": connectivity,
    }

    # GT 全空的病例：本分类未定义，显式标记而非静默返回 0
    if not z_present.any():
        out.update(dict.fromkeys(
            ["empty_fp_total_mm3", "empty_fp_connected_mm3",
             "empty_fp_isolated_mm3", "empty_fp_beyond_mm3",
             "empty_fp_interior_mm3", "empty_fp_after_lcc_mm3",
             "lcc_removable_frac"], np.nan))
        out["gt_all_empty"] = True
        return out
    out["gt_all_empty"] = False

    z_lo, z_hi = int(np.argmax(z_present)), int(Z - 1 - np.argmax(z_present[::-1]))
    z_empty = ~z_present
    z_interior = z_empty.copy()
    z_interior[:z_lo] = False
    z_interior[z_hi + 1:] = False
    z_beyond = z_empty & ~z_interior

    out["n_empty_slices"] = int(z_empty.sum())
    out["n_empty_interior"] = int(z_interior.sum())
    out["n_empty_beyond"] = int(z_beyond.sum())

    # ---- 空层上的预测前景（GT 在这些层上按定义全为 0，故全部是 FP）
    empty_mask = np.zeros_like(pred)
    empty_mask[z_empty] = pred[z_empty]
    out["empty_fp_total_mm3"] = float(empty_mask.sum() * vox_mm3)

    # ---- 连通域分解
    lab, n = _label(pred, connectivity)
    out["n_components"] = int(n)

    if n == 0:
        out.update({"empty_fp_connected_mm3": 0.0,
                    "empty_fp_isolated_mm3": 0.0,
                    "empty_fp_beyond_mm3": 0.0,
                    "empty_fp_interior_mm3": 0.0,
                    "empty_fp_after_lcc_mm3": 0.0,
                    "lcc_removable_frac": np.nan,
                    "n_gt_anchored_components": 0})
        return out

    # GT-anchored 判据：**与 GT 体素有空间重叠**（非"落在 GT-present 层上"）
    anchored_ids = set(np.unique(lab[gt & (lab > 0)]).tolist()) - {0}
    out["n_gt_anchored_components"] = len(anchored_ids)

    anchored_mask = np.isin(lab, list(anchored_ids)) if anchored_ids \
        else np.zeros_like(pred)

    conn = empty_mask & anchored_mask
    iso = empty_mask & ~anchored_mask
    out["empty_fp_connected_mm3"] = float(conn.sum() * vox_mm3)
    out["empty_fp_isolated_mm3"] = float(iso.sum() * vox_mm3)

    beyond = np.zeros_like(pred); beyond[z_beyond] = empty_mask[z_beyond]
    inter = np.zeros_like(pred); inter[z_interior] = empty_mask[z_interior]
    out["empty_fp_beyond_mm3"] = float(beyond.sum() * vox_mm3)
    out["empty_fp_interior_mm3"] = float(inter.sum() * vox_mm3)

    # ---- 规范 LCC 算子（模拟"做个最大连通域后处理不就解决了"）
    #      与流水线自身的后处理因子（v0.3 §3.4）是两回事，不得混同
    sizes = ndimage.sum_labels(np.ones_like(pred, dtype=np.int64), lab,
                               index=np.arange(1, n + 1))
    keep = int(np.argmax(sizes)) + 1          # 并列时取最小 label，确定性
    pred_lcc = (lab == keep)
    empty_after = np.zeros_like(pred)
    empty_after[z_empty] = pred_lcc[z_empty]
    out["empty_fp_after_lcc_mm3"] = float(empty_after.sum() * vox_mm3)
    tot = out["empty_fp_total_mm3"]
    out["lcc_removable_frac"] = (
        float((tot - out["empty_fp_after_lcc_mm3"]) / tot) if tot > 0 else np.nan)

    # ---- 相对负担（无量纲）
    out["empty_fp_rel_burden"] = (
        tot / out["gt_volume_mm3"] if out["gt_volume_mm3"] > 0 else np.nan)
    return out


# =============================================================== 自检（合成体模）
def _blank(Z=10, Y=20, X=20):
    return np.zeros((Z, Y, X), dtype=bool)


def _selftest():
    fails = []

    def chk(name, got, exp, tol=1e-9):
        ok = (np.isnan(got) and np.isnan(exp)) or abs(got - exp) <= tol
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: got={got} exp={exp}")
        if not ok:
            fails.append(name)

    sp = (2.0, 1.0, 1.0)          # dz=2mm -> 每体素 2 mm^3
    v = 2.0

    # T1 预测在上端越界一层，与主体融合 -> connected & beyond
    print("\n[T1] 越界一层，与主体融合")
    gt = _blank(); gt[3:7, 8:12, 8:12] = True
    pr = _blank(); pr[3:8, 8:12, 8:12] = True          # 多出 z=7 的 4x4=16 体素
    r = decompose_empty_slice_fp(pr, gt, sp)
    chk("T1 total", r["empty_fp_total_mm3"], 16 * v)
    chk("T1 connected", r["empty_fp_connected_mm3"], 16 * v)
    chk("T1 isolated", r["empty_fp_isolated_mm3"], 0.0)
    chk("T1 beyond", r["empty_fp_beyond_mm3"], 16 * v)
    chk("T1 interior", r["empty_fp_interior_mm3"], 0.0)
    chk("T1 LCC 可清除比例", r["lcc_removable_frac"], 0.0)   # LCC 清不掉

    # T2 空层上的孤立小块 -> isolated；LCC 可清除
    print("\n[T2] 空层孤立小块")
    gt = _blank(); gt[3:7, 8:12, 8:12] = True
    pr = _blank(); pr[3:7, 8:12, 8:12] = True
    pr[8, 2:4, 2:4] = True                              # 4 体素，远离主体
    r = decompose_empty_slice_fp(pr, gt, sp)
    chk("T2 total", r["empty_fp_total_mm3"], 4 * v)
    chk("T2 connected", r["empty_fp_connected_mm3"], 0.0)
    chk("T2 isolated", r["empty_fp_isolated_mm3"], 4 * v)
    chk("T2 LCC 可清除比例", r["lcc_removable_frac"], 1.0)

    # T3 GT 内部缺一层，预测填满 -> connected & interior
    print("\n[T3] GT 内部空层被填充")
    gt = _blank(); gt[3:7, 8:12, 8:12] = True; gt[5] = False
    pr = _blank(); pr[3:7, 8:12, 8:12] = True
    r = decompose_empty_slice_fp(pr, gt, sp)
    chk("T3 total", r["empty_fp_total_mm3"], 16 * v)
    chk("T3 connected", r["empty_fp_connected_mm3"], 16 * v)
    chk("T3 interior", r["empty_fp_interior_mm3"], 16 * v)
    chk("T3 beyond", r["empty_fp_beyond_mm3"], 0.0)

    # T4 完美预测 -> 全 0
    print("\n[T4] 完美预测")
    gt = _blank(); gt[3:7, 8:12, 8:12] = True
    r = decompose_empty_slice_fp(gt.copy(), gt, sp)
    chk("T4 total", r["empty_fp_total_mm3"], 0.0)
    chk("T4 LCC 比例(未定义)", r["lcc_removable_frac"], np.nan)

    # T5 空预测 -> 全 0，不报错
    print("\n[T5] 空预测")
    gt = _blank(); gt[3:7, 8:12, 8:12] = True
    r = decompose_empty_slice_fp(_blank(), gt, sp)
    chk("T5 total", r["empty_fp_total_mm3"], 0.0)
    chk("T5 components", float(r["n_components"]), 0.0)

    # T6 present 层上的离散错误块 -> **不计入空层 FP**（关键：分类不越界）
    print("\n[T6] present 层上的离散错误块不得计入空层 FP")
    gt = _blank(); gt[3:7, 8:12, 8:12] = True
    pr = _blank(); pr[3:7, 8:12, 8:12] = True
    pr[4, 2:4, 2:4] = True                              # 落在 GT-present 层
    r = decompose_empty_slice_fp(pr, gt, sp)
    chk("T6 total", r["empty_fp_total_mm3"], 0.0)
    chk("T6 isolated", r["empty_fp_isolated_mm3"], 0.0)

    # T7 connected + isolated 混合：LCC 只清得掉 isolated 部分
    print("\n[T7] 混合：LCC 只清得掉 isolated")
    gt = _blank(); gt[3:7, 8:12, 8:12] = True
    pr = _blank(); pr[3:8, 8:12, 8:12] = True           # 越界 16 体素（connected）
    pr[9, 2:4, 2:4] = True                              # 孤立 4 体素
    r = decompose_empty_slice_fp(pr, gt, sp)
    chk("T7 total", r["empty_fp_total_mm3"], 20 * v)
    chk("T7 connected", r["empty_fp_connected_mm3"], 16 * v)
    chk("T7 isolated", r["empty_fp_isolated_mm3"], 4 * v)
    chk("T7 LCC 可清除比例", r["lcc_removable_frac"], 4 / 20)

    # T8 连通性选择会改变结论：仅对角相邻的越界块
    print("\n[T8] 26 vs 6 连通性：对角相邻块的归属翻转")
    gt = _blank(); gt[3:7, 8:12, 8:12] = True
    pr = _blank(); pr[3:7, 8:12, 8:12] = True
    pr[7, 12:14, 12:14] = True                          # 与主体仅角接触
    r26 = decompose_empty_slice_fp(pr, gt, sp, connectivity=26)
    r6 = decompose_empty_slice_fp(pr, gt, sp, connectivity=6)
    chk("T8 26-connected", r26["empty_fp_connected_mm3"], 4 * v)
    chk("T8 26-isolated", r26["empty_fp_isolated_mm3"], 0.0)
    chk("T8 6-connected", r6["empty_fp_connected_mm3"], 0.0)
    chk("T8 6-isolated", r6["empty_fp_isolated_mm3"], 4 * v)
    chk("T8 26 LCC 可清除比例", r26["lcc_removable_frac"], 0.0)
    chk("T8 6 LCC 可清除比例", r6["lcc_removable_frac"], 1.0)

    # T9 GT 全空 -> 显式 NaN，不静默返回 0
    print("\n[T9] GT 全空病例显式 NaN")
    pr = _blank(); pr[5, 8:12, 8:12] = True
    r = decompose_empty_slice_fp(pr, _blank(), sp)
    chk("T9 total", r["empty_fp_total_mm3"], np.nan)
    print(f"  INFO  gt_all_empty={r['gt_all_empty']}")

    print("\n" + "=" * 60)
    if fails:
        print(f"[SELFTEST FAILED] {len(fails)} 项：{fails}")
        return 1
    print("[SELFTEST PASSED] 全部 9 组体模通过")
    print("⚠ 合成体模通过 ≠ 真实掩膜可用。锁定前须补做 ACDC 真实掩膜验证（spec §7）")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    ap.print_help()
