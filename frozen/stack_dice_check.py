#!/usr/bin/env python3
"""
STACK-DICE 快检  —  零 GPU，只读，不改任何冻结产物

回答一个问题：把空层（gt_px=0）的假阳性纳入分母之后，
overlap 指标与 EF 的排序错配是否消失？

结论边界（写作时不得越界）：
  可写 "Switching from present-slice Dice to whole-stack volumetric Dice
        did not eliminate the observed endpoint-ranking discordance."
  不可写 "Any overlap-based metric is intrinsically misaligned with EF."

关键实现点：
  1. 交集反推 inter = dice*(gt_px+pred_px)/2 —— 脚本先自证其精确性：
     inter 必须是整数，检查全体反推值与最近整数的偏差。偏差大 = 有平滑项，
     此时反推不可用，必须回到原始 mask。
  2. gt_px=0 的行贡献 inter=0，与该行 dice 记 0/1/NaN 无关，天然免疫空层约定问题。
  3. 最小分析单位 = run × patient × phase；ED/ES 分开报；再按共同 cell 配对聚合。
  4. 只在所有被比较臂同时存在的 cell 上比较（修欠账 #1 的构成混杂）。

用法:
  python3 stack_dice_check.py --dataset acdc
  python3 stack_dice_check.py --dataset mnms2
"""
import argparse, glob, os, re, sys
import numpy as np
import pandas as pd

ACDC = "/root/autodl-tmp/acdc/seg_lv"
MNMS = "/root/autodl-tmp/mnms/seg_lv"


# ------------------------------------------------------------------ 反推自证
def derive_inter(gt, pred, dice, label=""):
    """inter = dice*(gt+pred)/2，并自证其为整数。返回 (inter, 诊断字典)。"""
    denom = gt + pred
    inter = np.where(denom > 0, dice * denom / 2.0, 0.0)
    # gt=0 的行强制 inter=0（与该行 dice 记法无关）
    inter = np.where(gt == 0, 0.0, inter)
    live = (gt > 0) & (denom > 0)
    resid = np.abs(inter[live] - np.round(inter[live]))
    diag = dict(n=int(live.sum()),
                resid_max=float(resid.max()) if live.sum() else np.nan,
                resid_p999=float(np.percentile(resid, 99.9)) if live.sum() else np.nan,
                frac_gt_1e_3=float((resid > 1e-3).mean()) if live.sum() else np.nan)
    print(f"  [反推自证 {label}] n={diag['n']}  max|resid|={diag['resid_max']:.3e}  "
          f"p99.9={diag['resid_p999']:.3e}  frac>1e-3={diag['frac_gt_1e_3']:.4f}")
    if diag["resid_max"] > 1e-2:
        print("  !! 残差过大：Dice 可能带平滑项或 CSV 有四舍五入。"
              "反推不可用，请改从原始 mask 计算。")
    # 违反 inter<=min(gt,pred) 的行数（另一个独立自证）
    bad = int((inter[live] > np.minimum(gt, pred)[live] + 0.5).sum())
    print(f"  [反推自证 {label}] inter>min(gt,pred) 的行数 = {bad}")
    return inter, diag


def stack_dice(df, gtcol, pcol, dcol, keys, label=""):
    """按 keys 分组算体积 Dice；同时算 present-slice 均值 Dice 作对照。"""
    g = df[gtcol].to_numpy(float)
    p = df[pcol].to_numpy(float)
    d = df[dcol].to_numpy(float)
    inter, _ = derive_inter(g, p, d, label)
    t = df[keys].copy()
    t["_i"], t["_g"], t["_p"] = inter, g, p
    t["_present"] = (g > 0).astype(int)
    t["_dice_present"] = np.where(g > 0, d, np.nan)
    a = t.groupby(keys, dropna=False).agg(
        inter=("_i", "sum"), gt=("_g", "sum"), pred=("_p", "sum"),
        n_present=("_present", "sum"), n_slice=("_i", "size"),
        present_dice=("_dice_present", "mean")).reset_index()
    a["stack_dice"] = 2 * a["inter"] / (a["gt"] + a["pred"]).replace(0, np.nan)
    
    return a


def paired_common_cells(df, cellcols, armcol, valcol, arms):
    """只保留所有 arms 同时存在的 cell，返回逐 cell 配对差。"""
    w = df.pivot_table(index=cellcols, columns=armcol, values=valcol)
    w = w.dropna(subset=list(arms))
    out = w.copy()
    return out


def report_pairs(w, a1, a2, name):
    d = (w[a1] - w[a2]).dropna()
    if len(d) == 0:
        print(f"  {name}: 无共同 cell"); return
    boot = np.random.default_rng(0).choice(d.values, (5000, len(d)), replace=True).mean(1)
    print(f"  {name}: n_cell={len(d)}  mean Δ={d.mean():+.4f}  "
          f"median Δ={np.median(d):+.4f}  "
          f"95%CI=[{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}]  "
          f"支持 {a1} 的 cell={int((d>0).sum())}/{len(d)}")


# ------------------------------------------------------------------- ACDC
def run_acdc():
    files = sorted(glob.glob(os.path.join(ACDC, "eval*", "**", "test_per_slice.csv"),
                             recursive=True))
    print(f"找到 {len(files)} 个 test_per_slice.csv")
    rows = []
    for f in files:
        df = pd.read_csv(f)
        arms = sorted({m.group(1) for c in df.columns
                       if (m := re.match(r"predpx_(\w+)$", c))})
        # 从路径解析 cell（N、seed、split）——按你的目录命名调整这里
        mN = re.search(r"N(\d+)", f); ms = re.search(r"_s(\d+)", f)
        cell_N = int(mN.group(1)) if mN else -1
        cell_s = int(ms.group(1)) if ms else -1
        for arm in arms:
            dcol, pcol = f"dice_{arm}", f"predpx_{arm}"
            if dcol not in df.columns:
                continue
            sub = df[["key", "phase", "gt_px", dcol, pcol]].dropna(subset=[pcol])
            a = stack_dice(sub, "gt_px", pcol, dcol, ["key", "phase"],
                           label=f"{os.path.basename(os.path.dirname(f))}/{arm}")
            a["arm"], a["N"], a["seed"], a["src"] = arm, cell_N, cell_s, f
            rows.append(a)
    if not rows:
        print("无数据"); return
    A = pd.concat(rows, ignore_index=True)
    A.to_csv("stack_dice_acdc_patient_phase.csv", index=False)

    print("\n--- 逐相位、共同 cell 配对（B vs C） ---")
    for phase in ["ed", "es", "ED", "ES"]:
        sub = A[A.phase.astype(str).str.lower() == phase.lower()]
        if sub.empty:
            continue
        pat = sub.groupby(["N", "seed", "src", "arm"]).agg(
            stack=("stack_dice", "mean"), present=("present_dice", "mean")).reset_index()
        for val in ["present", "stack"]:
            w = paired_common_cells(pat, ["N", "seed", "src"], "arm", val, ("B", "C"))
            print(f"\n  [{phase} / {val}]")
            report_pairs(w, "B", "C", f"{phase} {val} B-C")
    print("\n输出: stack_dice_acdc_patient_phase.csv")


# ------------------------------------------------------------------ M&Ms-2
def run_mnms2():
    files = sorted(glob.glob(os.path.join(
        MNMS, "scope_control_v3_eval_rep*", "mnms2_scope_control_v3_per_slice_locked.csv")))
    print(f"找到 {len(files)} 个 per_slice_locked.csv")
    rows = []
    for f in files:
        df = pd.read_csv(f)
        need = {"unit_id", "seed", "arm", "domain", "key", "phase", "gt_px",
                "dice", "pred_px"}
        if not need <= set(df.columns):
            print(f"  {f} 缺列: {need - set(df.columns)}"); continue
        a = stack_dice(df, "gt_px", "pred_px", "dice",
                       ["unit_id", "seed", "arm", "domain", "key", "phase"],
                       label=os.path.basename(os.path.dirname(f)))
        a["rep"] = os.path.basename(os.path.dirname(f))
        rows.append(a)
    if not rows:
        print("无数据"); return
    M = pd.concat(rows, ignore_index=True)
    M.to_csv("stack_dice_mnms2_patient_phase.csv", index=False)

    print("\n--- 逐相位、unit 层配对（B vs Cneg） ---")
    armset = tuple(sorted(M.arm.unique()))
    print("  arms:", armset)
    a2 = "Cneg" if "Cneg" in armset else ("C" if "C" in armset else armset[-1])
    for phase in sorted(M.phase.astype(str).str.lower().unique()):
        sub = M[M.phase.astype(str).str.lower() == phase]
        unit = sub.groupby(["unit_id", "domain", "arm"]).agg(
            stack=("stack_dice", "mean"), present=("present_dice", "mean")).reset_index()
        for val in ["present", "stack"]:
            w = paired_common_cells(unit, ["unit_id", "domain"], "arm", val, ("B", a2))
            print(f"\n  [{phase} / {val}]")
            report_pairs(w, "B", a2, f"{phase} {val} B-{a2}")
    print("\n输出: stack_dice_mnms2_patient_phase.csv")
    print("\n注意：本脚本只产出 overlap 指标，不触碰冻结的 EF-MAE 主终点。"
          "与 EF 排名的比较请在台账里按 confirmatory / exploratory 分别标注。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["acdc", "mnms2"], required=True)
    args = ap.parse_args()
    print("=" * 78)
    print("STACK-DICE 快检 —— 只读，不写冻结目录")
    print("=" * 78)
    (run_acdc if args.dataset == "acdc" else run_mnms2)()
