#!/usr/bin/env python3
"""
STEP 0.6  —  零 GPU，只读
修 step05 的三个缺陷 + 补一个关键关联分析：
  (a) seed 解析：无 _s 后缀的目录 = seed 0（上一轮被 pivot_table 静默丢弃）
  (b) 去重：内容哈希失败（重评估有 ~1e-4 非确定性），改按 (N, seed) 显式配别名，
      并把两次评估的数值差作为敏感性报告 —— 该差本身是要写进论文的数据
  (c) n<5 时不报 bootstrap CI，直接列逐 seed 值
  (d) 新增：跨 run 的 stack/present Dice 与 EF-MAE 的关联
      —— 直接回答"stack Dice 是不是更好的终点代理"

用法:
  cd /root/autodl-tmp/sam3-main/
  python3 step06_fix.py > step06_report.txt 2>&1
  echo "exit=$?"; wc -c step06_report.txt
"""
import os, re, glob
import numpy as np
import pandas as pd

ACDC = "/root/autodl-tmp/acdc/seg_lv"


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def derive_inter(gt, pred, dice):
    den = gt + pred
    it = np.where(den > 0, dice * den / 2.0, 0.0)
    return np.where(gt == 0, 0.0, it)


def parse_cell(tag, path):
    """返回 (N, seed, is_alias)。无 _s 后缀 = seed 0。eval* 视为别名。"""
    m = re.search(r"N(\d+)", tag) or re.search(r"N(\d+)", path)
    N = int(m.group(1)) if m else None
    s = re.search(r"_s(\d+)", tag)
    seed = int(s.group(1)) if s else 0
    alias = tag.startswith("eval")
    if tag == "eval":          # 快照：eval 是 N100 seed0 的重复评估
        N, seed = 100, 0
    return N, seed, alias


# ============================================================ 读取全部 run
rows = []
for f in sorted(glob.glob(os.path.join(ACDC, "eval*", "**", "test_per_slice.csv"),
                          recursive=True)):
    df = pd.read_csv(f)
    tag = os.path.basename(os.path.dirname(f))
    N, seed, alias = parse_cell(tag, f)
    for c in df.columns:
        m = re.match(r"predpx_(\w+)$", c)
        if not m:
            continue
        arm = m.group(1)
        dcol = f"dice_{arm}"
        if dcol not in df.columns:
            continue
        sub = df[["key", "phase", "gt_px", dcol, c]].dropna(subset=[c]).copy()
        g = sub.gt_px.to_numpy(float)
        p = sub[c].to_numpy(float)
        d = sub[dcol].to_numpy(float)
        sub["_i"] = derive_inter(g, p, d)
        sub["_dp"] = np.where(g > 0, d, np.nan)
        a = sub.groupby(["key", "phase"]).agg(
            inter=("_i", "sum"), gt=("gt_px", "sum"), pred=(c, "sum"),
            present_dice=("_dp", "mean")).reset_index()
        emp = sub[sub.gt_px == 0].groupby(["key", "phase"])[c].sum()
        a = a.merge(emp.rename("empty_fp"), on=["key", "phase"], how="left")
        a["empty_fp"] = a["empty_fp"].fillna(0.0)
        a["stack_dice"] = 2 * a["inter"] / (a["gt"] + a["pred"]).replace(0, np.nan)
        a["arm"], a["N"], a["seed"], a["tag"], a["alias"] = arm, N, seed, tag, alias
        a["phase"] = a.phase.astype(str).str.lower()
        rows.append(a)

A = pd.concat(rows, ignore_index=True)
A = A[A.N.notna()].copy()
A["N"] = A.N.astype(int)
print(f"run 数 = {A.groupby(['tag','arm']).ngroups}   患者-相位行 = {len(A)}")

# ============================================================ 去重敏感性
hdr("0. 重复评估的数值差（别名 vs 正式），作为敏感性报告")
pv = A.groupby(["N", "seed", "arm", "alias", "phase"]).agg(
    stack=("stack_dice", "mean"), present=("present_dice", "mean"),
    empty=("empty_fp", "mean")).reset_index()
w = pv.pivot_table(index=["N", "seed", "arm", "phase"], columns="alias",
                   values=["stack", "present", "empty"])
both = w.dropna()
if len(both):
    for v in ["stack", "present", "empty"]:
        d = (both[(v, True)] - both[(v, False)])
        print(f"  {v:8s} 别名−正式:  n={len(d)}  mean={d.mean():+.6f}  "
              f"max|Δ|={d.abs().max():.6f}")
    print("  -> 上述量级即重跑评估的非确定性上界，写进 Methods 的可复现性说明")
else:
    print("  无同 (N,seed,arm) 的别名/正式配对，去重不影响任何 cell")

A = A[~A.alias].copy()          # 别名整体剔除，只留正式 run
print(f"剔除别名后 run 数 = {A.groupby(['tag','arm']).ngroups}")

# ============================================================ 逐 N 逐相位
hdr("1. B vs C：逐 N、逐相位、逐 seed（n<5 不报 CI，直列数值）")
cell = A.groupby(["N", "seed", "arm", "phase"]).agg(
    present=("present_dice", "mean"), stack=("stack_dice", "mean")).reset_index()
# overall = ED/ES 患者层平均
ov = A.groupby(["N", "seed", "arm"]).agg(
    present=("present_dice", "mean"), stack=("stack_dice", "mean")).reset_index()
ov["phase"] = "overall"
cell = pd.concat([cell, ov], ignore_index=True)

for phase in ["ed", "es", "overall"]:
    print(f"\n### phase = {phase}")
    c = cell[cell.phase == phase]
    for N in sorted(c.N.unique()):
        cc = c[c.N == N]
        w = cc.pivot_table(index="seed", columns="arm", values=["present", "stack"])
        if ("present", "B") not in w or ("present", "C") not in w:
            continue
        dp = (w[("present", "B")] - w[("present", "C")]).dropna()
        ds = (w[("stack", "B")] - w[("stack", "C")]).dropna()
        print(f"  N={N:<4d} present B−C: " +
              " ".join(f"s{i}={v:+.4f}" for i, v in dp.items()) +
              f"   mean={dp.mean():+.4f}")
        print(f"        stack   B−C: " +
              " ".join(f"s{i}={v:+.4f}" for i, v in ds.items()) +
              f"   mean={ds.mean():+.4f}" +
              ("   << 排序翻转" if dp.mean() * ds.mean() < 0 else ""))

# ============================================================ 空层假阳
hdr("2. 空层假阳性像素：是否随预算下降？（柱一/柱二的物理分解）")
e = A[A.phase == "es"].groupby(["N", "arm"])["empty_fp"].agg(
    mean="mean", median="median", q75=lambda x: x.quantile(.75),
    frac_gt100=lambda x: (x > 100).mean(), n="size").round(1)
print(e.to_string())
print("\n  逐臂对 N 的 Spearman（检验预算不敏感性）:")
from scipy.stats import spearmanr
for arm in sorted(A.arm.unique()):
    s = A[(A.phase == "es") & (A.arm == arm)]
    if s.N.nunique() < 3:
        continue
    r, p = spearmanr(s.N, s.empty_fp)
    print(f"    {arm:6s} rho={r:+.3f}  p={p:.3g}  (N 取值 {sorted(s.N.unique())})")

# ============================================================ 与 EF 的关联
hdr("3. 跨 run：present / stack Dice 谁更能预测 EF 误差？")
ef_rows = []
for f in sorted(glob.glob(os.path.join(ACDC, "eval*", "**", "test_ef_per_case.csv"),
                          recursive=True)):
    df = pd.read_csv(f)
    tag = os.path.basename(os.path.dirname(f))
    N, seed, alias = parse_cell(tag, f)
    if N is None or alias:
        continue
    for c in df.columns:
        m = re.match(r"ef_(\w+)$", c)
        if not m or m.group(1) in ("gt",):
            continue
        arm = m.group(1)
        mae = (df[c] - df["ef_gt"]).abs().mean()
        ef_rows.append(dict(N=N, seed=seed, arm=arm, ef_mae=mae))
E = pd.DataFrame(ef_rows)
M = ov.merge(E, on=["N", "seed", "arm"], how="inner")
print(f"  可配对的 run 数 = {len(M)}")
if len(M) > 5:
    for v in ["present", "stack"]:
        r, p = spearmanr(M[v], M.ef_mae)
        print(f"    {v:8s} vs EF-MAE:  Spearman rho={r:+.3f}  p={p:.3g}")
    # 控制 N 之后（组内去均值）
    Mc = M.copy()
    for v in ["present", "stack", "ef_mae"]:
        Mc[v] = Mc[v] - Mc.groupby("N")[v].transform("mean")
    print("  组内去均值（控制预算）后:")
    for v in ["present", "stack"]:
        r, p = spearmanr(Mc[v], Mc.ef_mae)
        print(f"    {v:8s} vs EF-MAE:  rho={r:+.3f}  p={p:.3g}")
M.to_csv("run_level_dice_vs_ef.csv", index=False)
A.to_csv("stack_dice_acdc_clean.csv", index=False)
print("\n输出: stack_dice_acdc_clean.csv, run_level_dice_vs_ef.csv")
print("不触碰冻结的 M&Ms-2 主终点。")
