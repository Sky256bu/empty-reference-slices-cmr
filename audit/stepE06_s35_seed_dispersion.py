#!/usr/bin/env python3
"""
stepE06 — Results §3.5 的 seed 级离散度只读复现

性质声明：
  本脚本对冻结表 stack_dice_acdc_clean.csv
  (SHA256 d29f91f36fb67a4e7554767631ac4ad8fa26ae9a82efb688e6eb428fe7ff7dae)
  只读。不新增分组口径、不改变聚合定义、不引入新统计量类别。
  所做的唯一操作是：把 §3.5 已登记的 pooled 描述量按 seed 拆开，
  与 PATCH_MECHANICAL_SIX §0 已对 frac_gt100 做过的分解属同一类。

预注册分支（写于执行之前，见 2026-08-17 会话）：
  B1  三个量的 seed 级离散与点估计同量级 -> §3.5 整段改为离散/区间陈述，不报 pooled 点值
  B2  三个量显著更稳                     -> 保留点值，并说明 frac_gt100 是该段唯一被击穿的量

闸门：先复现 §0 已登记的四个值。任一不符即中止，不进入分解。
"""
import sys
import pandas as pd
import numpy as np

import os as _os
_DATA = _os.environ.get("FROZEN_DATA_DIR", _os.path.dirname(_os.path.abspath(__file__)))
CSV = _os.path.join(_DATA, "stack_dice_acdc_clean.csv")
df = pd.read_csv(CSV)

print("=" * 68)
print("0. 冻结表基本形状")
print("=" * 68)
print(f"rows={len(df)}  cols={len(df.columns)}")
print("columns:", list(df.columns))
print("arm 取值 :", sorted(df['arm'].unique()))
print("N   取值 :", sorted(df['N'].unique()))
print("seed取值 :", sorted(df['seed'].unique()))
print("phase取值:", sorted(df['phase'].unique()))
print("alias 取值:", sorted(df['alias'].unique().tolist()))

# ---------------------------------------------------------------- 目标切片
# §3.5 / A9-R1: arm A, 最大预算档 (N=100 label), ES 相
sel = df[(df['arm'] == 'A') & (df['N'] == 100) & (df['phase'] == 'es')]

print()
print("=" * 68)
print("1. 闸门：复现 PATCH_MECHANICAL_SIX §0 已登记的四个值")
print("=" * 68)
print(f"切片 arm=A, N=100, phase=es  ->  {len(sel)} 行"
      f"  (期望 150 = 50 患者 x 3 seed)")

v = sel['empty_fp'].astype(float)
got = {
    "median": float(np.median(v)),
    "mean": float(v.mean()),
    "q75": float(np.percentile(v, 75)),
    "frac_gt100": float((v > 100).mean()),
}
exp = {"median": 33.0, "mean": 407.9, "q75": 746.2, "frac_gt100": 0.4133}

ok = True
for k in exp:
    hit = abs(got[k] - exp[k]) < (0.05 if k != "frac_gt100" else 0.0005)
    ok &= hit
    print(f"  {k:<11} 实测 {got[k]:>10.4f}   登记 {exp[k]:>9.4f}   "
          f"{'PASS' if hit else '**MISMATCH**'}")

print()
print("  逐 seed frac_gt100（登记 seed0=0.5600 seed1=0.2000 seed2=0.4800）：")
exp_frac = {0: 0.56, 1: 0.20, 2: 0.48}
for s in sorted(sel['seed'].unique()):
    vs = sel[sel['seed'] == s]['empty_fp'].astype(float)
    f = float((vs > 100).mean())
    hit = abs(f - exp_frac[s]) < 0.0005
    ok &= hit
    print(f"    seed {s}: n={len(vs):>3}  frac={f:.4f}  登记={exp_frac[s]:.4f}  "
          f"{'PASS' if hit else '**MISMATCH**'}")

if not ok:
    print("\n!! 闸门未通过。抽取口径与已登记值不符，中止，不进入分解。")
    sys.exit(1)
print("\n  闸门 PASS —— 抽取口径与已登记值完全一致。")

# ---------------------------------------------------------------- 分解
print()
print("=" * 68)
print("2. §3.5 三个 pooled 描述量的 seed 级分解（本轮新看到的部分）")
print("=" * 68)
print(f"{'seed':<6}{'n':>5}{'median':>12}{'mean':>12}{'q75':>12}{'frac>100':>11}")
rows = []
for s in sorted(sel['seed'].unique()):
    vs = sel[sel['seed'] == s]['empty_fp'].astype(float)
    r = dict(seed=int(s), n=len(vs),
             median=float(np.median(vs)), mean=float(vs.mean()),
             q75=float(np.percentile(vs, 75)), frac=float((vs > 100).mean()))
    rows.append(r)
    print(f"{r['seed']:<6}{r['n']:>5}{r['median']:>12.1f}{r['mean']:>12.1f}"
          f"{r['q75']:>12.1f}{r['frac']:>11.4f}")

print(f"{'pooled':<6}{len(v):>5}{got['median']:>12.1f}{got['mean']:>12.1f}"
      f"{got['q75']:>12.1f}{got['frac_gt100']:>11.4f}")

print()
print("3. 离散度 vs 点估计（分支判据）")
print("-" * 68)
print(f"{'量':<12}{'pooled':>10}{'min':>10}{'max':>10}{'极差':>10}{'极差/pooled':>13}")
verdict = {}
for k in ("median", "mean", "q75", "frac"):
    vals = [r[k] for r in rows]
    rng = max(vals) - min(vals)
    base = got['frac_gt100'] if k == 'frac' else got[k]
    ratio = rng / base if base else float('nan')
    verdict[k] = ratio
    print(f"{k:<12}{base:>10.2f}{min(vals):>10.2f}{max(vals):>10.2f}"
          f"{rng:>10.2f}{ratio:>13.2f}")

print()
print("判据：极差/pooled >= 0.5 视为『与点估计同量级』（frac_gt100 为 0.87，")
print("      即已被你判定为不可写成点值的那一档）")
for k in ("median", "mean", "q75"):
    print(f"  {k:<8} 极差/pooled = {verdict[k]:.2f}  ->  "
          f"{'同量级，不可写点值' if verdict[k] >= 0.5 else '相对稳定'}")

# ---------------------------------------------------------------- 旁证
print()
print("=" * 68)
print("4. 旁证：同段另外两个数（B / C 在同一 cell 的中位数）是否也摆动")
print("=" * 68)
for arm in ("A", "B", "C"):
    s2 = df[(df['arm'] == arm) & (df['N'] == 100) & (df['phase'] == 'es')]
    if not len(s2):
        continue
    per = [float(np.median(s2[s2['seed'] == s]['empty_fp'].astype(float)))
           for s in sorted(s2['seed'].unique())]
    pooled = float(np.median(s2['empty_fp'].astype(float)))
    print(f"  arm {arm}: pooled median = {pooled:>8.1f}   "
          f"per-seed = {[round(x,1) for x in per]}")

print()
print("=" * 68)
print("5. ED 相对照（⟦D-B⟧：ED 相中位数是否每个臂都为零）")
print("=" * 68)
for arm in ("A", "B", "C"):
    s3 = df[(df['arm'] == arm) & (df['N'] == 100) & (df['phase'] == 'ed')]
    if not len(s3):
        continue
    pooled = float(np.median(s3['empty_fp'].astype(float)))
    per = [float(np.median(s3[s3['seed'] == s]['empty_fp'].astype(float)))
           for s in sorted(s3['seed'].unique())]
    print(f"  arm {arm}: pooled median = {pooled:>8.1f}   per-seed = {per}")
