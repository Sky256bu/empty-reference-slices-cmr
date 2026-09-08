#!/usr/bin/env python3
"""
stepE06b — §3.5 / A9-R1 / A9-R2 的协议梯度 A < B < C 是否 seed 稳定

动机（写在执行前）：
  stepE06 发现 arm A 在 N=100/es 的 per-seed median 为 467 / 0 / 90，
  而 pooled 为 33。既然 pooled 中位数与任一 seed 都不接近，
  §3.5 赖以成立的 "A=33 < B=91 < C=906" 协议梯度是否成立就成为
  一个必须回答的问题，而不是可选的补充检查。

  这仍是对已登记量（per-case empty-slice FP volume 的中位数）的
  只读分解，不新增统计量类别。

预注册判读（写于看数之前）：
  S1  三个 seed 中 A<B<C 全部成立 -> §3.5 受损范围仅限点值写法，梯度结论可保留
  S2  梯度在部分 seed 上反转       -> §3.5 的梯度陈述本身不可写成事实，
                                     且 A9-R1/A9-R2 须在台账中登记更正
"""
import pandas as pd
import numpy as np

import os as _os
_DATA = _os.environ.get("FROZEN_DATA_DIR", _os.path.dirname(_os.path.abspath(__file__)))
df = pd.read_csv(_os.path.join(_DATA, "stack_dice_acdc_clean.csv"))
ARMS = ["A", "B", "C"]

print("=" * 72)
print("1. N=100 (最大预算档), es 相 —— §3.5 首段直接引用的那一格")
print("=" * 72)
sub = df[(df['N'] == 100) & (df['phase'] == 'es')]
print(f"{'seed':<8}" + "".join(f"{'arm '+a:>12}" for a in ARMS) + f"{'  排序':<20}")
for s in sorted(sub['seed'].unique()):
    meds = {}
    for a in ARMS:
        x = sub[(sub['arm'] == a) & (sub['seed'] == s)]['empty_fp'].astype(float)
        meds[a] = float(np.median(x))
    order = " < ".join(sorted(ARMS, key=lambda a: meds[a]))
    flag = "" if order == "A < B < C" else "   <-- 与登记梯度不符"
    print(f"{s:<8}" + "".join(f"{meds[a]:>12.1f}" for a in ARMS) + f"  {order}{flag}")

meds_p = {a: float(np.median(sub[sub['arm'] == a]['empty_fp'].astype(float)))
          for a in ARMS}
order_p = " < ".join(sorted(ARMS, key=lambda a: meds_p[a]))
print(f"{'pooled':<8}" + "".join(f"{meds_p[a]:>12.1f}" for a in ARMS) + f"  {order_p}")

print()
print("=" * 72)
print("2. 逐预算档：A<B<C 在多少个 seed 上成立（es 相）")
print("=" * 72)
print(f"{'N':<6}{'seed':<6}" + "".join(f"{'arm '+a:>11}" for a in ARMS)
      + "   A<B?   B<C?   完整梯度")
summary = []
for N in sorted(df['N'].unique()):
    s_ok = 0
    n_seed = 0
    for s in sorted(df['seed'].unique()):
        cell = df[(df['N'] == N) & (df['phase'] == 'es') & (df['seed'] == s)]
        if not all(len(cell[cell['arm'] == a]) for a in ARMS):
            continue
        n_seed += 1
        meds = {a: float(np.median(cell[cell['arm'] == a]['empty_fp'].astype(float)))
                for a in ARMS}
        ab = meds['A'] < meds['B']
        bc = meds['B'] < meds['C']
        full = ab and bc
        s_ok += full
        print(f"{N:<6}{s:<6}" + "".join(f"{meds[a]:>11.1f}" for a in ARMS)
              + f"   {'Y' if ab else 'N':<6} {'Y' if bc else 'N':<6} "
              + ("OK" if full else "**反转**"))
    if n_seed:
        summary.append((N, s_ok, n_seed))
    print("-" * 72)

print()
print("=" * 72)
print("3. 汇总：完整梯度 A<B<C 成立的 seed 数 / 该档 seed 总数（es 相）")
print("=" * 72)
for N, ok, tot in summary:
    print(f"  N = {N:<5} {ok}/{tot}")

print()
print("=" * 72)
print("4. B 与 C 的 seed 级稳定性（对照：它们是否也像 A 一样摆动）")
print("=" * 72)
print(f"{'arm':<6}{'N':<6}{'pooled':>10}{'per-seed medians':>34}{'极差/pooled':>14}")
for a in ARMS:
    for N in [100]:
        cell = df[(df['N'] == N) & (df['phase'] == 'es') & (df['arm'] == a)]
        per = [float(np.median(cell[cell['seed'] == s]['empty_fp'].astype(float)))
               for s in sorted(cell['seed'].unique())]
        pooled = float(np.median(cell['empty_fp'].astype(float)))
        rng = max(per) - min(per)
        ratio = rng / pooled if pooled else float('inf')
        print(f"{a:<6}{N:<6}{pooled:>10.1f}{str([round(x,1) for x in per]):>34}"
              f"{ratio:>14.2f}")
