#!/usr/bin/env python3
"""
stepE06b -- is the protocol gradient A < B < C of 3.5 / A9-R1 / A9-R2 stable across seeds

Motivation (written before execution):
  stepE06 found that the per-seed median of arm A at N=100/es is 467 / 0 / 90, while the
  pooled value is 33. Since the pooled median is close to none of the individual seeds,
  whether the protocol gradient "A=33 < B=91 < C=906" on which 3.5 rests actually holds
  becomes a question that must be answered, not an optional supplementary check.

  This remains a read-only decomposition of an already registered quantity (the median of the
  per-case empty-slice FP volume) and introduces no new class of statistic.

Pre-registered reading (written before the data were seen):
  S1  A<B<C holds in all three seeds -> the damage to 3.5 is confined to the point-value
      wording, and the gradient conclusion can be kept
  S2  the gradient reverses in some seeds -> the gradient statement of 3.5 cannot itself be
      written as fact, and A9-R1/A9-R2 must be recorded as corrected in the ledger
"""
import pandas as pd
import numpy as np

import os as _os
_DATA = _os.environ.get("FROZEN_DATA_DIR", _os.path.dirname(_os.path.abspath(__file__)))
df = pd.read_csv(_os.path.join(_DATA, "stack_dice_acdc_clean.csv"))
ARMS = ["A", "B", "C"]

print("=" * 72)
print("1. N=100 (largest budget level), es phase -- the very cell quoted in the opening of 3.5")
print("=" * 72)
sub = df[(df['N'] == 100) & (df['phase'] == 'es')]
print(f"{'seed':<8}" + "".join(f"{'arm '+a:>12}" for a in ARMS) + f"{'  ordering':<20}")
for s in sorted(sub['seed'].unique()):
    meds = {}
    for a in ARMS:
        x = sub[(sub['arm'] == a) & (sub['seed'] == s)]['empty_fp'].astype(float)
        meds[a] = float(np.median(x))
    order = " < ".join(sorted(ARMS, key=lambda a: meds[a]))
    flag = "" if order == "A < B < C" else "   <-- disagrees with the registered gradient"
    print(f"{s:<8}" + "".join(f"{meds[a]:>12.1f}" for a in ARMS) + f"  {order}{flag}")

meds_p = {a: float(np.median(sub[sub['arm'] == a]['empty_fp'].astype(float)))
          for a in ARMS}
order_p = " < ".join(sorted(ARMS, key=lambda a: meds_p[a]))
print(f"{'pooled':<8}" + "".join(f"{meds_p[a]:>12.1f}" for a in ARMS) + f"  {order_p}")

print()
print("=" * 72)
print("2. Per budget level: in how many seeds does A<B<C hold (es phase)")
print("=" * 72)
print(f"{'N':<6}{'seed':<6}" + "".join(f"{'arm '+a:>11}" for a in ARMS)
      + "   A<B?   B<C?   full gradient")
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
              + ("OK" if full else "**reversed**"))
    if n_seed:
        summary.append((N, s_ok, n_seed))
    print("-" * 72)

print()
print("=" * 72)
print("3. Summary: number of seeds in which the full gradient A<B<C holds / total seeds at that level (es phase)")
print("=" * 72)
for N, ok, tot in summary:
    print(f"  N = {N:<5} {ok}/{tot}")

print()
print("=" * 72)
print("4. Seed-level stability of B and C (control: do they swing as A does)")
print("=" * 72)
print(f"{'arm':<6}{'N':<6}{'pooled':>10}{'per-seed medians':>34}{'range/pooled':>14}")
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
