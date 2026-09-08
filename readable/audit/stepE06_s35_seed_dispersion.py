#!/usr/bin/env python3
"""
stepE06 -- read-only reproduction of the seed-level dispersion behind Results 3.5

Statement of nature:
  This script is read-only with respect to the frozen table stack_dice_acdc_clean.csv
  (SHA256 d29f91f36fb67a4e7554767631ac4ad8fa26ae9a82efb688e6eb428fe7ff7dae).
  It adds no grouping scope, changes no aggregation definition and introduces no new class
  of statistic. The only operation performed is to split the pooled descriptive quantities
  already registered in 3.5 by seed, which is of the same kind as the decomposition
  PATCH_MECHANICAL_SIX 0 already performed on frac_gt100.

Pre-registered branches (written before execution; see the 2026-08-17 session):
  B1  the seed-level dispersion of the three quantities is of the same order as the point
      estimate -> rewrite the whole of 3.5 as a dispersion/interval statement and report no
      pooled point value
  B2  the three quantities are markedly more stable -> keep the point values and note that
      frac_gt100 is the only quantity in that passage that is broken

Gate: first reproduce the four values already registered in 0. Any mismatch aborts, and the
decomposition is not entered.
"""
import sys
import pandas as pd
import numpy as np

import os as _os
_DATA = _os.environ.get("FROZEN_DATA_DIR", _os.path.dirname(_os.path.abspath(__file__)))
CSV = _os.path.join(_DATA, "stack_dice_acdc_clean.csv")
df = pd.read_csv(CSV)

print("=" * 68)
print("0. basic shape of the frozen table")
print("=" * 68)
print(f"rows={len(df)}  cols={len(df.columns)}")
print("columns:", list(df.columns))
print("arm values  :", sorted(df['arm'].unique()))
print("N   values  :", sorted(df['N'].unique()))
print("seed values :", sorted(df['seed'].unique()))
print("phase values:", sorted(df['phase'].unique()))
print("alias values:", sorted(df['alias'].unique().tolist()))

# ---------------------------------------------------------------- target slice
# 3.5 / A9-R1: arm A, largest budget level (N=100 label), ES phase
sel = df[(df['arm'] == 'A') & (df['N'] == 100) & (df['phase'] == 'es')]

print()
print("=" * 68)
print("1. Gate: reproduce the four values already registered in PATCH_MECHANICAL_SIX 0")
print("=" * 68)
print(f"slice arm=A, N=100, phase=es  ->  {len(sel)} rows"
      f"  (expected 150 = 50 patients x 3 seeds)")

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
    print(f"  {k:<11} measured {got[k]:>10.4f}   registered {exp[k]:>9.4f}   "
          f"{'PASS' if hit else '**MISMATCH**'}")

print()
print("  per-seed frac_gt100 (registered seed0=0.5600 seed1=0.2000 seed2=0.4800):")
exp_frac = {0: 0.56, 1: 0.20, 2: 0.48}
for s in sorted(sel['seed'].unique()):
    vs = sel[sel['seed'] == s]['empty_fp'].astype(float)
    f = float((vs > 100).mean())
    hit = abs(f - exp_frac[s]) < 0.0005
    ok &= hit
    print(f"    seed {s}: n={len(vs):>3}  frac={f:.4f}  registered={exp_frac[s]:.4f}  "
          f"{'PASS' if hit else '**MISMATCH**'}")

if not ok:
    print("\n!! Gate not passed. The extraction scope disagrees with the registered values; aborting without entering the decomposition.")
    sys.exit(1)
print("\n  Gate PASS -- the extraction scope agrees exactly with the registered values.")

# ---------------------------------------------------------------- decomposition
print()
print("=" * 68)
print("2. Seed-level decomposition of the three pooled descriptive quantities of 3.5 (what is newly seen this round)")
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
print("3. Dispersion vs point estimate (branch criterion)")
print("-" * 68)
print(f"{'quantity':<12}{'pooled':>10}{'min':>10}{'max':>10}{'range':>10}{'range/pooled':>14}")
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
print("Criterion: range/pooled >= 0.5 counts as 'of the same order as the point estimate' (frac_gt100 is 0.87,")
print("      i.e. the one you have already judged cannot be written as a point value)")
for k in ("median", "mean", "q75"):
    print(f"  {k:<8} range/pooled = {verdict[k]:.2f}  ->  "
          f"{'same order, no point value' if verdict[k] >= 0.5 else 'relatively stable'}")

# ---------------------------------------------------------------- corroboration
print()
print("=" * 68)
print("4. Corroboration: do the other two numbers in the same passage (the medians of B / C in the same cell) also swing")
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
print("5. ED-phase control (D-B: is the ED-phase median zero for every arm)")
print("=" * 68)
for arm in ("A", "B", "C"):
    s3 = df[(df['arm'] == arm) & (df['N'] == 100) & (df['phase'] == 'ed')]
    if not len(s3):
        continue
    pooled = float(np.median(s3['empty_fp'].astype(float)))
    per = [float(np.median(s3[s3['seed'] == s]['empty_fp'].astype(float)))
           for s in sorted(s3['seed'].unique())]
    print(f"  arm {arm}: pooled median = {pooled:>8.1f}   per-seed = {per}")
