#!/usr/bin/env python3
"""
STEP 0.6  --  zero GPU, read-only
Fixes the three defects of step05 and adds one key association analysis:
  (a) seed parsing: a directory without the _s suffix = seed 0 (silently dropped by
      pivot_table in the previous round)
  (b) de-duplication: content hashing fails (re-evaluation carries ~1e-4 non-determinism),
      so aliases are matched explicitly by (N, seed) instead, and the numerical difference
      between the two evaluations is reported as a sensitivity result -- that difference is
      itself data to be written into the paper
  (c) when n<5, no bootstrap CI is reported; the per-seed values are listed directly
  (d) new: the association across runs between stack/present Dice and EF-MAE
      -- answering directly whether stack Dice is the better endpoint proxy

Usage:
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
    """Return (N, seed, is_alias). No _s suffix = seed 0. eval* is treated as an alias."""
    m = re.search(r"N(\d+)", tag) or re.search(r"N(\d+)", path)
    N = int(m.group(1)) if m else None
    s = re.search(r"_s(\d+)", tag)
    seed = int(s.group(1)) if s else 0
    alias = tag.startswith("eval")
    if tag == "eval":          # snapshot: eval is the repeated evaluation of N100 seed0
        N, seed = 100, 0
    return N, seed, alias


# ============================================================ read all runs
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
print(f"runs = {A.groupby(['tag','arm']).ngroups}   patient-phase rows = {len(A)}")

# ============================================================ de-duplication sensitivity
hdr("0. Numerical difference of the repeated evaluation (alias vs formal), as a sensitivity report")
pv = A.groupby(["N", "seed", "arm", "alias", "phase"]).agg(
    stack=("stack_dice", "mean"), present=("present_dice", "mean"),
    empty=("empty_fp", "mean")).reset_index()
w = pv.pivot_table(index=["N", "seed", "arm", "phase"], columns="alias",
                   values=["stack", "present", "empty"])
both = w.dropna()
if len(both):
    for v in ["stack", "present", "empty"]:
        d = (both[(v, True)] - both[(v, False)])
        print(f"  {v:8s} alias-formal:  n={len(d)}  mean={d.mean():+.6f}  "
              f"max|Δ|={d.abs().max():.6f}")
    print("  -> the magnitude above is the upper bound on the non-determinism of a re-run evaluation; write it into the reproducibility note of the Methods")
else:
    print("  no alias/formal pair with the same (N,seed,arm); de-duplication affects no cell")

A = A[~A.alias].copy()          # aliases are dropped wholesale, keeping only the formal runs
print(f"runs after dropping aliases = {A.groupby(['tag','arm']).ngroups}")

# ============================================================ per N, per phase
hdr("1. B vs C: per N, per phase, per seed (no CI reported when n<5; values listed directly)")
cell = A.groupby(["N", "seed", "arm", "phase"]).agg(
    present=("present_dice", "mean"), stack=("stack_dice", "mean")).reset_index()
# overall = the patient-level mean of ED/ES
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
              ("   << ranking reversal" if dp.mean() * ds.mean() < 0 else ""))

# ============================================================ empty-slice false positives
hdr("2. Empty-slice false-positive pixels: do they fall with the budget? (the physical decomposition of pillars one and two)")
e = A[A.phase == "es"].groupby(["N", "arm"])["empty_fp"].agg(
    mean="mean", median="median", q75=lambda x: x.quantile(.75),
    frac_gt100=lambda x: (x > 100).mean(), n="size").round(1)
print(e.to_string())
print("\n  Per-arm Spearman against N (testing budget insensitivity):")
from scipy.stats import spearmanr
for arm in sorted(A.arm.unique()):
    s = A[(A.phase == "es") & (A.arm == arm)]
    if s.N.nunique() < 3:
        continue
    r, p = spearmanr(s.N, s.empty_fp)
    print(f"    {arm:6s} rho={r:+.3f}  p={p:.3g}  (N values {sorted(s.N.unique())})")

# ============================================================ association with EF
hdr("3. Across runs: which of present / stack Dice better predicts the EF error?")
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
print(f"  pairable runs = {len(M)}")
if len(M) > 5:
    for v in ["present", "stack"]:
        r, p = spearmanr(M[v], M.ef_mae)
        print(f"    {v:8s} vs EF-MAE:  Spearman rho={r:+.3f}  p={p:.3g}")
    # after controlling for N (within-group centring)
    Mc = M.copy()
    for v in ["present", "stack", "ef_mae"]:
        Mc[v] = Mc[v] - Mc.groupby("N")[v].transform("mean")
    print("  after within-group centring (controlling for the budget):")
    for v in ["present", "stack"]:
        r, p = spearmanr(Mc[v], Mc.ef_mae)
        print(f"    {v:8s} vs EF-MAE:  rho={r:+.3f}  p={p:.3g}")
M.to_csv("run_level_dice_vs_ef.csv", index=False)
A.to_csv("stack_dice_acdc_clean.csv", index=False)
print("\noutput: stack_dice_acdc_clean.csv, run_level_dice_vs_ef.csv")
print("Does not touch the frozen M&Ms-2 primary endpoint.")
