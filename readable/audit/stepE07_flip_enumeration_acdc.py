#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stepE07 — Complete enumeration of scope-induced ordering reversals (ACDC).

STATUS: post-hoc, uncorrected for multiplicity. This script EXTENDS the
existing post-hoc mechanistic analysis of Results 3.2-3.4 from two reported
instances to a complete enumeration. It introduces no new training, no new
seed, no new model and no new statistic beyond the three Dice definitions
already declared in Methods 2.6. It must be registered, hashed and declared
BEFORE execution, in the same manner as stepE02-stepE06.

WHAT IT DOES NOT DO
  - It does not touch the confirmatory M&Ms-2 endpoint.
  - It makes no cross-partition-regime claim. The 63 formal runs span two
    patient-partition regimes (low N in {5,10,15}: seed-dependent; high N in
    {25,50,100}: frozen nested). Every output row carries its regime, and no
    aggregate is computed across the boundary.
  - It does not compare slice-present against full-stack as if the difference
    were attributable to scope. That contrast differs in BOTH aggregation and
    scope and is emitted separately, flagged non_attributable.

THE IDENTITY THIS SCRIPT IS BUILT ON
  full_stack = volume_present - empty_fp_penalty          (per case-phase)
  => for any two arms X, Y evaluated on the same cases:
       delta_full_stack = delta_volume_present - delta_penalty
  An ordering reversal between the two volumetric scopes therefore occurs
  EXACTLY when |delta_volume_present| < |delta_penalty| and the two have the
  same sign. This is exact, not approximate, and it is what turns "we observed
  two reversals" into a characterisation of when reversals are possible at all.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 0. Declared provenance. Fill DECLARED_SHA256 from the frozen record and the
#    script will refuse to run against any other file.
# ----------------------------------------------------------------------------

IN_CSV = Path("stack_dice_acdc_clean.csv")
DECLARED_SHA256 = "d29f91f3"          # prefix as recorded; set to full 64 hex to enforce exactly
OUT_DIR = Path("stepE07_out")

# Partition regimes (Methods 2.8). No aggregate crosses this boundary.
REGIME = {5: "low_seed_dependent", 10: "low_seed_dependent", 15: "low_seed_dependent",
          25: "high_frozen_nested", 50: "high_frozen_nested", 100: "high_frozen_nested"}

# Arm I exists only at N in {5,10,15}; pairs involving I are confined there.
ARMS = ["A", "B", "C", "I"]

TOL = 1e-12


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# 1. Load, verify provenance, reconstruct the three scopes.
# ----------------------------------------------------------------------------

def load_and_reconstruct(path: Path) -> tuple[pd.DataFrame, dict]:
    digest = sha256_of(path)
    if DECLARED_SHA256 and not digest.startswith(DECLARED_SHA256.rstrip("\u2026.")):
        sys.exit(f"ABORT: input SHA256 {digest} does not match declared {DECLARED_SHA256}")

    d = pd.read_csv(path)

    # Alias de-duplication is already applied in the frozen table; assert it.
    if "alias" in d.columns and bool(d["alias"].any()):
        sys.exit("ABORT: alias rows present; this table is not the de-duplicated freeze")

    # Methods 2.6. gt is accumulated over reference-present slices by
    # construction; pred is accumulated over ALL slices; empty_fp is the
    # predicted foreground falling on reference-empty slices.
    d["volume_present"] = 2 * d["inter"] / (d["gt"] + d["pred"] - d["empty_fp"])
    d["full_stack"] = 2 * d["inter"] / (d["gt"] + d["pred"])
    d["slice_present"] = d["present_dice"]          # mean-of-slices, NOT volumetric
    d["penalty"] = d["volume_present"] - d["full_stack"]

    checks = {
        "input_sha256": digest,
        "n_rows": int(len(d)),
        "n_runs": int(d.groupby(["arm", "N", "seed"]).ngroups),
        # reconstruction must agree bitwise-ish with the frozen full-stack column
        "max_abs_dev_full_stack_vs_frozen_column":
            float(np.nanmax(np.abs(d["full_stack"] - d["stack_dice"]))),
        "penalty_nonnegative_everywhere": bool((d["penalty"] >= -TOL).all()),
        "penalty_min": float(d["penalty"].min()),
    }
    if checks["max_abs_dev_full_stack_vs_frozen_column"] > 1e-9:
        sys.exit("ABORT: reconstructed full-stack Dice disagrees with frozen column")
    if not checks["penalty_nonnegative_everywhere"]:
        sys.exit("ABORT: negative empty-slice penalty; scope definitions are inconsistent")
    return d, checks


# ----------------------------------------------------------------------------
# 2. Cell means. A cell is (arm, N, seed, phase), averaged over the 50 held-out
#    patients. This is the unit at which 3.3/3.4 report.
# ----------------------------------------------------------------------------

def cell_means(d: pd.DataFrame) -> pd.DataFrame:
    cols = ["slice_present", "volume_present", "full_stack", "penalty", "empty_fp"]
    c = (d.groupby(["arm", "N", "seed", "phase"])[cols]
           .mean().reset_index())
    c["n_cases"] = (d.groupby(["arm", "N", "seed", "phase"]).size()
                      .reset_index(drop=True).values)
    c["regime"] = c["N"].map(REGIME)
    return c


# ----------------------------------------------------------------------------
# 3. Enumerate every arm pair in every (N, seed, phase).
# ----------------------------------------------------------------------------

def enumerate_pairs(cell: pd.DataFrame) -> pd.DataFrame:
    idx = cell.set_index(["arm", "N", "seed", "phase"])
    rows = []
    for N in sorted(cell["N"].unique()):
        for seed in sorted(cell["seed"].unique()):
            for phase in ["ed", "es"]:
                present = [a for a in ARMS if (a, N, seed, phase) in idx.index]
                for x, y in itertools.combinations(present, 2):
                    rx, ry = idx.loc[(x, N, seed, phase)], idx.loc[(y, N, seed, phase)]
                    d_vp = rx["volume_present"] - ry["volume_present"]
                    d_fs = rx["full_stack"] - ry["full_stack"]
                    d_sp = rx["slice_present"] - ry["slice_present"]
                    d_pen = rx["penalty"] - ry["penalty"]

                    # exactness check on the decomposition
                    resid = abs(d_fs - (d_vp - d_pen))

                    rows.append(dict(
                        N=N, seed=seed, phase=phase, regime=REGIME[N],
                        arm_x=x, arm_y=y, pair=f"{x}-{y}",
                        delta_volume_present=d_vp,
                        delta_full_stack=d_fs,
                        delta_penalty=d_pen,
                        delta_slice_present=d_sp,
                        decomposition_residual=resid,
                        abs_delta_volume_present=abs(d_vp),
                        abs_delta_penalty=abs(d_pen),
                        # the exact flip condition, scope-only contrast
                        flippable=bool(abs(d_vp) < abs(d_pen)),
                        flip_scope_only=bool(np.sign(d_vp) * np.sign(d_fs) < 0),
                        # aggregation+scope: reported, never attributed to scope
                        flip_nonattributable=bool(np.sign(d_sp) * np.sign(d_fs) < 0),
                        # how close to a tie the comparison was, in units of the
                        # empty-slice penalty actually present in that cell
                        tie_ratio=abs(d_vp) / max(abs(d_pen), 1e-12),
                    ))
    out = pd.DataFrame(rows)
    if out["decomposition_residual"].max() > 1e-9:
        sys.exit("ABORT: delta_full_stack != delta_volume_present - delta_penalty")
    return out


# ----------------------------------------------------------------------------
# 4. Cell-level (N, phase, pair) summary: mean over the three seeds, plus the
#    per-seed flip count that 3.4 reports as "3/3 seeds".
# ----------------------------------------------------------------------------

def pair_cells(pairs: pd.DataFrame) -> pd.DataFrame:
    g = pairs.groupby(["N", "phase", "pair", "regime"])
    s = g.agg(
        n_seeds=("seed", "nunique"),
        mean_delta_volume_present=("delta_volume_present", "mean"),
        mean_delta_full_stack=("delta_full_stack", "mean"),
        mean_delta_penalty=("delta_penalty", "mean"),
        mean_delta_slice_present=("delta_slice_present", "mean"),
        n_seed_flips_scope_only=("flip_scope_only", "sum"),
        n_seed_flippable=("flippable", "sum"),
        mean_tie_ratio=("tie_ratio", "mean"),
    ).reset_index()
    s["mean_level_flip_scope_only"] = (
        np.sign(s["mean_delta_volume_present"]) * np.sign(s["mean_delta_full_stack"]) < 0)
    s["seed_unanimous"] = s["n_seed_flips_scope_only"] == s["n_seeds"]
    return s


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    d, checks = load_and_reconstruct(IN_CSV)
    cell = cell_means(d)
    pairs = enumerate_pairs(cell)
    cells = pair_cells(pairs)

    cell.to_csv(OUT_DIR / "stepE07_cell_means.csv", index=False)
    pairs.to_csv(OUT_DIR / "stepE07_pairwise_all.csv", index=False)
    cells.to_csv(OUT_DIR / "stepE07_pair_cell_summary.csv", index=False)

    # ---- readout -----------------------------------------------------------
    n_cmp = len(pairs)
    n_flip = int(pairs["flip_scope_only"].sum())
    n_flippable = int(pairs["flippable"].sum())
    L = []
    L.append("# stepE07 readout — complete scope-flip enumeration (ACDC, post-hoc)\n")
    L.append("## Provenance and self-checks\n")
    for k, v in checks.items():
        L.append(f"- `{k}` = `{v}`")
    L.append("")
    L.append("## Denominators\n")
    L.append(f"- seed-level comparisons enumerated: **{n_cmp}**")
    L.append(f"- of which structurally flippable (|Δ_vp| < |Δ_penalty|): "
             f"**{n_flippable}** ({n_flippable / n_cmp:.1%})")
    L.append(f"- of which actually reversed under scope-only contrast: "
             f"**{n_flip}** ({n_flip / n_cmp:.1%})")
    L.append(f"- (N, phase, pair) cells: **{len(cells)}**, of which mean-level "
             f"reversal in **{int(cells['mean_level_flip_scope_only'].sum())}**")
    L.append("")

    L.append("## Mean-level reversals, by cell (scope-only, attributable)\n")
    hit = cells[cells["mean_level_flip_scope_only"]].sort_values(
        ["regime", "N", "phase", "pair"])
    if hit.empty:
        L.append("_none_")
    else:
        L.append("| regime | N | phase | pair | Δ volume-present | Δ full-stack | "
                 "Δ penalty | seeds flipped |")
        L.append("|---|---|---|---|---|---|---|---|")
        for _, r in hit.iterrows():
            L.append(f"| {r.regime} | {r.N} | {r.phase} | {r['pair']} | "
                     f"{r.mean_delta_volume_present:+.6f} | {r.mean_delta_full_stack:+.6f} | "
                     f"{r.mean_delta_penalty:+.6f} | "
                     f"{int(r.n_seed_flips_scope_only)}/{int(r.n_seeds)} |")
    L.append("")

    L.append("## Sign-consistency table, scope-only contrast, by regime and phase\n")
    L.append("| regime | phase | comparisons | flippable | reversed | median |Δ_vp| | "
             "median |Δ_penalty| |")
    L.append("|---|---|---|---|---|---|---|")
    for (reg, ph), g in pairs.groupby(["regime", "phase"]):
        L.append(f"| {reg} | {ph} | {len(g)} | {int(g['flippable'].sum())} | "
                 f"{int(g['flip_scope_only'].sum())} | "
                 f"{g['abs_delta_volume_present'].median():.6f} | "
                 f"{g['abs_delta_penalty'].median():.6f} |")
    L.append("")

    L.append("## Non-attributable contrast (slice-present vs full-stack)\n")
    L.append("Reported for completeness only. This contrast differs in both "
             "aggregation and scope; a sign change under it cannot be attributed "
             "to the admission of reference-empty false positives.\n")
    L.append(f"- reversed in **{int(pairs['flip_nonattributable'].sum())}** of "
             f"{n_cmp} seed-level comparisons")
    L.append("")

    L.append("## Magnitude context for the near-tie objection\n")
    L.append("`tie_ratio` = |Δ volume-present| / |Δ penalty|. A reversal requires "
             "tie_ratio < 1 by the exact decomposition "
             "`Δ_full_stack = Δ_volume_present − Δ_penalty`.\n")
    q = pairs["tie_ratio"].quantile([.1, .25, .5, .75, .9])
    L.append("| quantile | tie_ratio |")
    L.append("|---|---|")
    for k, v in q.items():
        L.append(f"| {k:.2f} | {v:.3f} |")
    if n_flip:
        f = pairs[pairs["flip_scope_only"]]
        L.append("")
        L.append(f"- among reversed comparisons, |Δ volume-present| ranges "
                 f"{f['abs_delta_volume_present'].min():.6f} to "
                 f"{f['abs_delta_volume_present'].max():.6f}")
        L.append(f"- among reversed comparisons, |Δ penalty| ranges "
                 f"{f['abs_delta_penalty'].min():.6f} to "
                 f"{f['abs_delta_penalty'].max():.6f}")
    L.append("")
    L.append("## Standing constraints\n")
    L.append("- Post-hoc; uncorrected for multiplicity; never evidence for the "
             "efficacy of propagation.")
    L.append("- No aggregate crosses the partition-regime boundary.")
    L.append("- Arm C is not ground-truth-scope-matched to A/B/I (Methods 2.3); "
             "any pair containing C inherits that qualification.")
    L.append("- The model pool varies in supervision protocol and budget but NOT "
             "in architecture. Reversals are demonstrated instances within this "
             "pool.")

    (OUT_DIR / "STEPE07_READOUT.md").write_text("\n".join(L), encoding="utf-8")
    (OUT_DIR / "stepE07_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
