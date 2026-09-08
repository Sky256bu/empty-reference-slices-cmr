#!/usr/bin/env python3
"""
STACK-DICE quick check  --  zero GPU, read-only, modifies no frozen product

It answers one question: once false positives on empty slices (gt_px=0) enter the denominator,
does the ranking discordance between the overlap metric and EF disappear?

Boundaries of the conclusion (not to be exceeded in writing):
  May write: "Switching from present-slice Dice to whole-stack volumetric Dice
        did not eliminate the observed endpoint-ranking discordance."
  May not write: "Any overlap-based metric is intrinsically misaligned with EF."

Key implementation points:
  1. The intersection is back-derived as inter = dice*(gt_px+pred_px)/2 -- the script first
     proves this exact: inter must be an integer, so the deviation of every back-derived value
     from the nearest integer is checked. A large deviation means a smoothing term is present,
     in which case back-derivation is unusable and one must return to the raw masks.
  2. Rows with gt_px=0 contribute inter=0 regardless of whether dice is recorded as 0/1/NaN on
     that row, which makes the check naturally immune to the empty-slice convention problem.
  3. The smallest unit of analysis is run x patient x phase; ED/ES are reported separately and
     then aggregated pairwise over common cells.
  4. Comparisons are made only on cells where every compared arm exists simultaneously (fixing
     the composition confounding of open item #1).

Usage:
  python3 stack_dice_check.py --dataset acdc
  python3 stack_dice_check.py --dataset mnms2
"""
import argparse, glob, os, re, sys
import numpy as np
import pandas as pd

ACDC = "/root/autodl-tmp/acdc/seg_lv"
MNMS = "/root/autodl-tmp/mnms/seg_lv"


# ------------------------------------------------------------------ back-derivation self-proof
def derive_inter(gt, pred, dice, label=""):
    """inter = dice*(gt+pred)/2, with a self-proof that it is an integer. Returns (inter, diagnostic dict)."""
    denom = gt + pred
    inter = np.where(denom > 0, dice * denom / 2.0, 0.0)
    # rows with gt=0 are forced to inter=0 (regardless of how dice is notated on that row)
    inter = np.where(gt == 0, 0.0, inter)
    live = (gt > 0) & (denom > 0)
    resid = np.abs(inter[live] - np.round(inter[live]))
    diag = dict(n=int(live.sum()),
                resid_max=float(resid.max()) if live.sum() else np.nan,
                resid_p999=float(np.percentile(resid, 99.9)) if live.sum() else np.nan,
                frac_gt_1e_3=float((resid > 1e-3).mean()) if live.sum() else np.nan)
    print(f"  [back-derivation self-proof {label}] n={diag['n']}  max|resid|={diag['resid_max']:.3e}  "
          f"p99.9={diag['resid_p999']:.3e}  frac>1e-3={diag['frac_gt_1e_3']:.4f}")
    if diag["resid_max"] > 1e-2:
        print("  !! residual too large: Dice may carry a smoothing term, or the CSV may be rounded."
              "back-derivation unusable; recompute from the raw masks instead.")
    # number of rows violating inter<=min(gt,pred) (a second, independent self-proof)
    bad = int((inter[live] > np.minimum(gt, pred)[live] + 0.5).sum())
    print(f"  [back-derivation self-proof {label}] rows with inter>min(gt,pred) = {bad}")
    return inter, diag


def stack_dice(df, gtcol, pcol, dcol, keys, label=""):
    """Compute the volumetric Dice grouped by keys; also compute the present-slice mean Dice for comparison."""
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
    """Keep only the cells in which all arms exist simultaneously; return the paired difference per cell."""
    w = df.pivot_table(index=cellcols, columns=armcol, values=valcol)
    w = w.dropna(subset=list(arms))
    out = w.copy()
    return out


def report_pairs(w, a1, a2, name):
    d = (w[a1] - w[a2]).dropna()
    if len(d) == 0:
        print(f"  {name}: no common cell"); return
    boot = np.random.default_rng(0).choice(d.values, (5000, len(d)), replace=True).mean(1)
    print(f"  {name}: n_cell={len(d)}  mean Δ={d.mean():+.4f}  "
          f"median Δ={np.median(d):+.4f}  "
          f"95%CI=[{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}]  "
          f"cells favouring {a1}={int((d>0).sum())}/{len(d)}")


# ------------------------------------------------------------------- ACDC
def run_acdc():
    files = sorted(glob.glob(os.path.join(ACDC, "eval*", "**", "test_per_slice.csv"),
                             recursive=True))
    print(f"found {len(files)} test_per_slice.csv files")
    rows = []
    for f in files:
        df = pd.read_csv(f)
        arms = sorted({m.group(1) for c in df.columns
                       if (m := re.match(r"predpx_(\w+)$", c))})
        # parse the cell from the path (N, seed, split) -- adjust this to your own directory naming
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
        print("no data"); return
    A = pd.concat(rows, ignore_index=True)
    A.to_csv("stack_dice_acdc_patient_phase.csv", index=False)

    print("\n--- per phase, paired over common cells (B vs C) ---")
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
    print("\noutput: stack_dice_acdc_patient_phase.csv")


# ------------------------------------------------------------------ M&Ms-2
def run_mnms2():
    files = sorted(glob.glob(os.path.join(
        MNMS, "scope_control_v3_eval_rep*", "mnms2_scope_control_v3_per_slice_locked.csv")))
    print(f"found {len(files)} per_slice_locked.csv files")
    rows = []
    for f in files:
        df = pd.read_csv(f)
        need = {"unit_id", "seed", "arm", "domain", "key", "phase", "gt_px",
                "dice", "pred_px"}
        if not need <= set(df.columns):
            print(f"  {f} is missing columns: {need - set(df.columns)}"); continue
        a = stack_dice(df, "gt_px", "pred_px", "dice",
                       ["unit_id", "seed", "arm", "domain", "key", "phase"],
                       label=os.path.basename(os.path.dirname(f)))
        a["rep"] = os.path.basename(os.path.dirname(f))
        rows.append(a)
    if not rows:
        print("no data"); return
    M = pd.concat(rows, ignore_index=True)
    M.to_csv("stack_dice_mnms2_patient_phase.csv", index=False)

    print("\n--- per phase, paired at unit level (B vs Cneg) ---")
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
    print("\noutput: stack_dice_mnms2_patient_phase.csv")
    print("\nNote: this script produces overlap metrics only and does not touch the frozen EF-MAE primary endpoint."
          "Comparisons against the EF ranking must be labelled confirmatory or exploratory separately in the ledger.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["acdc", "mnms2"], required=True)
    args = ap.parse_args()
    print("=" * 78)
    print("STACK-DICE quick check -- read-only, does not write to the frozen directory")
    print("=" * 78)
    (run_acdc if args.dataset == "acdc" else run_mnms2)()
