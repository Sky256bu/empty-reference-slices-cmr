"""
Fig. 6 / Table S5 / Table S6 -- pure rendering of the frozen cells (Q-4 ruling)

Constraints (not to be relaxed):
  * The only input is the frozen tail-risk table mnms2_scope_control_v3_R_aggregate_tail_risk.csv
  * No mean / extremum / range / standard deviation is computed across training units, repeats or arms
  * Fig. 6 draws no mean line, no box plot and no error bar
  * repeat_averaged_ensemble_prediction does not enter Fig. 6; it is listed separately in Table S5c and labelled as a sensitivity analysis
  * The only derived quantity is pct = n / prediction_instances x 100, which has been verified against the frozen column
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os
_DATA = _os.environ.get("FROZEN_DATA_DIR", _os.path.dirname(_os.path.abspath(__file__)))
SRC = _os.path.join(_DATA, "mnms2_scope_control_v3_R_aggregate_tail_risk.csv")
OUT = _os.environ.get("FIG_OUT_DIR", ".")

df = pd.read_csv(SRC)

# ---- integrity self-check: the frozen pct columns agree with n/prediction_instances ----
for flag in ["ae_gt20", "ae_gt40", "invalid_ef"]:
    assert np.allclose(df[f"{flag}_n"] / df["prediction_instances"] * 100,
                       df[f"{flag}_pct"]), f"pct mismatch: {flag}"

REPEATS = ["locked_repeat_00", "locked_repeat_01", "locked_repeat_02"]
ENSEMBLE = "repeat_averaged_ensemble_prediction"
UNITS = [f"u{i:02d}" for i in range(10)]

ext = df[df.domain == "External_pooled"].copy()
assert set(ext.prediction_instances) == {141}, "External_pooled n must be 141"


def cell(table, unit, arm, col):
    """Look the cell up directly; perform no aggregation."""
    r = ext[(ext.table == table) & (ext.unit_id == unit) & (ext.arm == arm)]
    assert len(r) == 1
    return r.iloc[0][col]


# =====================================================================
# Fig. 6
# =====================================================================
ARMS = [("B", "#1f4e79", "o", "Propagated (B)"),
        ("Cneg", "#b03030", "s", "Anchor-only control (Cneg)")]
# the three repeats are separated by a fixed small offset so they read as three discrete runs rather than as error bars
REP_OFFSET = {REPEATS[0]: -0.075, REPEATS[1]: 0.0, REPEATS[2]: 0.075}
ARM_OFFSET = {"B": -0.19, "Cneg": 0.19}

fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True)

for ax, col, ylab in [
        (axes[0], "ae_gt20_pct", "Cases with |ΔEF| > 20 EF points (%)"),
        (axes[1], "ae_gt40_pct", "Cases with |ΔEF| > 40 EF points (%)")]:
    for xi, unit in enumerate(UNITS):
        for arm, colour, marker, _ in ARMS:
            for rep in REPEATS:
                ax.plot(xi + ARM_OFFSET[arm] + REP_OFFSET[rep],
                        cell(rep, unit, arm, col),
                        marker=marker, ms=5.2, mfc=colour, mec=colour,
                        alpha=0.9, linestyle="none", zorder=3)
    for xi in range(1, len(UNITS)):
        ax.axvline(xi - 0.5, color="0.90", lw=0.7, zorder=0)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_ylim(bottom=-2)
    ax.margins(y=0.14)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

axes[1].set_xticks(range(len(UNITS)))
axes[1].set_xticklabels(UNITS, fontsize=8)
axes[1].set_xlabel("Formal training unit", fontsize=9)
axes[0].set_xlim(-0.6, len(UNITS) - 0.4)

handles = [plt.Line2D([], [], marker=m, color=c, mfc=c, mec=c, ls="none",
                      ms=5.5, label=lab) for _, c, m, lab in ARMS]
fig.legend(handles=handles, fontsize=8, frameon=False, ncol=2,
           loc="upper center", bbox_to_anchor=(0.5, 0.945))

fig.suptitle("Pre-specified tail-risk sensitivity analysis "
             "(held-out vendor domains pooled)", fontsize=10, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.925])
fig.savefig(f"{OUT}/Fig6_tail_risk_per_unit.png", dpi=400)
fig.savefig(f"{OUT}/Fig6_tail_risk_per_unit.pdf")
plt.close(fig)


# =====================================================================
# Table S5 -- External_pooled frozen cells, transcribed cell by cell
# =====================================================================
def block(col_n, col_pct):
    head = ("| Unit | B rep00 | B rep01 | B rep02 | Cneg rep00 | Cneg rep01 "
            "| Cneg rep02 |\n|---|---:|---:|---:|---:|---:|---:|\n")
    rows = []
    for unit in UNITS:
        vals = []
        for arm in ["B", "Cneg"]:
            for rep in REPEATS:
                n = int(cell(rep, unit, arm, col_n))
                p = cell(rep, unit, arm, col_pct)
                vals.append(f"{n} ({p:.2f})")
        rows.append(f"| {unit} | " + " | ".join(vals) + " |")
    return head + "\n".join(rows)


def ensemble_block():
    head = ("| Unit | Arm | AE>20 n (%) | AE>40 n (%) | invalid EF n (%) |\n"
            "|---|---|---:|---:|---:|\n")
    rows = []
    for unit in UNITS:
        for arm in ["B", "Cneg"]:
            a = int(cell(ENSEMBLE, unit, arm, "ae_gt20_n"))
            ap = cell(ENSEMBLE, unit, arm, "ae_gt20_pct")
            b = int(cell(ENSEMBLE, unit, arm, "ae_gt40_n"))
            bp = cell(ENSEMBLE, unit, arm, "ae_gt40_pct")
            c = int(cell(ENSEMBLE, unit, arm, "invalid_ef_n"))
            cp = cell(ENSEMBLE, unit, arm, "invalid_ef_pct")
            rows.append(f"| {unit} | {arm} | {a} ({ap:.2f}) | {b} ({bp:.2f}) "
                        f"| {c} ({cp:.2f}) |")
    return head + "\n".join(rows)


s5 = f"""# Table S5 — Pre-specified tail-risk sensitivity analysis
### `External_pooled` (Philips + GE), frozen cells, n = 141 cases per cell

> **Presentation specification (Q-4 ruling)**: this table transcribes the frozen tail-risk
> table cell by cell, **with no aggregation applied**. The frozen table is stratified only by
> `unit × arm × domain × repeat` and contains no aggregate rows; this table therefore
> **does not provide** means, extrema, ranges or standard deviations across training units, and
> the main text may not quote such figures either.
>
> Tail risk is a **pre-specified sensitivity endpoint** (the thresholds and `tail_risk_table`
> were frozen together with 84/85 on 2026-07-21, before the locked evaluation). **EF-MAE is the
> sole confirmatory primary endpoint**; tail risk does not substitute for it under any circumstance.
>
> Data source: `mnms2_scope_control_v3_R_aggregate_tail_risk.csv`
> (repeat aggregator `87_aggregate_repeats_v3_1.py`,
> sha256 `aeab8b9b3d8a1d9d35d341a755608c1578b781571918ef6618d321d9d4f8c66f`).
> The complete per-domain frozen table is **Table S6**.

---

## S5a — Cases with absolute EF error > 20 EF points, n (%)

{block('ae_gt20_n', 'ae_gt20_pct')}

---

## S5b — Cases with absolute EF error > 40 EF points, n (%)

{block('ae_gt40_n', 'ae_gt40_pct')}

---

## S5c — Ensemble sensitivity (`repeat_averaged_ensemble_prediction`)

> **This block is an ensemble sensitivity analysis, not the primary view.** The primary view is
> the three per-repeat columns of S5a / S5b, which describe the tail risk faced by a **single
> training run**; the prediction-averaged ensemble rows are listed separately and labelled as
> such, **must not** be used as a headline figure, and **do not enter Fig. 6**.

{ensemble_block()}
"""
open(f"{OUT}/Table_S5_tail_risk_external_pooled.md", "w").write(s5)

# =====================================================================
# Table S6 -- the complete frozen table, emitted unchanged
# =====================================================================
raw = open(SRC).read()
open(f"{OUT}/Table_S6_tail_risk_full_frozen.csv", "w").write(raw)
assert open(f"{OUT}/Table_S6_tail_risk_full_frozen.csv").read() == raw

print("Fig. 6 + Table S5 + Table S6 written.")
print("rows in S6:", len(df), "| tables:", sorted(df.table.unique()))
