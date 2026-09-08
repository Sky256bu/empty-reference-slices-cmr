"""
Fig. 6 / Table S5 / Table S6 — 纯渲染冻结单元格（⟦Q-4 决议⟧）

约束（不得放宽）：
  * 唯一输入 = 冻结 tail-risk 表 mnms2_scope_control_v3_R_aggregate_tail_risk.csv
  * 不计算任何跨训练单位、跨重复、跨臂的均值 / 极值 / 范围 / 标准差
  * Fig. 6 不画均值线、不画箱线、不加误差棒
  * repeat_averaged_ensemble_prediction 不入 Fig. 6，单列于 Table S5c 并标 sensitivity
  * 唯一派生量：pct = n / prediction_instances × 100，且已核验与冻结列一致
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

# ---- 完整性自检：冻结的 pct 列与 n/prediction_instances 一致 ----
for flag in ["ae_gt20", "ae_gt40", "invalid_ef"]:
    assert np.allclose(df[f"{flag}_n"] / df["prediction_instances"] * 100,
                       df[f"{flag}_pct"]), f"pct mismatch: {flag}"

REPEATS = ["locked_repeat_00", "locked_repeat_01", "locked_repeat_02"]
ENSEMBLE = "repeat_averaged_ensemble_prediction"
UNITS = [f"u{i:02d}" for i in range(10)]

ext = df[df.domain == "External_pooled"].copy()
assert set(ext.prediction_instances) == {141}, "External_pooled n must be 141"


def cell(table, unit, arm, col):
    """逐格查表；不作任何聚合。"""
    r = ext[(ext.table == table) & (ext.unit_id == unit) & (ext.arm == arm)]
    assert len(r) == 1
    return r.iloc[0][col]


# =====================================================================
# Fig. 6
# =====================================================================
ARMS = [("B", "#1f4e79", "o", "Propagated (B)"),
        ("Cneg", "#b03030", "s", "Anchor-only control (Cneg)")]
# 三次重复用固定的小偏移分开，使其可读为三个离散 run，而非误差棒
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
# Table S5 — External_pooled 冻结单元格，逐格抄写
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

> **⟦Q-4 决议⟧ 呈现规格**：本表逐格抄写冻结 tail-risk 表，**未作任何聚合**。
> 冻结表仅按 `unit × arm × domain × repeat` 分层，不含聚合行；因此本表**不提供**
> 跨训练单位的均值、极值、范围或标准差，正文亦不得引用此类数字。
>
> Tail risk is a **pre-specified sensitivity endpoint**（阈值与 `tail_risk_table`
> 于 2026-07-21 随 84/85 冻结，早于 locked evaluation）。**EF-MAE 是唯一
> confirmatory 主终点**；tail risk 在任何情况下都不得替代它。
>
> 数据源：`mnms2_scope_control_v3_R_aggregate_tail_risk.csv`
> （repeat 聚合器 `87_aggregate_repeats_v3_1.py`,
> sha256 `aeab8b9b3d8a1d9d35d341a755608c1578b781571918ef6618d321d9d4f8c66f`）。
> 完整分域冻结表见 **Table S6**。

---

## S5a — Cases with absolute EF error > 20 EF points, n (%)

{block('ae_gt20_n', 'ae_gt20_pct')}

---

## S5b — Cases with absolute EF error > 40 EF points, n (%)

{block('ae_gt40_n', 'ae_gt40_pct')}

---

## S5c — Ensemble sensitivity (`repeat_averaged_ensemble_prediction`)

> **This block is an ensemble sensitivity analysis, not the primary view.**
> 主视图为 S5a / S5b 的三个 per-repeat 行，它们描述**单次训练 run** 面对的尾部
> 风险；prediction-averaged ensemble 行以此单列并标注，**不得**用作 headline 数字，
> 亦**不入 Fig. 6**。

{ensemble_block()}
"""
open(f"{OUT}/Table_S5_tail_risk_external_pooled.md", "w").write(s5)

# =====================================================================
# Table S6 — 完整冻结表，原样输出
# =====================================================================
raw = open(SRC).read()
open(f"{OUT}/Table_S6_tail_risk_full_frozen.csv", "w").write(raw)
assert open(f"{OUT}/Table_S6_tail_risk_full_frozen.csv").read() == raw

print("Fig. 6 + Table S5 + Table S6 written.")
print("rows in S6:", len(df), "| tables:", sorted(df.table.unique()))
