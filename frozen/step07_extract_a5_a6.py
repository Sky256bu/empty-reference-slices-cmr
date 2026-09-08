from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(
    "/root/autodl-tmp/acdc_step06_freeze_20260725/"
    "stack_dice_acdc_clean.csv"
)
OUT = Path("/root/autodl-tmp/acdc_step07_extract_20260725")
OUT.mkdir(parents=True, exist_ok=True)

PHASE_ORDER = ["ed", "es", "overall"]
ARM_ORDER = ["A", "B", "C", "I"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def opposite_sign(a: float, b: float, eps: float = 1e-12) -> bool:
    return bool((a > eps and b < -eps) or (a < -eps and b > eps))


if not SRC.exists():
    raise FileNotFoundError(SRC)

d = pd.read_csv(SRC)

required = {
    "key",
    "phase",
    "inter",
    "gt",
    "pred",
    "present_dice",
    "empty_fp",
    "stack_dice",
    "arm",
    "N",
    "seed",
    "alias",
}
missing = required - set(d.columns)
if missing:
    raise RuntimeError(f"缺少字段: {sorted(missing)}")

if d["alias"].astype(bool).any():
    raise RuntimeError("冻结输入中仍含 alias=True 行，停止。")

if set(d["phase"].unique()) != {"ed", "es"}:
    raise RuntimeError(
        f"phase 异常: {sorted(d['phase'].astype(str).unique())}"
    )

# ------------------------------------------------------------------
# 关键定义：
#
# present_dice:
#   现有正式指标，属于“GT-present 切片上的逐切片平均”。
#
# stack_dice:
#   整个 stack 的体积聚合 Dice，包含 GT-empty 层上的假阳性。
#
# volume_present_dice:
#   使用与 stack_dice 完全相同的体积聚合方式，
#   但从预测体积中减去 GT-empty 层假阳性。
#
# 因此：
#   empty_fp_penalty = volume_present_dice - stack_dice
#
# 它只反映是否计入空层假阳，不混入“逐切片平均 vs 体积聚合”的差异。
# ------------------------------------------------------------------

present_pred = d["pred"].astype(float) - d["empty_fp"].astype(float)
den_present_volume = d["gt"].astype(float) + present_pred

if (present_pred < -1e-6).any():
    bad = d.loc[present_pred < -1e-6].head()
    raise RuntimeError(
        "发现 pred-empty_fp<0：\n" + bad.to_string(index=False)
    )

d["volume_present_dice"] = np.where(
    den_present_volume > 0,
    2.0 * d["inter"].astype(float) / den_present_volume,
    np.nan,
)

d["empty_fp_penalty"] = (
    d["volume_present_dice"] - d["stack_dice"]
)

if d["volume_present_dice"].max() > 1.00001:
    raise RuntimeError(
        "volume_present_dice>1，请检查 inter/gt/pred 定义。"
    )

if d["empty_fp_penalty"].min() < -1e-8:
    bad = d.nsmallest(10, "empty_fp_penalty")
    raise RuntimeError(
        "empty_fp_penalty 出现明显负值：\n"
        + bad.to_string(index=False)
    )

# 浮点误差归零。
d.loc[d["empty_fp_penalty"].abs() < 1e-12, "empty_fp_penalty"] = 0.0

# 每个正式 run、phase 应为 50 位患者。
counts = (
    d.groupby(["arm", "N", "seed", "phase"], observed=True)
    .size()
    .rename("n")
    .reset_index()
)

if not (counts["n"] == 50).all():
    raise RuntimeError(
        "并非所有 run-phase 都有50例：\n"
        + counts[counts["n"] != 50].to_string(index=False)
    )

# 患者等权，先在每个 run-phase 内汇总。
run = (
    d.groupby(
        ["arm", "N", "seed", "phase"],
        observed=True,
        as_index=False,
    )
    .agg(
        n_patients=("key", "nunique"),
        slice_present_mean=("present_dice", "mean"),
        volume_present_mean=("volume_present_dice", "mean"),
        stack_mean=("stack_dice", "mean"),
        empty_fp_penalty_mean=("empty_fp_penalty", "mean"),
        empty_fp_penalty_median=("empty_fp_penalty", "median"),
        empty_fp_mean_px=("empty_fp", "mean"),
        empty_fp_median_px=("empty_fp", "median"),
    )
)

# overall 对 ED/ES 等权，仍以 run 为单位。
metric_cols = [
    "slice_present_mean",
    "volume_present_mean",
    "stack_mean",
    "empty_fp_penalty_mean",
    "empty_fp_penalty_median",
    "empty_fp_mean_px",
    "empty_fp_median_px",
]

overall = (
    run.groupby(["arm", "N", "seed"], observed=True)[metric_cols]
    .mean()
    .reset_index()
)
overall["phase"] = "overall"
overall["n_patients"] = 100

run_all = pd.concat(
    [run, overall[run.columns]],
    ignore_index=True,
)

run_all["phase"] = pd.Categorical(
    run_all["phase"],
    categories=PHASE_ORDER,
    ordered=True,
)
run_all["arm"] = pd.Categorical(
    run_all["arm"],
    categories=ARM_ORDER,
    ordered=True,
)
run_all = run_all.sort_values(
    ["phase", "N", "seed", "arm"]
).reset_index(drop=True)

run_all.to_csv(
    OUT / "step07_run_level_scope_metrics.csv",
    index=False,
)

# ------------------------------------------------------------------
# A/B/C 是所有 N、seed 均共同存在的可配对臂。
# I 只存在于低预算，不能混入跨预算总体量。
# ------------------------------------------------------------------

abc = run[
    run["arm"].isin(["A", "B", "C"])
].copy()

complete = (
    abc.groupby(["N", "seed", "phase"], observed=True)["arm"]
    .nunique()
)

if len(complete) != 36 or not (complete == 3).all():
    raise RuntimeError(
        "A/B/C 共同 cell 不完整：\n"
        + complete.to_string()
    )

# A5/A6：逐 arm、N、phase，保留全部三个 seed。
records = []

for (arm, n_value, phase), g in run.groupby(
    ["arm", "N", "phase"],
    observed=True,
):
    g = g.sort_values("seed")

    seed_penalty = " ".join(
        f"s{int(r.seed)}={r.empty_fp_penalty_mean:+.6f}"
        for r in g.itertuples()
    )

    seed_slice = " ".join(
        f"s{int(r.seed)}={r.slice_present_mean:.6f}"
        for r in g.itertuples()
    )

    seed_stack = " ".join(
        f"s{int(r.seed)}={r.stack_mean:.6f}"
        for r in g.itertuples()
    )

    records.append(
        {
            "arm": arm,
            "N": int(n_value),
            "phase": phase,
            "n_seeds": len(g),
            "empty_fp_penalty_mean": g[
                "empty_fp_penalty_mean"
            ].mean(),
            "empty_fp_penalty_median_across_runs": g[
                "empty_fp_penalty_mean"
            ].median(),
            "slice_present_mean": g["slice_present_mean"].mean(),
            "volume_present_mean": g["volume_present_mean"].mean(),
            "stack_mean": g["stack_mean"].mean(),
            "seed_empty_fp_penalties": seed_penalty,
            "seed_slice_present": seed_slice,
            "seed_stack": seed_stack,
        }
    )

by_arm_n_phase = pd.DataFrame(records)
by_arm_n_phase.to_csv(
    OUT / "step07_a5_penalty_by_arm_N_phase.csv",
    index=False,
)

# A6 主汇总：只对 A/B/C 共同 cell 等权。
phase_summary = (
    abc.groupby("phase", observed=True)
    .agg(
        n_run_cells=("empty_fp_penalty_mean", "size"),
        penalty_mean=("empty_fp_penalty_mean", "mean"),
        penalty_median=("empty_fp_penalty_mean", "median"),
        penalty_q25=(
            "empty_fp_penalty_mean",
            lambda x: x.quantile(0.25),
        ),
        penalty_q75=(
            "empty_fp_penalty_mean",
            lambda x: x.quantile(0.75),
        ),
        slice_present_mean=("slice_present_mean", "mean"),
        volume_present_mean=("volume_present_mean", "mean"),
        stack_mean=("stack_mean", "mean"),
    )
    .reset_index()
)

phase_summary.to_csv(
    OUT / "step07_a6_phase_summary_ABC_matched.csv",
    index=False,
)

phase_map = phase_summary.set_index("phase")["penalty_mean"].to_dict()
ed_penalty = float(phase_map["ed"])
es_penalty = float(phase_map["es"])
es_ed_ratio = (
    es_penalty / ed_penalty
    if ed_penalty > 1e-12
    else np.inf
)

# 每个 arm×N×seed 的 ES−ED 配对差。
paired_phase = abc.pivot(
    index=["arm", "N", "seed"],
    columns="phase",
    values="empty_fp_penalty_mean",
).reset_index()

paired_phase["es_minus_ed"] = (
    paired_phase["es"] - paired_phase["ed"]
)
paired_phase["es_div_ed"] = np.where(
    paired_phase["ed"] > 1e-12,
    paired_phase["es"] / paired_phase["ed"],
    np.nan,
)
paired_phase.to_csv(
    OUT / "step07_a6_paired_ES_ED_ABC.csv",
    index=False,
)

# ------------------------------------------------------------------
# A5：空层损失与策略间差异是否同量级。
# 每个 N×seed×phase 内比较 A/B/C，避免预算构成混杂。
# ------------------------------------------------------------------

scale_records = []

for (n_value, seed, phase), g in abc.groupby(
    ["N", "seed", "phase"],
    observed=True,
):
    q = g.set_index("arm")

    scale_records.append(
        {
            "N": int(n_value),
            "seed": int(seed),
            "phase": phase,
            "mean_empty_fp_penalty_ABC": q[
                "empty_fp_penalty_mean"
            ].mean(),
            "strategy_range_slice_present_ABC": (
                q["slice_present_mean"].max()
                - q["slice_present_mean"].min()
            ),
            "strategy_range_volume_present_ABC": (
                q["volume_present_mean"].max()
                - q["volume_present_mean"].min()
            ),
            "strategy_range_stack_ABC": (
                q["stack_mean"].max()
                - q["stack_mean"].min()
            ),
            "abs_B_minus_C_slice_present": abs(
                q.loc["B", "slice_present_mean"]
                - q.loc["C", "slice_present_mean"]
            ),
            "abs_B_minus_C_volume_present": abs(
                q.loc["B", "volume_present_mean"]
                - q.loc["C", "volume_present_mean"]
            ),
            "abs_B_minus_C_stack": abs(
                q.loc["B", "stack_mean"]
                - q.loc["C", "stack_mean"]
            ),
        }
    )

scale_cells = pd.DataFrame(scale_records)
scale_cells.to_csv(
    OUT / "step07_a5_scale_comparison_by_cell.csv",
    index=False,
)

scale_summary = (
    scale_cells.groupby("phase", observed=True)
    .agg(
        n_cells=("N", "size"),
        mean_empty_fp_penalty=(
            "mean_empty_fp_penalty_ABC",
            "mean",
        ),
        median_empty_fp_penalty=(
            "mean_empty_fp_penalty_ABC",
            "median",
        ),
        mean_strategy_range_stack=(
            "strategy_range_stack_ABC",
            "mean",
        ),
        mean_abs_BC_stack=(
            "abs_B_minus_C_stack",
            "mean",
        ),
        mean_abs_BC_volume_present=(
            "abs_B_minus_C_volume_present",
            "mean",
        ),
        mean_abs_BC_slice_present=(
            "abs_B_minus_C_slice_present",
            "mean",
        ),
    )
    .reset_index()
)

scale_summary["penalty_over_strategy_range_stack"] = np.where(
    scale_summary["mean_strategy_range_stack"] > 0,
    scale_summary["mean_empty_fp_penalty"]
    / scale_summary["mean_strategy_range_stack"],
    np.nan,
)

scale_summary["penalty_over_abs_BC_stack"] = np.where(
    scale_summary["mean_abs_BC_stack"] > 0,
    scale_summary["mean_empty_fp_penalty"]
    / scale_summary["mean_abs_BC_stack"],
    np.nan,
)

scale_summary.to_csv(
    OUT / "step07_a5_scale_summary_ABC_matched.csv",
    index=False,
)

# ------------------------------------------------------------------
# 排名翻转：
# 1) slice-present vs stack：现有 step06 观察；
# 2) volume-present vs stack：同聚合方式，能否归因于空层假阳。
# ------------------------------------------------------------------

bc = run_all[
    run_all["arm"].isin(["B", "C"])
].copy()

rank_records = []

for (n_value, seed, phase), g in bc.groupby(
    ["N", "seed", "phase"],
    observed=True,
):
    q = g.set_index("arm")

    if not {"B", "C"}.issubset(q.index):
        continue

    ds = float(
        q.loc["B", "slice_present_mean"]
        - q.loc["C", "slice_present_mean"]
    )
    dv = float(
        q.loc["B", "volume_present_mean"]
        - q.loc["C", "volume_present_mean"]
    )
    dt = float(
        q.loc["B", "stack_mean"]
        - q.loc["C", "stack_mean"]
    )

    rank_records.append(
        {
            "N": int(n_value),
            "seed": int(seed),
            "phase": str(phase),
            "delta_BC_slice_present": ds,
            "delta_BC_volume_present": dv,
            "delta_BC_stack": dt,
            "flip_slice_vs_stack": opposite_sign(ds, dt),
            "flip_volume_vs_stack": opposite_sign(dv, dt),
        }
    )

rank_seed = pd.DataFrame(rank_records)
rank_seed.to_csv(
    OUT / "step07_BC_ranking_by_seed.csv",
    index=False,
)

rank_summary = (
    rank_seed.groupby(["N", "phase"], observed=True)
    .agg(
        n_seeds=("seed", "size"),
        mean_delta_BC_slice_present=(
            "delta_BC_slice_present",
            "mean",
        ),
        mean_delta_BC_volume_present=(
            "delta_BC_volume_present",
            "mean",
        ),
        mean_delta_BC_stack=(
            "delta_BC_stack",
            "mean",
        ),
        seed_flips_slice_vs_stack=(
            "flip_slice_vs_stack",
            "sum",
        ),
        seed_flips_volume_vs_stack=(
            "flip_volume_vs_stack",
            "sum",
        ),
    )
    .reset_index()
)

rank_summary["mean_flip_slice_vs_stack"] = [
    opposite_sign(a, b)
    for a, b in zip(
        rank_summary["mean_delta_BC_slice_present"],
        rank_summary["mean_delta_BC_stack"],
    )
]

rank_summary["mean_flip_volume_vs_stack"] = [
    opposite_sign(a, b)
    for a, b in zip(
        rank_summary["mean_delta_BC_volume_present"],
        rank_summary["mean_delta_BC_stack"],
    )
]

rank_summary.to_csv(
    OUT / "step07_BC_ranking_summary.csv",
    index=False,
)

# ------------------------------------------------------------------
# 人类可读报告。
# ------------------------------------------------------------------

report_path = OUT / "step07_A5_A6_report.txt"

with report_path.open("w", encoding="utf-8") as f:
    def p(text: str = "") -> None:
        print(text, file=f)

    p("=" * 92)
    p("STEP07 — A5/A6 只读提取与同聚合方式控制")
    p("=" * 92)
    p(f"source: {SRC}")
    p(f"source_sha256: {sha256(SRC)}")
    p(f"rows: {len(d)}")
    p(f"formal runs: {run[['arm','N','seed']].drop_duplicates().shape[0]}")
    p()

    p("关键定义")
    p("- slice_present_mean：正式 present-slice 逐切片平均 Dice")
    p("- volume_present_mean：体积聚合 Dice，但排除 GT-empty 层假阳")
    p("- stack_mean：体积聚合 Dice，包含 GT-empty 层假阳")
    p("- empty_fp_penalty = volume_present_mean - stack_mean")
    p("  这是空层假阳的纯损失，不混入逐切片平均与体积聚合差异。")
    p()

    p("=" * 92)
    p("A6：A/B/C 共同 cell 等权的 ED/ES 空层损失")
    p("=" * 92)
    p(phase_summary.to_string(index=False))
    p()
    p(f"ED mean penalty = {ed_penalty:.8f}")
    p(f"ES mean penalty = {es_penalty:.8f}")
    p(f"ES/ED ratio     = {es_ed_ratio:.4f}")
    p()

    p("ES−ED 配对差概览（arm×N×seed 为单位）")
    p(
        paired_phase["es_minus_ed"]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .to_string()
    )
    p()

    p("=" * 92)
    p("A5：空层损失与同 cell 策略差异的量级比较")
    p("=" * 92)
    p(scale_summary.to_string(index=False))
    p()

    p("=" * 92)
    p("B vs C：逐 N、逐 phase 的三种评价口径")
    p("=" * 92)
    p(rank_summary.to_string(index=False))
    p()

    direct_flips = rank_summary[
        rank_summary["mean_flip_volume_vs_stack"]
    ]

    p("=" * 92)
    p("因果措辞判定")
    p("=" * 92)

    if len(direct_flips) > 0:
        p(
            "存在 volume-present → full-stack 的均值排序翻转。"
        )
        p(
            "这两种指标采用相同体积聚合方式，差异仅为空层假阳；"
        )
        p(
            "因此相应 cell 可将翻转直接归因于空层假阳是否进入评价。"
        )
        p()
        p(direct_flips.to_string(index=False))
    else:
        p(
            "未观察到 volume-present → full-stack 的均值排序翻转。"
        )
        p(
            "因此 slice-present → stack 的翻转只能表述为"
            "“评价定义敏感性”，"
        )
        p(
            "不能把排序翻转完全归因于空层假阳，"
            "因为其中还包含聚合方式变化。"
        )

    p()
    p("所有结果均为 post-hoc mechanistic/descriptive；")
    p("未触碰冻结的 M&Ms-2 确认性 EF-MAE 主终点。")

print(report_path)

# 冻结本次提取产物。
with (OUT / "SHA256SUMS.txt").open("w", encoding="utf-8") as f:
    for path in sorted(OUT.glob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            print(f"{sha256(path)}  {path}", file=f)
