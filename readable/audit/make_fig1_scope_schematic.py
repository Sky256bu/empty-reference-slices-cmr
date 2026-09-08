#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig1_scope_schematic.py
============================
Replacement Fig. 1: a **purely definitional schematic** of the three metric scopes.

Identity statement (**this determines the legitimacy of the figure; do not delete**)
------------------------------------
This figure **contains no model output, reads no data file and produces no statistic**.
It illustrates the three definitions and one identity of Methods 2.6, all of them analytic.

The original Fig. 1 design (a "near-full-size blood pool on the slice adjacent to and beyond
the ES basal boundary") was retired under the Fig. 1 ruling: that design depended on the
**slice position** (P-A3) and the **shape** (P-A8) of the false positive, neither of which was
measured, and D3 ruled that this manuscript may not fill such gaps with new analyses.

The **slice position and size of the illustrated false positive are arbitrary**, and carry no
empirical assertion. This statement must appear verbatim in the figure caption.

Output: Fig1_metric_scope_schematic.pdf / .svg
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

Z = 9                      # number of illustrated stack slices
GT_LO, GT_HI = 2, 6        # GT axial support (endpoints included)
FP_SLICE = 7               # an arbitrarily chosen GT-empty slice, for illustration only

C_GT = "#4878A8"
C_FP = "#C0504D"
C_OFF = "#D9D9D9"
C_TXT = "#222222"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})

fig = plt.figure(figsize=(7.16, 3.55))          # double-column width
gs = fig.add_gridspec(1, 4, width_ratios=[1.62, 1, 1, 1], wspace=0.34)


def draw_stack(ax, title, include, show_pred=True, tag=""):
    """include: the set of slices this scope admits into the computation"""
    ax.set_xlim(-0.45, 2.15)
    ax.set_ylim(-2.15, Z + 0.4)
    ax.axis("off")
    ax.set_title(title, fontsize=8.8, pad=6, color=C_TXT)
    for z in range(Z):
        y = Z - 1 - z
        inc = z in include
        ax.add_patch(Rectangle((0, y), 2.0, 0.82,
                               facecolor="white" if inc else C_OFF,
                               edgecolor="#666666" if inc else "#BBBBBB",
                               linewidth=0.9 if inc else 0.6,
                               zorder=1, alpha=1.0 if inc else 0.65))
        if GT_LO <= z <= GT_HI:                       # GT foreground
            ax.add_patch(Rectangle((0.45, y + 0.19), 0.52, 0.45,
                                   facecolor=C_GT, edgecolor="none",
                                   zorder=2, alpha=1.0 if inc else 0.30))
        if show_pred and z == FP_SLICE:               # illustrated false positive (position arbitrary)
            ax.add_patch(Rectangle((0.45, y + 0.19), 0.44, 0.45,
                                   facecolor=C_FP, edgecolor="none",
                                   zorder=2, alpha=1.0 if inc else 0.30))
        if not inc:
            ax.plot([0.05, 1.95], [y + 0.41, y + 0.41],
                    color="#999999", lw=0.7, zorder=3)
    if tag:
        ax.text(1.0, -1.35, tag, ha="center", va="center",
                fontsize=7.6, color=C_TXT)


# ---------------------------------------------------------------- (a) stack
ax0 = fig.add_subplot(gs[0, 0])
draw_stack(ax0, "(a) Short-axis stack", include=set(range(Z)))
ax0.set_xlim(-0.55, 5.05)
ax0.annotate("", xy=(2.12, Z - 1 - GT_HI + 0.10), xytext=(2.12, Z - 1 - GT_LO + 0.72),
             arrowprops=dict(arrowstyle="-", color=C_GT, lw=1.8))
ax0.text(2.28, (Z - 1 - GT_HI + Z - 1 - GT_LO) / 2.0 + 0.41, "GT-present\nslices",
         fontsize=7.3, va="center", ha="left", color=C_GT)
ax0.annotate("", xy=(2.05, Z - 1 - FP_SLICE + 0.41), xytext=(2.26, Z - 1 - FP_SLICE + 0.41),
             arrowprops=dict(arrowstyle="-", color=C_FP, lw=0.9))
ax0.text(2.28, Z - 1 - FP_SLICE + 0.41, "predicted foreground\non a GT-empty slice",
         fontsize=7.3, va="center", ha="left", color=C_FP)
ax0.annotate("", xy=(-0.34, 0.3), xytext=(-0.34, Z - 0.3),
             arrowprops=dict(arrowstyle="-|>", color="#777777", lw=0.9))
ax0.text(-0.50, Z / 2.0, "slice axis", fontsize=7.2, rotation=90,
         va="center", ha="center", color="#777777")

# ------------------------------------------------- (b)(c)(d) the three scopes
present = set(range(GT_LO, GT_HI + 1))
allsl = set(range(Z))

ax1 = fig.add_subplot(gs[0, 1])
draw_stack(ax1, "(b) Slice-present Dice", include=present,
           tag="mean over GT-present\nslices of per-slice Dice")

ax2 = fig.add_subplot(gs[0, 2])
draw_stack(ax2, "(c) Volume-present Dice", include=present,
           tag="volumetric Dice,\naccumulated over\nGT-present slices only")

ax3 = fig.add_subplot(gs[0, 3])
draw_stack(ax3, "(d) Full-stack Dice", include=allsl,
           tag="same volumetric Dice,\naccumulated over\nall slices")

# ---------------------------------------------------------- the two difference annotations
fig.text(0.435, 0.038, "differ only in aggregation", ha="center",
         fontsize=7.6, color="#555555")
fig.text(0.760, 0.038, "differ only in whether GT-empty slices are admitted",
         ha="center", fontsize=7.6, color=C_FP)
fig.patches.append(FancyArrowPatch((0.345, 0.078), (0.525, 0.078),
                                   transform=fig.transFigure, arrowstyle="<->",
                                   mutation_scale=7, color="#888888", lw=0.8))
fig.patches.append(FancyArrowPatch((0.600, 0.078), (0.920, 0.078),
                                   transform=fig.transFigure, arrowstyle="<->",
                                   mutation_scale=7, color=C_FP, lw=0.9))

fig.subplots_adjust(left=0.035, right=0.985, top=0.905, bottom=0.135)
for ext in ("pdf", "svg"):
    fig.savefig(f"Fig1_metric_scope_schematic.{ext}", bbox_inches="tight")
print("wrote Fig1_metric_scope_schematic.pdf / .svg")
print("Reminder: the caption must contain the sentence 'the slice position and size of the illustrated false positive are arbitrary, and carry no empirical assertion'.")
