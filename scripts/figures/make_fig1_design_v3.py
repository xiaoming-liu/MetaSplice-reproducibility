"""Figure 1 (draft): MetaSplice study design, with the pooled temporal test set. Counts are those of Table 1 (committed
results logs); no new numbers.

Writes paper/AJHG/Fig1_design_v3.pdf and .png (v3: origin split removed from the test-set box).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7.5})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures")
INK, MUTED, EDGE = "#1f2933", "#52606d", "#9aa5b1"
FILL = {"data": "#eef2f7", "model": "#e6f0ec", "eval": "#f4efe6", "acmg": "#f6e7e3"}

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(-1.5, 64); ax.axis("off")


def box(x, y, w, h, title, lines, kind, title_size=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2", fc=FILL[kind], ec=EDGE, lw=0.7))
    ax.text(x + 1.2, y + h - 1.4, title, ha="left", va="top", fontsize=title_size, fontweight="bold", color=INK)
    ax.text(x + 1.2, y + h - 4.6, "\n".join(lines), ha="left", va="top", fontsize=6.6, color=MUTED, linespacing=1.35)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=MUTED,
                                 shrinkA=1, shrinkB=1))


# column headers
for x, t in [(1, "Data"), (36, "Model"), (69, "Evaluation")]:
    ax.text(x, 62.5, t, fontsize=9, fontweight="bold", color=INK, va="top")

# data column
box(1, 49, 30, 10.5, "Training",
    ["ClinVar 2025-08-31: 413,199 intronic", "SNVs (41,032 P/LP) in 6,961 genes"], "data")
box(1, 31, 30, 16, "Temporal test set",
    ["ClinVar 2026-09-28: 46,112 variants", "(6,056 P/LP), not in training and not", "P/LP or B/LB in the training release",
     "Non-canonical: 40,619 (602 P/LP)"], "data")
box(1, 1, 30, 21, "External and population data",
    ["SpliceVarDB: 22,274 assayed variants", "Literature: 697 splice-altering variants",
     "ClinVar VUS: 60,776 remaining in the", "    calibration cohort", "gnomAD v4.1: 100,000 reference"], "data")

# model column
box(36, 36.5, 28, 23, "MetaSplice",
    ["Intronic SNVs in GENCODE v50", "MANE Select and Plus Clinical", "transcripts (GRCh38)", "",
     "38 features, no allele frequency:", "SpliceTransformer (16), Pangolin (3),", "conservation (4), GPN-MSA, DANN,",
     "gene-level (3), exon context (3),", "region (7)"], "model")
box(36, 18, 28, 15.5, "Gradient-boosted trees",
    ["XGBoost; one feature pipeline", "for training and scoring", "Five gene-grouped folds", "(out-of-fold predictions)"], "model")

# evaluation column
box(69, 46, 30, 13.5, "Discrimination",
    ["auPRC against SpliceAI and CADD on", "non-canonical variants of the temporal", "test set, by region (variant- and",
     "gene-clustered CIs)"], "eval")
box(69, 31.5, 30, 12, "What drives the score",
    ["Standalone inputs and baselines;", "ablations; gene-disjoint retraining;", "within-gene concordance; SHAP"], "eval")
box(69, 18, 30, 11, "Splicing endpoints",
    ["SpliceVarDB (measured splicing);", "literature-confirmed variants", "(positives only)"], "eval")
box(69, 0, 30, 16, "ACMG/AMP evidence",
    ["Likelihood-ratio calibration on", "training (out of fold)", "PP3/BP4 at Supporting:",
     "score \u2265 0.544 / \u2264 0.0017", "vs ClinGen's SpliceAI rule on the", "test set; resolved and remaining VUS"], "acmg")

# training -> model
arrow(31.6, 54, 35.4, 54)
arrow(50, 36.3, 50, 34.0)
# evaluation data -> model (one bracket), labelled
for y in (39.0, 11.5):
    ax.plot([31.6, 33.4], [y, y], color=MUTED, lw=0.7)
ax.plot([33.4, 33.4], [11.5, 39.0], color=MUTED, lw=0.7)
arrow(33.4, 25.75, 35.4, 25.75)
ax.text(33.0, 8.0, "scored by the\nfrozen model", fontsize=6.2, color=MUTED, ha="left", va="top", style="italic")
# model -> evaluations (one bracket)
arrow(64.6, 25.75, 66.6, 25.75)
ax.plot([66.6, 66.6], [8.0, 52.75], color=MUTED, lw=0.7)
for y in (52.75, 37.5, 23.5, 8.0):
    arrow(66.6, y, 68.4, y)

fig.savefig(f"{OUT}/Fig1_design_v3.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/Fig1_design_v3.png", dpi=300, bbox_inches="tight")
print("wrote Fig1_design_v3.pdf/.png")
