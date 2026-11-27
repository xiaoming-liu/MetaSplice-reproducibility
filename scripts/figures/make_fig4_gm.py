"""Figure 4 of the manuscript: ACMG/AMP evidence from MetaSplice scores at the lower bound on the prior, read from
results/S15_pooled.tsv and S15_thresholds.tsv (S15_lower_bound_prior.py). Writes Figure4.pdf and .png.
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
INK, MUTED, GRID = "#1f2933", "#616e7c", "#d9dee4"
COL = {"MetaSplice": "#0072B2", "SpliceAI": "#D55E00", "CADD": "#8C8C8C"}   # Okabe-Ito blue / vermillion, grey
REGIONS = ["non-canonical", "Donor Region", "Acceptor Region", "PPT Region", "Deep Intronic",
           "Deep Intronic, sensitivity set"]
LABEL = {"non-canonical": "Non-canonical", "Donor Region": "Donor region", "Acceptor Region": "Acceptor region",
         "PPT Region": "PPT region", "Branch Point Region": "Branch-point region", "Deep Intronic": "Deep intronic",
         "Deep Intronic, sensitivity set": "Deep intronic,\nsensitivity set",
         "Canonical Acceptor Site": "Canonical acceptor", "Canonical Donor Site": "Canonical donor"}


def tsv(name):
    return pd.read_csv(os.path.join(RES, name), sep="\t")


def panel(ax, letter):
    ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, va="bottom")


# no Matplotlib name or creation date in the files (Matplotlib writes them by default)
NO_META = {"pdf": {"Creator": None, "Producer": None, "CreationDate": None}, "png": {"Software": None}}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "..", "..", "figures", f"{name}.{ext}"), dpi=300, bbox_inches="tight", metadata=NO_META[ext])
    plt.close(fig)
    print("wrote", name)


def fig4():
    q = tsv("S15_pooled.tsv")
    thr = tsv("S15_thresholds.tsv")
    thr = thr[(thr.tool == "MetaSplice") & (thr.cohort == ">= 1 star")].set_index("level")
    need = thr.LR_needed
    lv = q[(q.section == "levels") & (q.cohort == "calibration cohort, >= 1 star") & (q.rule == "MetaSplice")]
    lv = lv[lv.code != "Indeterminate"].set_index("code")
    order = [c for c in ["PP3_VeryStrong", "PP3_Strong", "PP3_Moderate", "PP3_Supporting", "BP4_Supporting", "BP4_Moderate",
                         "BP4_Strong"] if c in lv.index]
    t = thr.score_threshold
    names = {"PP3_Moderate": f"PP3 Moderate (\u2265 {t['PP3_Moderate']:.3f})",
             "PP3_Supporting": f"PP3 Supporting ({t['PP3_Supporting']:.3f} to {t['PP3_Moderate']:.3f})",
             "BP4_Supporting": f"BP4 Supporting (\u2264 {t['BP4_Supporting']:.5f})"}
    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.75)
    ax = fig.add_subplot(gs[0])
    ys = np.arange(len(order))[::-1]
    for y, code in zip(ys, order):
        r = lv.loc[code]
        c = COL["MetaSplice"] if code.startswith("PP3") else "#009E73"
        ax.errorbar(r.LR, y, xerr=[[r.LR - r.lo95_exact], [r.hi95_exact - r.LR]], fmt="o", ms=3.5, color=c,
                    elinewidth=0.9, capsize=0)
        ax.plot([need[code]] * 2, [y - 0.32, y + 0.32], color=INK, lw=1.2)
    ax.axvline(1, color=MUTED, lw=0.5, ls=":")
    ax.set_xscale("log"); ax.set_xlim(3e-3, 3e3)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{names[c]}\n{int(lv.loc[c].P)} P/LP, {int(lv.loc[c].B):,} B/LB" for c in order], fontsize=6)
    ax.set_xlabel("Interval likelihood ratio (exact 95% CI)")
    panel(ax, "A")

    ax = fig.add_subplot(gs[1])
    rule = q[q.section == "rule"]
    groups = []
    for cohort, cl in [("calibration cohort, >= 1 star", "Calibration cohort"), ("resolved VUS, >= 1 star", "Resolved VUS")]:
        for cls, col in [("P/LP", "P"), ("B/LB", "B")]:
            for rl, rlab in [("MetaSplice", "MetaSplice rule"), ("SpliceAI raw, first record", "SpliceAI rule")]:
                d = rule[(rule.cohort == cohort) & (rule.rule == rl)].set_index("code")
                tot = d[col].sum()
                assert tot == d["N" + col].iloc[0], (cohort, rl, cls)
                groups.append((f"{cl}, {'later ' if cl == 'Resolved VUS' else ''}{cls} (n={int(tot):,})", rlab,
                               d.loc["PP3", col] / tot, d.loc["neither", col] / tot, d.loc["BP4", col] / tot))
    ys = []
    y = 0
    for i, g in enumerate(groups):
        if i and i % 2 == 0:
            y -= 1.0
        ys.append(y); y -= 1
    for yy, (grp, rlab, pp3, none, bp4) in zip(ys, groups):
        ax.barh(yy, pp3, color="#CC79A7", height=0.8)
        ax.barh(yy, none, left=pp3, color="#e4e7eb", height=0.8)
        ax.barh(yy, bp4, left=pp3 + none, color="#009E73", height=0.8)
        fmt = lambda v: f"{v:.1%}" if v >= 0.01 or v == 0 else f"{v:.2%}"
        if pp3 >= 0.10:
            ax.text(pp3 / 2, yy, fmt(pp3), ha="center", va="center", fontsize=5.6, color="white")
        else:
            ax.text(pp3 + 0.01, yy, fmt(pp3), ha="left", va="center", fontsize=5.6, color="#8a3c6b")
        if bp4 >= 0.10:
            ax.text(pp3 + none + bp4 / 2, yy, fmt(bp4), ha="center", va="center", fontsize=5.6, color="white")
        else:
            ax.text(1.015, yy, fmt(bp4), ha="left", va="center", fontsize=5.6, color="#00664a", clip_on=False)
    ax.set_yticks(ys)
    ax.set_yticklabels([rlab for _, rlab, *_ in groups], fontsize=6)
    for i in range(0, len(groups), 2):
        ax.text(0, ys[i] + 0.62, groups[i][0], ha="left", va="bottom", fontsize=6, color=INK)
    ax.set_xlim(0, 1); ax.set_xlabel("Share of variants")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#CC79A7", label="PP3"), Patch(color="#e4e7eb", label="No code"),
                       Patch(color="#009E73", label="BP4")], loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3,
              fontsize=6, handlelength=1)
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 1.3)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.text(-0.30, 1.04, "B", transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, va="bottom")
    save(fig, "Figure4")


if __name__ == "__main__":
    fig4()
