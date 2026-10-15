"""Figure S2 -- local posterior probability of pathogenicity and ClinGen evidence intervals (v2.0).

v1's figure, ported: MetaSplice v2.0 and SpliceAI (v2.0 has no frequency-free model).
Reads results/S5_curves.joblib written by S5_clingen_calibration.py. One panel per tool
(small multiples; each panel carries a single curve, so the panel title names it). The
curve is the local posterior at the prior 0.0441; the band spans the one-sided 95%
bootstrap bounds used to set thresholds. Grey reference lines mark the posterior each
evidence level requires. The strip under each panel shows the score intervals derived
for each level: blue for BP4, red for PP3, darker for stronger evidence (ordinal ramps
checked for monotone lightness and single hue). A Very Strong interval, where reached,
is drawn as Strong. For missense predictors ClinGen recommends up to PP3 Strong and BP4
Moderate (Bergquist et al. 2025), and its splicing guidance (Walker et al. 2023) applies
splice predictions at Supporting weight, so every level above Supporting here is a candidate.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
from common import OUT

rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                 "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7})
MM = 1 / 25.4
PRIOR = 0.0441
ODDS = PRIOR / (1 - PRIOR)
LR = {"Supporting": 2.406, "Moderate": 5.790, "Strong": 33.53}
INK, MUTED, REF = "#2b2b2b", "#6b6b6b", "#bdbdbd"
BLUE = {"Supporting": "#86b6ef", "Moderate": "#3987e5", "Strong": "#1c5cab"}
RED = {"Supporting": "#f09a93", "Moderate": "#e34948", "Strong": "#b02a2b"}
NEUTRAL = "#f0efec"

PANELS = [("MetaSplice", "a  MetaSplice v2.0", "MetaSplice v2.0 score", "logit"),
          ("SpliceAI", "b  SpliceAI", "SpliceAI maximum delta score", "linear")]


def post(lr):
    lr = np.asarray(lr, dtype=float)
    p = lr * ODDS / (1 + lr * ODDS)
    p = np.where(np.isinf(lr), 1.0, p)
    return np.clip(p, 2e-4, 1 - 1e-4)


def intervals(thr, lo, hi):
    """Ordered (start, end, colour) segments of the evidence strip on [lo, hi]."""
    pp3, bp4 = thr
    segs, cur = [], lo
    for lev in ["Strong", "Moderate", "Supporting"]:          # benign side, lowest scores first
        t = bp4.get(lev)
        if t is not None and t > cur:
            segs.append((cur, t, BLUE[lev]))
            cur = t
    strong = pp3.get("Strong") if pp3.get("Strong") is not None else pp3.get("VeryStrong")
    pts = [(t, l) for t, l in [(pp3.get("Supporting"), "Supporting"),
                               (pp3.get("Moderate"), "Moderate"), (strong, "Strong")] if t is not None]
    segs.append((cur, pts[0][0] if pts else hi, NEUTRAL))
    for i, (t, l) in enumerate(pts):
        segs.append((t, pts[i + 1][0] if i + 1 < len(pts) else hi, RED[l]))
    return segs


def main():
    curves = joblib.load(f"{OUT}/results/S5_curves.joblib")
    fig = plt.figure(figsize=(130 * MM, 78 * MM))
    gs = fig.add_gridspec(2, len(PANELS), height_ratios=[1, 0.07], hspace=0.08, wspace=0.12,
                          left=0.10, right=0.83, top=0.90, bottom=0.25)
    for j, (key, title, xlabel, xscale) in enumerate(PANELS):
        c = curves[key]
        g = c["grid"].astype(float)
        if xscale == "logit":
            keep = (g > 1e-6) & (g < 1 - 1e-6)
            lo_x, hi_x = 1e-6, 1 - 1e-6
        else:
            keep = np.ones_like(g, dtype=bool)
            lo_x, hi_x = -0.01, 1.01
        ax = fig.add_subplot(gs[0, j])
        strip = fig.add_subplot(gs[1, j], sharex=ax)
        for lev, lr in LR.items():
            for yv in (post(lr), post(1 / lr)):
                ax.axhline(yv, color=REF, lw=0.6, zorder=1)
        ax.fill_between(g[keep], post(c["lower"][keep]), post(c["upper"][keep]),
                        color=INK, alpha=0.13, lw=0, zorder=2)
        ax.plot(g[keep], post(c["est"][keep]), color=INK, lw=1.1, zorder=3,
                solid_joinstyle="round", solid_capstyle="round")
        ax.set_xscale(xscale)
        ax.set_yscale("logit")
        ax.set_ylim(5e-4, 0.9995)
        ax.set_xlim(lo_x if xscale == "linear" else 1e-6, hi_x)
        ax.set_title(title, loc="left", fontsize=7.5, color=INK)
        # upper left is empty in every panel: the curves rise from lower left to upper right
        ax.text(0.03, 0.97, f"n = {c['n']:,}\n{c['P']:,} pathogenic", transform=ax.transAxes,
                fontsize=6, color=MUTED, va="top", ha="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED); ax.spines[sp].set_linewidth(0.6)
        ax.tick_params(colors=MUTED, labelcolor=INK, width=0.6, length=2.5, labelsize=6)
        ax.tick_params(axis="x", labelbottom=False)
        ax.minorticks_off()
        if xscale == "logit":
            ax.set_xticks([1e-5, 1e-3, 0.1, 0.9, 0.999])
        else:
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])
        if j == 0:
            ax.set_yticklabels(["0.001", "0.01", "0.1", "0.5", "0.9", "0.99", "0.999"])
            ax.set_ylabel("Local posterior probability of pathogenicity", color=INK, fontsize=6.5)
        else:
            ax.set_yticklabels([])
        if j == len(PANELS) - 1:
            for lev, lr in LR.items():
                ax.text(1.03, post(lr), f"PP3 {lev}", transform=ax.get_yaxis_transform(),
                        fontsize=5.8, color=MUTED, va="center")
                ax.text(1.03, post(1 / lr), f"BP4 {lev}", transform=ax.get_yaxis_transform(),
                        fontsize=5.8, color=MUTED, va="center")
        # evidence strip
        for a, b, col in intervals(c["thr"], lo_x, hi_x):
            if b > a:
                strip.axvspan(a, b, color=col, lw=0, zorder=1)
                strip.axvline(a, color="white", lw=1.0, zorder=2)   # 2px-style surface gap
        strip.set_yticks([])
        for sp in ("top", "right", "left"):
            strip.spines[sp].set_visible(False)
        strip.spines["bottom"].set_color(MUTED); strip.spines["bottom"].set_linewidth(0.6)
        strip.tick_params(colors=MUTED, labelcolor=INK, width=0.6, length=2.5, labelsize=6)
        strip.minorticks_off()
        if xscale == "logit":
            strip.set_xticks([1e-5, 1e-3, 0.1, 0.9, 0.999])
            strip.set_xticklabels(["1e-5", "0.001", "0.1", "0.9", "0.999"])
        strip.set_xlabel(xlabel, color=INK, fontsize=6.5)
    handles = ([Patch(color=BLUE[l], label=f"BP4 {l}") for l in ["Strong", "Moderate", "Supporting"]]
               + [Patch(facecolor=NEUTRAL, edgecolor=REF, lw=0.4, label="Indeterminate")]
               + [Patch(color=RED[l], label=f"PP3 {l}") for l in ["Supporting", "Moderate", "Strong"]])
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6,
               handlelength=1.4, handleheight=0.9, columnspacing=1.2, bbox_to_anchor=(0.465, 0.0))
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/figures/supplementary_fig2.{ext}", dpi=300)
    print(f"wrote {OUT}/figures/supplementary_fig2.pdf and .png")


if __name__ == "__main__":
    main()
