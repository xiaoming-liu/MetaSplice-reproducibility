"""Figures 2-5 for MetaSplice_AJHG_v2 (draft). Every value is read from the committed result files in
reproducibility_v2/results/; nothing is hard-coded except labels and the ClinGen requirement constants, which are
checked against S6_thresholds.tsv.

  Figure 2  discrimination on the temporal test set        Q1_pooled_pr_curves.tsv, Q1_pooled_comparison.tsv
  Figure 3  what the score depends on                        P1_..._pooled.tsv, P2_..._pooled.tsv, Q1_pooled_comparison.tsv
  Figure 4  measured splicing and literature-confirmed       A6_splicevardb.txt, literature_recovery.txt
  Figure 5  ACMG/AMP evidence                                Q2_pooled_calibration_rule.tsv, S6_thresholds.tsv

Writes paper/AJHG/Fig{2,3,4,5}_v2.pdf and .png.
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


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "..", "..", "figures", f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ Figure 2
def fig2():
    q = tsv("Q1_pooled_comparison.tsv")
    pr = tsv("Q1_pooled_pr_curves.tsv")
    main = q[q.section.str.startswith("1.")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1, 1.05, 1], "wspace": 0.75})

    def curves(ax, row, title):
        for tool in ["MetaSplice", "SpliceAI", "CADD"]:
            p = pr[(pr.row == row) & (pr.tool == tool)]
            m = main[(main.row == row) & (main.tool == tool)].iloc[0]
            ax.plot(p.recall, p.precision, color=COL[tool], lw=1.1,
                    label=f"{tool} {m.auprc:.3f} ({m.coverage:.1%})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(title, fontsize=7, color=INK, loc="left")
        ax.legend(loc="lower left", fontsize=6, handlelength=1.2, title="auPRC (coverage)", title_fontsize=6)
        ax.grid(color=GRID, lw=0.4)

    n = main[(main.row == "non-canonical") & (main.tool == "MetaSplice")].iloc[0]
    curves(axes[0], "non-canonical", f"Non-canonical\n{int(n.n):,} variants, {int(n.P)} P/LP")
    panel(axes[0], "A")

    ax = axes[1]
    ys = np.arange(len(REGIONS))[::-1]
    for k, (tool, off) in enumerate([("SpliceAI", 0.15), ("CADD", -0.15)]):
        d = main[main.tool == f"MetaSplice minus {tool}"].set_index("row").loc[REGIONS]
        ax.errorbar(d["diff"], ys + off, xerr=[d["diff"] - d.lo, d.hi - d["diff"]], fmt="o", ms=3.2,
                    color=COL[tool], ecolor=COL[tool], elinewidth=0.9, capsize=0, label=f"MetaSplice minus {tool}")
    ax.axvline(0, color=MUTED, lw=0.6, ls="--")
    ax.set_yticks(ys); ax.set_yticklabels([LABEL[r] for r in REGIONS])
    ax.set_xlabel("Paired difference in auPRC (95% CI)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6, handlelength=1, ncol=1)
    ax.grid(axis="x", color=GRID, lw=0.4)
    ax.set_yticklabels([LABEL[r] for r in REGIONS], fontsize=6.2)
    panel(ax, "B")

    a = main[(main.row == "all non-exonic") & (main.tool == "MetaSplice")].iloc[0]
    curves(axes[2], "all non-exonic", f"All non-exonic\n{int(a.n):,} variants, {int(a.P):,} P/LP")
    panel(axes[2], "C")
    save(fig, "Fig2_v2")


# ------------------------------------------------------------------ Figure 3
def fig3():
    p1 = tsv("P1_inputs_baselines_ablations_pooled.tsv")
    p2 = tsv("P2_importance_within_gene_pooled.tsv")
    q = tsv("Q1_pooled_comparison.tsv")
    nc = p1[(p1.cohort == "pooled") & (p1.row == "non-canonical")]
    main = q[q.section.str.startswith("1.") & (q.row == "non-canonical")].set_index("tool")
    gd = q[q.section.str.startswith("2.") & (q.row == "non-canonical")].iloc[0]

    def pick(section, name):
        r = nc[(nc.section == section) & (nc.comparator == name)]
        assert len(r) == 1, (section, name, len(r))
        return r.iloc[0]

    ab = nc[nc.section == "ablation"].comparator.tolist()
    assert len(ab) == 4, ab
    bars = [("MetaSplice (38 features)", main.loc["MetaSplice"], COL["MetaSplice"]),
            ("Gene-disjoint retraining", gd, "#56B4E9")]
    names = ["(a) without gene-level features", "(b) without DANN and GPN-MSA", "(c) without both",
             "(d) (c), gene-disjoint"]
    for lab, name in zip(names, ab):
        bars.append((f"Ablation {lab}", pick("ablation", name), "#56B4E9"))
    bars += [("XGBoost, 19 splicing features", pick("baselines", "XGBoost, 19 deep-learning features"), "#9ecae1"),
             ("Logistic regression, 3 inputs", pick("baselines", "logistic regression, 3 features"), "#9ecae1"),
             ("Pangolin maximum alone", pick("each input alone", "Pangolin max"), "#c6dbef"),
             ("SpliceTransformer alone", pick("each input alone", "SpliceTransformer"), "#c6dbef")]

    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.05, 0.75], wspace=0.75)
    ax = fig.add_subplot(gs[0])
    ys = np.arange(len(bars))[::-1]
    for y, (lab, r, c) in zip(ys, bars):
        ax.barh(y, r.auprc, color=c, height=0.66)
        ax.plot([r.lo, r.hi], [y, y], color=INK, lw=0.7)
        ax.text(0.02, y, f"{r.auprc:.3f}", va="center", ha="left", fontsize=5.8, color="white" if c == COL["MetaSplice"] else INK)
    for tool in ["SpliceAI", "CADD"]:
        v = main.loc[tool].auprc
        ax.axvline(v, color=COL[tool], lw=0.8, ls="--")
        ax.text(v, ys[0] + 0.85, tool, color=COL[tool], fontsize=6, ha="center")
    ax.set_yticks(ys); ax.set_yticklabels([b[0] for b in bars], fontsize=6.2)
    ax.set_xlim(0, 1); ax.set_xlabel("auPRC, non-canonical variants (95% CI)")
    ax.set_ylim(-0.7, len(bars) - 0.1)
    panel(ax, "A")

    ax = fig.add_subplot(gs[1])
    imp = p2[(p2.section == "importance") & (p2.scope == "non-canonical")]
    tr = imp[imp.set == "training, out of fold"].set_index("group").mean_abs_shap
    te = imp[imp.set == "pooled, full model"].set_index("group").mean_abs_shap
    order = tr.sort_values().index
    ys = np.arange(len(order))
    ax.barh(ys + 0.19, tr[order], height=0.36, color="#9ecae1", label="Training, out of fold")
    ax.barh(ys - 0.19, te[order], height=0.36, color=COL["MetaSplice"], label="Temporal test set")
    ax.set_yticks(ys); ax.set_yticklabels(order)
    ax.set_xlabel("Mean |SHAP|, summed within group")
    ax.legend(loc="lower right", fontsize=6, handlelength=1)
    panel(ax, "B")

    ax = fig.add_subplot(gs[2])
    wg = p2[p2.section == "within-gene concordance"].set_index("scope")
    for k, tool in enumerate(["MetaSplice", "SpliceAI", "CADD"]):
        r = wg.loc[tool]
        ax.errorbar(k, r.estimate, yerr=[[r.estimate - r.lo], [r.hi - r.estimate]], fmt="o", ms=4,
                    color=COL[tool], elinewidth=0.9, capsize=0)
    ax.set_xticks(range(3)); ax.set_xticklabels(["MetaSplice", "SpliceAI", "CADD"], rotation=30, ha="right")
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Within-gene concordance (95% CI)")
    ax.grid(axis="y", color=GRID, lw=0.4)
    r = wg.loc["MetaSplice"]
    ax.set_title(f"{int(r.genes)} genes, {int(r.pairs):,} pairs", fontsize=6.5, color=INK, loc="left")
    panel(ax, "C")
    save(fig, "Fig3_v2")


# ------------------------------------------------------------------ Figure 4
def parse_a6():
    txt = open(os.path.join(RES, "A6_splicevardb.txt")).read()
    strict = txt[txt.index("STRICT ("):txt.index("BROAD (")]
    out = {}
    for block, key in [("WITHIN REGION (the interpretable comparison)", "auprc"),
                       ("Within-region AUROC (prevalence-independent):", "auroc")]:
        sub = strict[strict.index(block):]
        rows = {}
        for line in sub.splitlines()[2:]:
            m = re.match(r"\s{2}(\S.*?)\s{2,}([\d,]+)(?:\s+([\d,]+))?\s+(0\.\d{4})\s+(0\.\d{4})\s+(0\.\d{4})\s*$", line)
            if m:
                rows[m.group(1)] = [float(m.group(i)) for i in (4, 5, 6)]
            elif rows and not line.strip():
                break
        out[key] = rows
    assert len(out["auprc"]) == 7 and len(out["auroc"]) == 7, out
    return out


def parse_literature():
    rows = {}
    for line in open(os.path.join(RES, "literature_recovery.txt")):
        m = re.match(r"\s{2}(\S.*?)\s{2,}(\d+)\s+(\d+) \(\s*([\d.]+)%\)\s+(\d+) \(\s*([\d.]+)%\)\s+(\d+) \(\s*([\d.]+)%\)", line)
        if m:
            rows[m.group(1)] = (int(m.group(2)), float(m.group(4)), float(m.group(6)), float(m.group(8)))
    assert len(rows) == 9 and rows["all regions"][0] == 697, rows
    return rows


def fig4():
    a6 = parse_a6()
    lit = parse_literature()
    regs = ["Canonical Acceptor Site", "Donor Region", "Acceptor Region", "PPT Region", "Branch Point Region",
            "Deep Intronic", "Deep Intronic, sensitivity set"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw={"wspace": 0.38})
    for ax, key, ylab, letter in [(axes[0], "auprc", "Within-region auPRC", "A"),
                                  (axes[1], "auroc", "Within-region AUROC", "B")]:
        x = np.arange(len(regs))
        for k, tool in enumerate(["MetaSplice", "SpliceAI", "CADD"]):
            ax.bar(x + (k - 1) * 0.26, [a6[key][r][k] for r in regs], width=0.26, color=COL[tool], label=tool)
        ax.set_xticks(x); ax.set_xticklabels([LABEL[r].replace("\n", " ") for r in regs], rotation=40, ha="right")
        ax.set_ylim(0, 1.0); ax.set_ylabel(ylab)
        ax.grid(axis="y", color=GRID, lw=0.4); ax.set_axisbelow(True)
        if key == "auroc":
            ax.axhline(0.5, color=MUTED, lw=0.6, ls=":")
        panel(ax, letter)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.58, 1.02), fontsize=6, handlelength=1, ncol=3)
    ax = axes[2]
    lregs = ["all regions", "Canonical Donor Site", "Canonical Acceptor Site", "Donor Region", "Acceptor Region",
             "PPT Region", "Branch Point Region", "Deep Intronic", "Deep Intronic, sensitivity set"]
    x = np.arange(len(lregs))
    for k, (thr, c) in enumerate([("0.1", "#9ecae1"), ("0.2", "#56B4E9"), ("0.5", COL["MetaSplice"])]):
        ax.bar(x + (k - 1) * 0.26, [lit[r][1 + k] for r in lregs], width=0.26, color=c, label=f"score ≥ {thr}")
    ax.set_xticks(x)
    ax.set_xticklabels([("All" if r == "all regions" else LABEL[r].replace("\n", " ")) + f" ({lit[r][0]})"
                        for r in lregs], rotation=40, ha="right")
    ax.set_ylim(0, 100); ax.set_ylabel("Literature variants recovered (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.6, 1.02), fontsize=6, handlelength=1, ncol=3, columnspacing=0.8)
    ax.grid(axis="y", color=GRID, lw=0.4); ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=6)
    panel(ax, "C")
    save(fig, "Fig4_v2")


# ------------------------------------------------------------------ Figure 5
def fig5():
    q = tsv("Q2_pooled_calibration_rule.tsv")
    thr = tsv("S6_thresholds.tsv")
    need = thr[thr.tool == "MetaSplice"].set_index("level").LR_needed
    lv = q[(q.section == "S6 levels") & (q.cohort == "calibration cohort, >= 1 star") & (q.rule == "MetaSplice")]
    lv = lv[lv.code != "Indeterminate"].set_index("code")
    order = ["PP3_VeryStrong", "PP3_Strong", "PP3_Moderate", "PP3_Supporting", "BP4_Supporting", "BP4_Moderate",
             "BP4_Strong"]
    names = {"PP3_VeryStrong": "PP3 Very Strong", "PP3_Strong": "PP3 Strong", "PP3_Moderate": "PP3 Moderate",
             "PP3_Supporting": "PP3 Supporting", "BP4_Supporting": "BP4 Supporting", "BP4_Moderate": "BP4 Moderate",
             "BP4_Strong": "BP4 Strong interval (not used at Strong)"}
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
    ax.set_xscale("log"); ax.set_xlim(1e-3, 3e5)
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
    save(fig, "Fig5_v2")


if __name__ == "__main__":
    fig2(); fig3(); fig4(); fig5()
