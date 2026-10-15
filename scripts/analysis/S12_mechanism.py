"""S12 -- P/LP variants that MetaSplice v2.0 assigns candidate PP3 and SpliceAI does not (Task 11 Step 7; AJHG plan
Task 3.5, v2.0 only). Descriptive.

Groups (validation cohort of the strict test set: non-canonical, gnomAD AF < 0.01, genes with >= 1 P/LP; SpliceAI-scored
P/LP only): both PP3; MetaSplice-only PP3 (SpliceAI indeterminate or BP4); SpliceAI-only PP3; neither. Evidence uses the
S6 (main) candidate thresholds for both tools -- the AJHG plan's code used S5's; S6's are the main thresholds S7-S9 use
(decided at the 2026-10-01 resumption, before this ran). Thresholds are exploratory candidate thresholds under the
fallback prior.
TreeSHAP contributions of the full v2.0 model are summed by feature group (v2.0's groups: conservation, DANN, GPN-MSA,
gene-level, exon context, SpliceTransformer, Pangolin, region). A 'splice-model signal' is pangolin_max >= 0.2 or
spt_orig >= 0.2; MetaSplice-only calls without it are reported as driven by other groups (conservation, gene-level,
exon context including distance), which may reflect non-splicing mechanisms, gene priors or ascertainment.
Replicated on the derivation cohort with the five fold models, each applied to its own out-of-fold variants.

Writes results/S12_mechanism.txt and .tsv, and figures/supplementary_fig3.pdf/.png.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from common import FEATURES, MODEL_DIR, ROOT, Tee
from calib import LEVELS, assign, load_derivation_cohort, load_thresholds, load_validation_cohort

S6 = f"{ROOT}/results/S6_thresholds.tsv"
GROUPS = {"conservation": ["phyloP100way_vertebrate", "phyloP470way_mammalian", "phastCons100way_vertebrate", "phastCons470way_mammalian"],
          "DANN": ["DANN_score"], "GPN-MSA": ["gpn_msa_score"],
          "gene-level": ["gnomAD_pLI", "gnomAD_LOEUF", "ClinGen_Haploinsufficiency_Score"],
          "exon context": ["exon_length", "exon_length_mod3", "dist_to_boundary"],
          "SpliceTransformer": [f for f in FEATURES if f.startswith("spt_")],
          "Pangolin": ["pangolin_inc", "pangolin_dec", "pangolin_max"],
          "region": [f for f in FEATURES if f.startswith("r_")]}
PP3 = [f"PP3_{l}" for l in LEVELS]


def contributions(booster, d):
    X = d[FEATURES].astype(np.float32).values
    dm = xgb.DMatrix(X)
    contrib = booster.predict(dm, pred_contribs=True)
    margin = booster.predict(dm, output_margin=True)
    worst = float(np.abs(contrib.sum(axis=1) - margin).max())
    if worst > 1e-4:
        raise ValueError(f"SHAP contributions plus bias do not reproduce the margin (max |diff| {worst:.2e})")
    return contrib[:, :-1], worst


def grouped(d, ms_col):
    ms = np.isin(assign(d[ms_col].values.astype(float), *load_thresholds("MetaSplice", path=S6)), PP3)
    sa = np.isin(assign(d.SpliceAI.values.astype(float), *load_thresholds("SpliceAI", path=S6)), PP3)
    return np.select([ms & sa, ms & ~sa, ~ms & sa], ["both", "MetaSplice only", "SpliceAI only"], "neither")


def summarise(log, d):
    d["splice_signal"] = (d.pangolin_max >= 0.2) | (d.spt_orig >= 0.2)
    d["abs_dist"] = d.dist_to_boundary.abs()
    summary = d.groupby("group").agg(n=("label", "size"), median_abs_dist=("abs_dist", "median"), splice_signal=("splice_signal", "mean"),
                                     **{f"shap_{g}": (f"shap_{g}", "mean") for g in GROUPS})
    log(summary.to_string(float_format=lambda v: f"{v:.3g}"))
    log("\n  region mix by group:\n" + pd.crosstab(d.group, d.region).to_string())
    return summary


def main():
    log = Tee(f"{ROOT}/results/S12_mechanism.txt")
    log("=" * 96)
    log("S12 -- P/LP CALLED PP3 BY METASPLICE v2.0 AND NOT BY SPLICEAI (S6 candidate thresholds; descriptive)")
    log("=" * 96)
    val = load_validation_cohort("strict")
    d = val[(val.label == 1) & val.SpliceAI.notna()].copy().reset_index(drop=True)
    d["group"] = grouped(d, "MetaSplice")
    booster = joblib.load(f"{MODEL_DIR}/metasplice_v2_full.joblib").get_booster()
    contrib, worst = contributions(booster, d)
    for g, cols in GROUPS.items():
        d[f"shap_{g}"] = contrib[:, [FEATURES.index(c) for c in cols]].sum(axis=1)
    log(f"\nValidation cohort, SpliceAI-scored P/LP: n={len(d)}; SHAP sums plus bias reproduce the margin (max |diff| {worst:.1e})")
    summ = summarise(log, d)
    d.drop(columns=FEATURES).to_csv(f"{ROOT}/results/S12_mechanism.tsv", sep="\t", index=False)

    log("\nReplication on the derivation cohort (training, out of fold; each variant explained by the fold model that held it out)")
    der = load_derivation_cohort()
    r = der[(der.label == 1) & der.SpliceAI.notna()].copy().reset_index(drop=True)
    r["group"] = grouped(r, "MetaSplice")
    folds = json.load(open(f"{MODEL_DIR}/metasplice_v2_folds.json"))
    k = r.gene_id.map(folds)
    if k.isna().any():
        raise ValueError("a derivation variant has no fold")
    worst = 0.0
    for f in range(5):
        m = (k == f).values
        b = joblib.load(f"{MODEL_DIR}/metasplice_v2_fold{f}.joblib").get_booster()
        c, w = contributions(b, r[m])
        worst = max(worst, w)
        for g, cols in GROUPS.items():
            r.loc[m, f"shap_{g}"] = c[:, [FEATURES.index(x) for x in cols]].sum(axis=1)
    log(f"  derivation, SpliceAI-scored P/LP: n={len(r):,}; SHAP check max |diff| {worst:.1e}")
    rep = summarise(log, r)
    same_order = list(summ.loc["MetaSplice only"].filter(like="shap_").sort_values(ascending=False).index[:3]) if "MetaSplice only" in summ.index else []
    rep_order = list(rep.loc["MetaSplice only"].filter(like="shap_").sort_values(ascending=False).index[:3]) if "MetaSplice only" in rep.index else []
    log(f"\n  top three groups for 'MetaSplice only': validation {same_order}; derivation {rep_order}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#222222", "#5f6368", "#e3e3e3"
    colors = {"both": "#1f5f8b", "MetaSplice only": "#b0482e"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    names = list(GROUPS)
    for j, (grp, off) in enumerate([("both", -0.15), ("MetaSplice only", 0.15)]):
        if grp in summ.index:
            vals = [summ.loc[grp, f"shap_{g}"] for g in names]
            ax.scatter(vals, np.arange(len(names)) + off, s=26, color=colors[grp], label=f"{grp} (n={int(summ.loc[grp, 'n'])})", zorder=3)
    ax.axvline(0, color=muted, lw=0.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7.5)
    ax.invert_yaxis(); ax.grid(axis="x", color=grid, lw=0.6)
    ax.set_xlabel("mean SHAP contribution (log-odds)", fontsize=8, color=ink)
    ax.set_title("a  Feature groups behind PP3 calls (validation P/LP)", fontsize=8.5, loc="left", color=ink)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax = axes[1]
    mix = pd.crosstab(d.group, d.region, normalize="index").reindex(["both", "MetaSplice only"]).dropna(how="all")
    left = np.zeros(len(mix))
    palette = ["#1f5f8b", "#5b8db8", "#9fbfd9", "#d9a441", "#b0482e"]
    for i, reg in enumerate(mix.columns):
        ax.barh(range(len(mix)), mix[reg].values, left=left, color=palette[i % len(palette)], label=reg, height=0.55,
                edgecolor="white", linewidth=1)
        left += mix[reg].values
    ax.set_yticks(range(len(mix))); ax.set_yticklabels(mix.index, fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("share of the group's variants", fontsize=8, color=ink)
    ax.set_title("b  Region mix", fontsize=8.5, loc="left", color=ink)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=7, colors=muted, labelcolor=ink)
    fig.tight_layout()
    os.makedirs(f"{ROOT}/figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{ROOT}/figures/supplementary_fig3.{ext}", dpi=200)
    log("\nwrote results/S12_mechanism.tsv and figures/supplementary_fig3.pdf/.png")
    log.close()


if __name__ == "__main__":
    main()
