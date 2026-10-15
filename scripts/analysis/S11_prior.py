"""S11_prior -- Estimate and freeze the prior for rare non-canonical variants (docs/METASPLICE_V2_PLAN.md, Task 11,
Steps 3-4; the method of docs/AJHG_ANALYSIS_PLAN.md Task 3.3, v2.0 only).

The prior is the mixing weight of the P/LP component in the population reference's score distribution, with
the components fixed from labelled ClinVar scores (calib.estimate_prior: EM on logit-score bins). Out-of-fold
scores on both sides:
  components  the >= 1-star derivation cohort (training; non-canonical, gnomAD AF < 0.01, gene with >= 1 P/LP),
              MetaSplice v2.0 out-of-fold scores, split by label;
  reference   the uniform gnomAD v4.1 sample (primary), each variant scored by the fold model that excluded its
              gene (MetaSplice_fold, S11_reference_scores.py).
The temporal validation cohort is not read here: no validation-cohort loader is imported or called.

Rows, each with the estimate, a 95% CI (calib.prior_bootstrap: 1,000 iterations, components resampled) and the
fit (calib.mixture_fit): the primary row (50 bins, pooled, all allele frequencies, uniform sample, all genes),
and one-factor-at-a-time variations around it -- 25 and 100 bins; reference AF < 0.001 and 0.001 <= AF < 0.01;
each region with >= 25 P/LP in the >= 1-star derivation cohort, with the deep-intronic sensitivity set beside
Deep Intronic; the same regions in the stratified sample (secondary check); genes outside the fold map excluded.

Promotion rule (fixed in the plan before this ran): the primary estimate becomes the prior only if its fit is
<= 0.02, the 25- and 100-bin estimates are within a factor of 1.5 of it, and its CI excludes 0.0441. Otherwise
0.0441 stays the prior and the estimate is reported as sensitivity.

Writes results/S11_prior.txt, results/S11_prior.tsv, results/S11_prior_frozen.json and figures/S11_prior_fit.png.
"""
import datetime, hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import NONCANONICAL, ROOT, Tee, deep_intronic_sensitivity
from calib import PRIOR, estimate_prior, load_derivation_cohort, mixture_fit, prior_bootstrap

PRIMARY = dict(bins=50, af="all", region="pooled", sample="uniform", genes="all")
INPUTS = ["data/v2_gnomad_reference_scores.csv", "data/v2_oof.csv", "data/v2_features_training.csv",
          "data/v2_pathogenic_genes.csv", "models/metasplice_v2_folds.json"]


def md5(path):
    return hashlib.md5(open(f"{ROOT}/{path}", "rb").read()).hexdigest()


def select(der, ref, row):
    """Components (P/LP and B/LB scores) and reference scores for one row of the grid."""
    d, r = der, ref[ref["sample"] == row["sample"]]
    if row["region"] == "Deep Intronic, sensitivity set":
        d, r = d[deep_intronic_sensitivity(d).values], r[deep_intronic_sensitivity(r).values]
    elif row["region"] != "pooled":
        d, r = d[d.region == row["region"]], r[r.region == row["region"]]
    if row["af"] == "AF < 0.001":
        r = r[r.gnomad_af < 0.001]
    elif row["af"] == "0.001 <= AF < 0.01":
        r = r[(r.gnomad_af >= 0.001) & (r.gnomad_af < 0.01)]
    if row["genes"] == "in the fold map":
        r = r[r.gene_in_training]
    return d.MetaSplice.values[d.label.values == 1], d.MetaSplice.values[d.label.values == 0], r.MetaSplice_fold.values


def evaluate(args):
    row, s_p, s_b, ref = args
    est = estimate_prior(ref, s_p, s_b, row["bins"])
    lo, hi = prior_bootstrap(ref, s_p, s_b, row["bins"], n_boot=1000, seed=42, resample_components=True)
    return dict(row, n_ref=len(ref), n_P=len(s_p), n_B=len(s_b), prior=est, lo=lo, hi=hi,
                fit=mixture_fit(ref, s_p, s_b, est, row["bins"]))


def figure(ref, s_p, s_b, pi, bins, path):
    """Observed reference histogram against the fitted mixture, on logit-score bins (primary row)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from calib import _components, _logit
    f_p, f_b, n = _components(ref, s_p, s_b, bins)
    z_all = _logit(np.concatenate([s_p, s_b, ref]))
    edges = np.unique(np.quantile(z_all, np.linspace(0, 1, bins + 1)))
    mid = (edges[:-1] + edges[1:]) / 2
    obs, fitted, pathogenic = n / n.sum(), pi * f_p + (1 - pi) * f_b, pi * f_p
    ink, muted, grid, bar, path_c = "#222222", "#5f6368", "#e3e3e3", "#b9c0c7", "#b03a2e"
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(mid, obs, width=np.diff(edges) * 0.92, color=bar, label="gnomAD reference (observed)", zorder=2)
    ax.plot(mid, fitted, color=ink, lw=1.8, marker="o", ms=3, label=f"fitted mixture, prior {pi:.4f}", zorder=3)
    ax.fill_between(mid, 0, pathogenic, color=path_c, alpha=0.35, lw=0, label="P/LP component x prior", zorder=2)
    ax.set_xlabel("MetaSplice v2.0 fold score, logit scale (bins of equal count over all scores)", color=ink, fontsize=8)
    ax.set_ylabel("share of reference variants", color=ink, fontsize=8)
    ax.grid(axis="y", color=grid, lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7, colors=muted, labelcolor=ink)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title("Primary row: uniform gnomAD sample, 50 bins, pooled, all allele frequencies", fontsize=8.5, loc="left", color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    log = Tee(f"{ROOT}/results/S11_prior.txt")
    log("=" * 96)
    log("S11_prior -- PRIOR FOR RARE NON-CANONICAL VARIANTS, MetaSplice v2.0 (Task 11, Steps 3-4)")
    log("=" * 96)
    der = load_derivation_cohort(one_star_only=True)
    ref = pd.read_csv(f"{ROOT}/data/v2_gnomad_reference_scores.csv", low_memory=False)
    log(f"components: >= 1-star derivation cohort, out-of-fold: P/LP {int(der.label.sum()):,}, B/LB {int((der.label == 0).sum()):,}")
    log(f"reference: uniform {int((ref['sample'] == 'uniform').sum()):,}, stratified {int((ref['sample'] == 'stratified').sum()):,} "
        f"(fold scores; genes outside the fold map flagged)")
    p_by_region = der[der.label == 1].region.value_counts()
    regions = [r for r in NONCANONICAL if p_by_region.get(r, 0) >= 25]
    log(f"P/LP by region in the >= 1-star derivation cohort: {p_by_region.to_dict()}; regions with >= 25: {regions}")

    rows = [dict(PRIMARY), dict(PRIMARY, bins=25), dict(PRIMARY, bins=100),
            dict(PRIMARY, af="AF < 0.001"), dict(PRIMARY, af="0.001 <= AF < 0.01"),
            dict(PRIMARY, genes="in the fold map")]
    for sample in ["uniform", "stratified"]:
        for r in regions:
            rows.append(dict(PRIMARY, region=r, sample=sample))
            if r == "Deep Intronic":
                rows.append(dict(PRIMARY, region="Deep Intronic, sensitivity set", sample=sample))
    jobs = [(row,) + select(der, ref, row) for row in rows]
    with ProcessPoolExecutor(max_workers=min(12, len(jobs))) as ex:
        res = pd.DataFrame(list(ex.map(evaluate, jobs)))
    res.to_csv(f"{ROOT}/results/S11_prior.tsv", sep="\t", index=False)
    log(f"\n{'sample':<11}{'region':<32}{'AF stratum':<20}{'genes':<17}{'bins':>5}{'n ref':>8}{'P':>6}{'B':>8}"
        f"{'prior':>9}{'95% CI':>20}{'fit':>8}")
    for r in res.itertuples():
        log(f"{r.sample:<11}{r.region:<32}{r.af:<20}{r.genes:<17}{r.bins:>5}{r.n_ref:>8,}{r.n_P:>6,}{r.n_B:>8,}"
            f"{r.prior:>9.4f}   [{r.lo:.4f}, {r.hi:.4f}]{r.fit:>8.4f}")

    p = res.iloc[0]
    b25, b100 = res.iloc[1].prior, res.iloc[2].prior
    checks = {"fit <= 0.02": bool(p.fit <= 0.02),
              "25- and 100-bin estimates within a factor of 1.5": bool(max(b25, b100, p.prior) / min(b25, b100, p.prior) <= 1.5),
              "95% CI excludes 0.0441": bool(not (p.lo <= PRIOR <= p.hi))}
    promoted = all(checks.values())
    failed = [k for k, v in checks.items() if not v]
    reason = ("all three promotion conditions hold" if promoted else "not promoted; failed: " + "; ".join(failed))
    prior = float(p.prior) if promoted else PRIOR
    log("\nDecision (promotion rule fixed in the plan before this run):")
    for k, v in checks.items():
        log(f"  {'PASS' if v else 'FAIL'}  {k}")
    log(f"  primary estimate {p.prior:.4f} [{p.lo:.4f}, {p.hi:.4f}], fit {p.fit:.4f}; 25 bins {b25:.4f}; 100 bins {b100:.4f}")
    log(f"  -> prior for Stage 3: {prior:.4f} ({reason})")
    frozen = dict(prior=prior, promoted=promoted, reason=reason, estimate=float(p.prior), ci95=[float(p.lo), float(p.hi)],
                  fit=float(p.fit), estimate_25_bins=float(b25), estimate_100_bins=float(b100), primary_row=PRIMARY,
                  model="MetaSplice v2.0 (out-of-fold components; gene-matched fold scores of the reference)",
                  created=datetime.datetime.now().isoformat(timespec="seconds"),
                  inputs={f: md5(f) for f in INPUTS})
    json.dump(frozen, open(f"{ROOT}/results/S11_prior_frozen.json", "w"), indent=2)
    os.makedirs(f"{ROOT}/figures", exist_ok=True)
    s_p, s_b, r0 = jobs[0][1:]
    figure(r0, s_p, s_b, float(p.prior), PRIMARY["bins"], f"{ROOT}/figures/S11_prior_fit.png")
    log("\nwrote results/S11_prior.tsv, results/S11_prior_frozen.json and figures/S11_prior_fit.png")
    log.close()


if __name__ == "__main__":
    main()
