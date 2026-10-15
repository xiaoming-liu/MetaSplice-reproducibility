"""S5 -- ClinGen-style PP3/BP4 calibration of MetaSplice v2.0 and SpliceAI (Amendment 3).

v1's S5 method, unchanged (Pejaver et al., Am J Hum Genet 2022;109:2163-2177): prior 0.0441, c = 1124;
the local likelihood ratio estimated in windows of >= 100 labelled variants and >= 25 of each class;
each threshold the most lenient score whose one-sided 95% bootstrap bound (10,000 iterations) meets the
level's LR at every score beyond it; thresholds are candidate thresholds. They are derived on the
training data out of fold and validated on later submissions by interval likelihood ratios with
conservative exact 95% CIs. Sensitivity: windows of >= 10 and >= 50 of each class, >= 3% of derivation
variants, >= 100 labelled only; priors 0.10 and 0.01; thresholds derived directly on the test set; and
SpliceAI's local posterior at the ClinGen splicing cut-offs.

Cohorts (calib.in_cohort: non-canonical, gnomAD AF < 0.01, gene with >= 1 P/LP in the training release):
  derivation  training variants, MetaSplice v2.0 out-of-fold scores (V06), SpliceAI (V08)
  validation  the v2.0 strict test set, the full model's scores
  yield       uncertain and conflicting variants of the 2026-04-15 release (V02), no labels
v1 also calibrated a frequency-free model; v2.0 has no allele-frequency feature, so there is one model.
The by-region check reports the deep-intronic sensitivity set beside Deep Intronic.

Writes results/S5_clingen_calibration.txt, S5_thresholds.tsv, S5_validation_intervals.tsv,
S5_vus_yield.tsv and S5_curves.joblib.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib
import numpy as np
import pandas as pd
from common import (NONCANONICAL, OUT, ROOT, Tee, add_comparators, auprc, deep_intronic_sensitivity, in_training,
                    score)
from calib import (N_BOOT, MIN_LABELLED, MIN_CLASS, LEVELS, PRIOR, C_MAIN, LR_MAIN, CODES, lr_levels, post_from_lr,
                   calibrate, assign, interval_lr, verdicts, fmt_t, pathogenic_genes, in_cohort,
                   load_derivation_cohort, load_validation_cohort)

TOOLS = ["MetaSplice", "SpliceAI"]
PRIORS = ((PRIOR, 1124.0), (0.10, None), (0.01, None))
log = Tee(f"{OUT}/results/S5_clingen_calibration.txt")


def report_thresholds(res, prior=PRIOR):
    cc, need, pp3, bp4 = res["by_prior"][prior]
    log(f"  {res['name']}: n={res['n']:,} (P={res['P']:,}, B={res['B']:,}); grid {len(res['grid'])} points; "
        f"median window half-width {np.median(res['eps']):.3g}")
    for lev in LEVELS[::-1]:
        log(f"    PP3_{lev:<11} (LR>={need[lev]:.4g}, posterior>={post_from_lr(need[lev], prior):.3f}):  score >= {fmt_t(pp3[lev])}")
    for lev in LEVELS:
        log(f"    BP4_{lev:<11} (LR<={1/need[lev]:.4g}, posterior<={post_from_lr(1/need[lev], prior):.4f}):  score <= {fmt_t(bp4[lev])}")
    return pp3, bp4


def load_yield(pgenes):
    vus = pd.read_csv(f"{ROOT}/data/v2_features_vus.csv", low_memory=False)
    if in_training(vus).any():
        raise ValueError("a VUS is in the training data")
    vus = add_comparators(vus)
    vus["MetaSplice"], vus["SpliceAI"] = score(vus), vus.spliceai_max
    return vus[in_cohort(vus, pgenes)].reset_index(drop=True)


def main():
    log("=" * 88)
    log("S5 -- ClinGen PP3/BP4 CALIBRATION OF METASPLICE v2.0 AND SPLICEAI (Amendment 3; Pejaver et al. 2022)")
    log("=" * 88)
    log(f"prior {PRIOR}; c = {C_MAIN:.0f}; LR needed: " + ", ".join(f"{l} {LR_MAIN[l]:.4g}" for l in LEVELS))
    log(f"windows: >= {MIN_LABELLED} labelled variants and >= {MIN_CLASS} of each class; {N_BOOT:,} bootstrap "
        f"iterations; one-sided 95% bounds")
    der, val = load_derivation_cohort(), load_validation_cohort("strict")
    yld = load_yield(pathogenic_genes())

    log("\nCohorts (non-canonical, gnomAD AF < 0.01, gene with >= 1 P/LP in the training release):")
    for nm, d in [("derivation (training, out of fold)", der), ("validation (strict test set)", val)]:
        log(f"  {nm:<36} n={len(d):,}  P={int(d.label.sum()):,}  B={int((d.label == 0).sum()):,}  "
            + "  ".join(f"{t} scored {d[t].notna().mean():.1%}" for t in TOOLS))
    log(f"  {'yield (VUS and conflicting)':<36} n={len(yld):,}  " + "  ".join(f"{t} scored {yld[t].notna().mean():.1%}" for t in TOOLS))
    log("\nauPRC on the validation cohort, each tool on the variants it scores: "
        + "  ".join(f"{t} {auprc(val.label.values, val[t].values):.4f}" for t in TOOLS))
    sh = val.SpliceAI.notna().values
    log(f"auPRC on the {sh.sum():,} validation variants both tools score: "
        + "  ".join(f"{t} {auprc(val.label.values[sh], val[t].values[sh]):.4f}" for t in TOOLS))
    log(f"SpliceAI leaves {int((~sh).sum()):,} validation variants unscored ({int(val.label.values[~sh].sum())} P/LP)")

    log("\n" + "-" * 88)
    log("1. CANDIDATE THRESHOLDS derived on the training data (out of fold), prior 0.0441")
    log("-" * 88)
    cal, thr = {}, {}
    for t in TOOLS:
        cal[t] = calibrate(der[t].values.astype(float), der.label.values, t, priors=PRIORS)
        thr[t] = report_thresholds(cal[t])

    log("\n" + "-" * 88)
    log("2. VALIDATION on the strict test set: interval LR must meet the level's LR")
    log("   point = point estimate meets the requirement (Pejaver et al.'s criterion);")
    log("   CI = the conservative exact 95% CI also lies wholly on the required side")
    log("-" * 88)
    val_rows = []
    for t in TOOLS:
        codes = assign(val[t].values.astype(float), *thr[t])
        log(f"  {t}")
        log(f"    {'interval':<20}{'P':>6}{'B':>8}{'LR':>10}{'exact 95% CI':>24}  needed    point    CI")
        for code, kp, kb, lr, lo, hi in interval_lr(codes, val.label.values):
            verdict, ci, ns = verdicts(code, lr, lo, hi)
            lrs = "inf" if np.isinf(lr) else f"{lr:.3g}"
            his = "inf" if np.isinf(hi) else f"{hi:.3g}"
            log(f"    {code:<20}{kp:>6}{kb:>8}{lrs:>10}   [{lo:.3g}, {his}]".ljust(64) + f"{ns:<10}{verdict:<9}{ci}")
            val_rows.append((t, code, kp, kb, lr, lo, hi, verdict, ci))
    pd.DataFrame(val_rows, columns=["tool", "interval", "P", "B", "LR", "lo95_exact", "hi95_exact", "point_verdict",
                                    "ci_verdict"]).to_csv(f"{OUT}/results/S5_validation_intervals.tsv", sep="\t", index=False)

    log("\n" + "-" * 88)
    log("3. EVIDENCE ASSIGNED on the strict test set (share of each class)")
    log("-" * 88)
    pp_all, mod_up, bb_all = [f"PP3_{l}" for l in LEVELS], ["PP3_Moderate", "PP3_Strong", "PP3_VeryStrong"], [f"BP4_{l}" for l in LEVELS]
    shared = val[val.SpliceAI.notna()]
    log(f"  on the {len(shared):,} variants SpliceAI scores ({int(shared.label.sum())} P/LP), separating accuracy from coverage:")
    for t in TOOLS:
        codes, yv = assign(shared[t].values.astype(float), *thr[t]), shared.label.values
        P, B = codes[yv == 1], codes[yv == 0]
        log(f"    {t:<16} P with PP3 {np.isin(P, pp_all).mean():.1%} (Moderate+ {np.isin(P, mod_up).mean():.1%}); "
            f"B with BP4 {np.isin(B, bb_all).mean():.1%}; B given PP3 {np.isin(B, pp_all).mean():.2%}; P given BP4 {np.isin(P, bb_all).mean():.1%}")
    log("  on all validation variants (variants a tool does not score receive no evidence):")
    for t in TOOLS:
        codes, yv = assign(val[t].values.astype(float), *thr[t]), val.label.values
        log(f"  {t:<16} P/LP with PP3: {np.isin(codes[yv == 1], pp_all).mean():.1%} (Moderate or stronger "
            f"{np.isin(codes[yv == 1], mod_up).mean():.1%}); B/LB with BP4: {np.isin(codes[yv == 0], bb_all).mean():.1%}")
        log(f"  {'':<16} B/LB given PP3: {np.isin(codes[yv == 0], pp_all).mean():.2%} of B/LB; P/LP given BP4: "
            f"{np.isin(codes[yv == 1], bb_all).mean():.1%} of P/LP")

    log("\n" + "-" * 88)
    log("4. EVIDENCE YIELD on uncertain and conflicting variants (same cohort filters)")
    log("-" * 88)
    ytab = pd.DataFrame({t: pd.Series(assign(yld[t].values.astype(float), *thr[t])).value_counts().reindex(CODES, fill_value=0)
                         for t in TOOLS})
    ytab.loc["total"] = ytab.sum()
    log(ytab.to_string())
    ytab.to_csv(f"{OUT}/results/S5_vus_yield.tsv", sep="\t")
    both = yld.MetaSplice.notna() & yld.SpliceAI.notna()
    ct = pd.crosstab(pd.Series(assign(yld.loc[both, "MetaSplice"].values.astype(float), *thr["MetaSplice"]), name="MetaSplice"),
                     pd.Series(assign(yld.loc[both, "SpliceAI"].values.astype(float), *thr["SpliceAI"]), name="SpliceAI"))
    log(f"\n  VUS scored by both (n={int(both.sum()):,}): MetaSplice evidence (rows) x SpliceAI (columns)")
    log(ct.to_string())

    rows = []
    for t in TOOLS:
        for prior in [p for p, _ in PRIORS]:
            cc, need, pp3, bp4 = cal[t]["by_prior"][prior]
            for lev in LEVELS:
                rows.append((t, prior, round(cc, 1), "PP3_" + lev, need[lev], pp3[lev]))
                rows.append((t, prior, round(cc, 1), "BP4_" + lev, 1 / need[lev], bp4[lev]))
    pd.DataFrame(rows, columns=["tool", "prior", "c", "level", "LR_needed", "score_threshold"]).to_csv(
        f"{OUT}/results/S5_thresholds.tsv", sep="\t", index=False)

    log("\n" + "-" * 88)
    log("5. SENSITIVITY")
    log("-" * 88)
    log("  a. prior of pathogenicity (same local LR curves; only the LR needed changes)")
    for prior in [0.10, 0.01]:
        cc, need = lr_levels(prior)
        log(f"     prior {prior}: c = {cc:.0f}; LR needed " + ", ".join(f"{l} {need[l]:.3g}" for l in LEVELS))
        for t in TOOLS:
            _, _, pp3, bp4 = cal[t]["by_prior"][prior]
            log(f"       {t:<16} PP3 " + "  ".join(f"{l[:3]} {fmt_t(pp3[l])}" for l in LEVELS)
                + "   |  BP4 " + "  ".join(f"{l[:3]} {fmt_t(bp4[l])}" for l in LEVELS))
    log("  b. window size (prior 0.0441)")
    for lab, kw in [(">=10 of each class", dict(min_class=10)), (">=50 of each class", dict(min_class=50)),
                    (">=3% of derivation variants (gnomAD-rule analogue)", dict(min_class=0, min_fraction=0.03)),
                    (">=100 labelled only", dict(min_class=0))]:
        log(f"     {lab}")
        for t in TOOLS:
            r = calibrate(der[t].values.astype(float), der.label.values, t, n_boot=min(N_BOOT, 2000), **kw)
            _, _, pp3, bp4 = r["by_prior"][PRIOR]
            log(f"       {t:<16} PP3 " + "  ".join(f"{l[:3]} {fmt_t(pp3[l])}" for l in LEVELS)
                + "   |  BP4 " + "  ".join(f"{l[:3]} {fmt_t(bp4[l])}" for l in LEVELS))
    log("  c. thresholds derived directly on the strict test set (prior 0.0441)")
    for t in TOOLS:
        r = calibrate(val[t].values.astype(float), val.label.values, t, n_boot=min(N_BOOT, 2000))
        _, _, pp3, bp4 = r["by_prior"][PRIOR]
        log(f"       {t:<16} PP3 " + "  ".join(f"{l[:3]} {fmt_t(pp3[l])}" for l in LEVELS)
            + "   |  BP4 " + "  ".join(f"{l[:3]} {fmt_t(bp4[l])}" for l in LEVELS))
    log("  d. local posterior of SpliceAI at the commonly applied cut-offs 0.2 (PP3) and 0.1 (BP4)")
    r = cal["SpliceAI"]
    for cut in (0.2, 0.1):
        i = int(np.argmin(np.abs(r["grid"] - cut)))
        log(f"       SpliceAI {r['grid'][i]:.2f}: lr {r['est'][i]:.3g} (one-sided 95% bounds {r['lower'][i]:.3g} - "
            f"{r['upper'][i]:.3g}); posterior {post_from_lr(r['est'][i], PRIOR):.3f}")

    log("\n" + "-" * 88)
    log("6. BY REGION on the strict test set (derivation thresholds; interval LR for any PP3 / any BP4)")
    log("-" * 88)
    groups = [(r, val.region == r) for r in NONCANONICAL] + [("Deep Intronic, sensitivity set", deep_intronic_sensitivity(val))]
    for t in TOOLS:
        log(f"  {t}")
        for reg, m in groups:
            d = val[m.values]
            if d.label.sum() < 5:
                log(f"    {reg:<32} P={int(d.label.sum())} -- too few P/LP")
                continue
            codes, yv = assign(d[t].values.astype(float), *thr[t]), d.label.values
            NP_, NB_ = (yv == 1).sum(), (yv == 0).sum()
            pp, bb = np.isin(codes, pp_all), np.isin(codes, bb_all)
            lr_pp = (pp[yv == 1].sum() / NP_) / (pp[yv == 0].sum() / NB_) if pp[yv == 0].sum() else np.inf
            lr_bb = (bb[yv == 1].sum() / NP_) / (bb[yv == 0].sum() / NB_) if bb[yv == 0].sum() else np.nan
            log(f"    {reg:<32} P={NP_:>4} B={NB_:>6}  PP3 in {pp[yv == 1].mean():>5.1%} of P, LR {lr_pp:>7.3g}   "
                f"BP4 in {bb[yv == 0].mean():>5.1%} of B, LR {lr_bb:.3g}")

    joblib.dump({t: {k: cal[t][k] for k in ("grid", "est", "lower", "upper", "n", "P", "B")} | {"thr": thr[t]} for t in TOOLS},
                f"{OUT}/results/S5_curves.joblib")
    log("\nwrote results/S5_thresholds.tsv, S5_validation_intervals.tsv, S5_vus_yield.tsv, S5_curves.joblib")
    log.close()


if __name__ == "__main__":
    main()
