"""S11_recalibrate -- Recalibration with the frozen prior and the population reference, every configuration validated
(docs/METASPLICE_V2_PLAN.md, Task 11 Step 6; the method of docs/AJHG_ANALYSIS_PLAN.md Task 3.4, v2.0 only).

The prior comes only from results/S11_prior_frozen.json (committed 396cdd7): 0.0441, not promoted. It is the
procedure's fallback, not a validated population prior, so the population-yield flags below are relative to it, and
every threshold here is an exploratory candidate threshold, not clinically validated (Amendment 4 Outcome, d97f9be).
Tools: MetaSplice v2.0 (out-of-fold derivation scores) and SpliceAI, on the >= 1-star derivation cohort (S6).
Configurations of the AJHG plan:
  1. frozen prior, main window rule;  2. prior 0.0441, Pejaver's reference-window rule;
  3. frozen prior, Pejaver's rule;     4. S6 main.
With the prior not promoted, 1 = 4 and 3 = 2, so two configurations are run and each is reported once:
  main        windows of >= 100 labelled and >= 25 of each class (must reproduce S6's thresholds);
  reference   Pejaver's rule: >= 100 labelled, >= 3% of the reference variants, no class minimum. The reference is
              the uniform gnomAD sample: MetaSplice's gene-matched fold scores (to match the out-of-fold derivation
              scores); SpliceAI's scores from V08b (the variants it scores).
Every configuration is validated on the >= 1-star temporal cohort (interval_lr, verdicts).
Population yield (Pejaver's check for overcalling): the share of reference variants in each candidate interval, and at
or above each PP3 level, for the deployed scores (MetaSplice's full model; SpliceAI), on all reference variants each
tool scores and on the variants both score. Flags, for investigation only, never automatic removal: share at or above a
PP3 level greater than the prior, and greater than prior / that level's posterior threshold. Flagged levels are broken
down by region (with the deep-intronic sensitivity set), allele-frequency stratum and gene. Added at Stage 3's
resumption (2026-10-01), before this script first ran: the yield in each Deep Intronic distance band (<= 50, 51-500,
> 500 bp from the exon boundary) and in the sensitivity set's bands, for every tool, configuration and basis.
Post-freeze sensitivity (AJHG Task 3.4 Step 5): the prior estimated with full-model scores on both sides, components from
the >= 1-star temporal cohort. Reported beside the frozen prior; it changes nothing.

Writes results/S11_recalibrate.txt, S11_thresholds.tsv, S11_validation_intervals.tsv and S11_population_yield.tsv.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import KEY, ROOT, Tee, deep_intronic_sensitivity
from calib import (CODES, LEVELS, PRIOR, assign, calibrate, estimate_prior, fmt_t, interval_lr, load_derivation_cohort,
                   load_thresholds, load_validation_cohort, lr_levels, mixture_fit, post_from_lr, prior_bootstrap, verdicts)

TOOLS = ["MetaSplice", "SpliceAI"]
PP3 = [f"PP3_{l}" for l in LEVELS]


def main():
    log = Tee(f"{ROOT}/results/S11_recalibrate.txt")
    log("=" * 96)
    log("S11_recalibrate -- RECALIBRATION WITH THE FROZEN PRIOR AND THE POPULATION REFERENCE (MetaSplice v2.0, SpliceAI)")
    log("=" * 96)
    frozen = json.load(open(f"{ROOT}/results/S11_prior_frozen.json"))
    prior, promoted = float(frozen["prior"]), bool(frozen["promoted"])
    c = 1124.0 if not promoted else None
    cc, need = lr_levels(prior, c)
    log(f"frozen prior {prior} (promoted: {promoted}; {frozen['reason']}); c = {cc:.0f}. The fallback 0.0441 is not a validated "
        f"population prior; yield flags below are relative to it.")
    log("EXPLORATORY: every threshold below is a candidate threshold under the fallback prior, not clinically validated.")
    configs = [("main", dict()), ("reference", dict(min_class=0, min_fraction=0.03))]
    if promoted:
        raise SystemExit("the prior was promoted: configurations 1 and 3 differ from 4 and 2; extend this script first")
    log("prior not promoted: configurations 1 = 4 (main) and 3 = 2 (reference), each reported once")

    der = load_derivation_cohort(one_star_only=True)
    val = load_validation_cohort("strict", one_star_only=True)
    ref = pd.read_csv(f"{ROOT}/data/v2_gnomad_reference_scores.csv", low_memory=False)
    ref = ref[ref["sample"] == "uniform"].reset_index(drop=True)
    sai = pd.read_csv(f"{ROOT}/data/v2_gnomad_spliceai.csv", low_memory=False)
    ref = ref.merge(sai[KEY + ["status", "spliceai_max"]], on=KEY, how="left", validate="one_to_one")
    ref["SpliceAI"] = ref.spliceai_max
    ref = ref.merge(pd.read_csv(f"{ROOT}/data/v2_features_gnomad_reference.csv", usecols=KEY + ["dist_to_boundary"]),
                    on=KEY, how="left", validate="one_to_one")
    if ref.dist_to_boundary.isna().any():
        raise ValueError("a reference variant has no distance to the exon boundary")
    window_ref = {"MetaSplice": ref.MetaSplice_fold.values, "SpliceAI": ref.SpliceAI.dropna().values}
    log(f"derivation >= 1 star: n={len(der):,} P={int(der.label.sum()):,}; validation >= 1 star: n={len(val):,} P={int(val.label.sum())}")
    log(f"reference (uniform gnomAD): {len(ref):,}; SpliceAI-scored {int(ref.SpliceAI.notna().sum()):,} "
        f"({ref.status.value_counts().to_dict()})")

    thr, thr_rows, val_rows = {}, [], []
    s6 = {t: load_thresholds(t, path=f"{ROOT}/results/S6_thresholds.tsv") for t in TOOLS}
    for t in TOOLS:
        for cfg, kw in configs:
            res = calibrate(der[t].values.astype(float), der.label.values, t, priors=((prior, c),),
                            ref=window_ref[t] if cfg == "reference" else None, **kw)
            _, _, pp3, bp4 = res["by_prior"][prior]
            thr[(t, cfg)] = (pp3, bp4)
            for lev in LEVELS:
                thr_rows += [(t, cfg, prior, "PP3_" + lev, need[lev], pp3[lev]), (t, cfg, prior, "BP4_" + lev, 1 / need[lev], bp4[lev])]
            same = cfg != "main" or all(
                (a.get(l) is None and b.get(l) is None) or (a.get(l) is not None and b.get(l) is not None and np.isclose(a[l], b[l], rtol=1e-12, atol=0))
                for a, b in [(pp3, s6[t][0]), (bp4, s6[t][1])] for l in LEVELS)
            log(f"\n{t}, {cfg} window rule: median window half-width {np.median(res['eps']):.3g}"
                + ("" if cfg != "main" else f"; equals S6's thresholds: {'yes' if same else 'NO'}"))
            if not same:
                raise SystemExit(f"{t}: the main configuration does not reproduce S6's thresholds")
            log("  PP3 " + "  ".join(f"{l} >= {fmt_t(pp3[l])}" for l in LEVELS[::-1])
                + "\n  BP4 " + "  ".join(f"{l} <= {fmt_t(bp4[l])}" for l in LEVELS))
            codes = assign(val[t].values.astype(float), pp3, bp4)
            log(f"  validation (>= 1 star): {'interval':<16}{'P':>5}{'B':>7}{'LR':>11}   exact 95% CI         needed    point  CI")
            for code, kp, kb, lr, lo, hi in interval_lr(codes, val.label.values):
                point, ci, ns = verdicts(code, lr, lo, hi, need)
                his = "inf" if np.isinf(hi) else f"{hi:.3g}"
                lrs = "inf" if np.isinf(lr) else f"{lr:.3g}"
                log(f"  {'':<24}{code:<16}{kp:>5}{kb:>7}{lrs:>11}   [{lo:.3g}, {his}]".ljust(80) + f"{ns:<10}{point:<7}{ci}")
                val_rows.append((t, cfg, code, kp, kb, lr, lo, hi, point, ci))
    pd.DataFrame(thr_rows, columns=["tool", "config", "prior", "level", "LR_needed", "score_threshold"]).to_csv(
        f"{ROOT}/results/S11_thresholds.tsv", sep="\t", index=False)
    pd.DataFrame(val_rows, columns=["tool", "config", "interval", "P", "B", "LR", "lo95_exact", "hi95_exact", "point_verdict",
                                    "ci_verdict"]).to_csv(f"{ROOT}/results/S11_validation_intervals.tsv", sep="\t", index=False)

    log("\n" + "-" * 96)
    log("POPULATION YIELD on the uniform gnomAD reference (deployed scores: MetaSplice full model; SpliceAI)")
    log(f"flags (investigation only): share at or above a PP3 level > prior {prior}; > prior / that level's posterior threshold")
    log("-" * 96)
    both = ref.SpliceAI.notna().values
    yrows, flagged = [], []
    for t in TOOLS:
        for cfg, _ in configs:
            for basis, m in [("all the tool scores", ref[t].notna().values), ("both tools score", both)]:
                s = ref[t].values[m].astype(float)
                codes = assign(s, *thr[(t, cfg)])
                shares = pd.Series(codes).value_counts(normalize=True).reindex(CODES, fill_value=0.0)
                log(f"\n  {t}, {cfg}, reference variants {basis} (n={m.sum():,}):")
                log("    interval shares: " + "  ".join(f"{k} {v:.2%}" for k, v in shares.items() if v > 0))
                for i, lev in enumerate(LEVELS):
                    at_or_above = float(np.isin(codes, PP3[i:]).mean())
                    bound = prior / post_from_lr(need[lev], prior)
                    flags = [f for f, hit in [("above prior", at_or_above > prior), ("above prior/posterior", at_or_above > bound)] if hit]
                    yrows.append(dict(tool=t, config=cfg, basis=basis, n=int(m.sum()), level=f"PP3 >= {lev}", share=at_or_above,
                                      prior=prior, bound=bound, flags="; ".join(flags)))
                    if thr[(t, cfg)][0].get(lev) is None:
                        continue
                    log(f"    PP3 >= {lev:<11} share {at_or_above:7.3%}   prior {prior:.2%}   prior/posterior {bound:7.2%}"
                        + (f"   FLAG: {', '.join(flags)}" if flags else ""))
                    if flags and basis == "all the tool scores":
                        flagged.append((t, cfg, lev, m, codes, i))
                d = ref[m]
                dist = np.abs(d.dist_to_boundary.values)
                groups = []
                for set_name, sel in [("Deep Intronic", (d.region == "Deep Intronic").values),
                                      ("DI sensitivity set", deep_intronic_sensitivity(d).values)]:
                    for b, lo, hi in [("<= 50 bp", 0, 50), ("51-500 bp", 50, 500), ("> 500 bp", 500, np.inf)]:
                        groups.append((f"{set_name} {b}", sel & (dist > lo) & (dist <= hi)))
                log("    by Deep Intronic distance band, share at or above each PP3 level reached:")
                for g, gm in groups:
                    cells = []
                    for i, lev in enumerate(LEVELS):
                        if thr[(t, cfg)][0].get(lev) is None or not gm.any():
                            continue
                        sh = float(np.isin(codes[gm], PP3[i:]).mean())
                        cells.append(f"{lev} {sh:.3%}")
                        yrows.append(dict(tool=t, config=cfg, basis=basis, n=int(gm.sum()), level=f"PP3 >= {lev}", share=sh,
                                          prior=prior, bound=prior / post_from_lr(need[lev], prior), flags="", group=g))
                    log(f"      {g:<30} n={int(gm.sum()):>7,}  " + "  ".join(cells))
                for lev in LEVELS:
                    yrows.append(dict(tool=t, config=cfg, basis=basis, n=int(m.sum()), level=f"BP4_{lev}",
                                      share=float((codes == f"BP4_{lev}").mean()), prior=prior, bound=np.nan, flags=""))
    pd.DataFrame(yrows).to_csv(f"{ROOT}/results/S11_population_yield.tsv", sep="\t", index=False)

    log("\n" + "-" * 96)
    log("FLAGGED LEVELS, by region, allele-frequency stratum and gene (investigation; no automatic removal)")
    log("-" * 96)
    if not flagged:
        log("  none")
    for t, cfg, lev, m, codes, i in flagged:
        d = ref[m].reset_index(drop=True)
        hit = np.isin(codes, PP3[i:])
        log(f"\n  {t}, {cfg}, PP3 >= {lev}: {int(hit.sum()):,} of {len(d):,} reference variants")
        groups = [(r, (d.region == r).values) for r in ["Donor Region", "Acceptor Region", "PPT Region", "Branch Point Region", "Deep Intronic"]]
        groups.append(("Deep Intronic, sensitivity set", deep_intronic_sensitivity(d).values))
        for name, g in groups:
            if g.sum():
                log(f"    {name:<32} n={int(g.sum()):>7,}  share {hit[g].mean():7.3%}")
        for name, g in [("AF < 0.001", (d.gnomad_af < 0.001).values), ("0.001 <= AF < 0.01", (d.gnomad_af >= 0.001).values)]:
            log(f"    {name:<32} n={int(g.sum()):>7,}  share {hit[g].mean():7.3%}")
        top = d[hit].gene.value_counts().head(10)
        log(f"    genes with the most flagged variants: " + ", ".join(f"{g} {n}" for g, n in top.items())
            + f"; {d[hit].gene.nunique():,} genes in all")

    log("\n" + "-" * 96)
    log("POST-FREEZE SENSITIVITY OF THE PRIOR (reported only; the frozen prior and every threshold are unchanged)")
    log("-" * 96)
    s_p, s_b, r = val.MetaSplice.values[val.label.values == 1], val.MetaSplice.values[val.label.values == 0], ref.MetaSplice.values
    est = estimate_prior(r, s_p, s_b, 50)
    lo, hi = prior_bootstrap(r, s_p, s_b, 50, n_boot=1000, seed=42)
    log(f"  components: >= 1-star temporal cohort, full model (P/LP {len(s_p)}, B/LB {len(s_b):,}); reference: uniform gnomAD, full model")
    log(f"  estimate {est:.4f} [{lo:.4f}, {hi:.4f}], fit {mixture_fit(r, s_p, s_b, est, 50):.4f}; frozen prior {prior} "
        f"(primary estimate {frozen['estimate']:.4f}, fit {frozen['fit']:.2f}, not promoted)")
    log("\nwrote results/S11_thresholds.tsv, S11_validation_intervals.tsv, S11_population_yield.tsv")
    log.close()


if __name__ == "__main__":
    main()
