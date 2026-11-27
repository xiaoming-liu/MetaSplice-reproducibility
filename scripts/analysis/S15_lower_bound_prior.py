"""S15 -- Calibration of MetaSplice and SpliceAI at a lower bound on the prior probability of pathogenicity for rare
non-canonical intronic variants, and the resulting PP3/BP4 rule on the pooled temporal test set.

1. Lower bound. The population is every rare intronic SNV that gnomAD v4.1 (joint) observes in the calibration's scope:
   FILTER PASS, 0 < AF < 0.01, in one of the five non-canonical regions by the scorer's scope(), in a gene with >= 1 P/LP
   SNV in the training release (the genes of the calibration cohort). Its size is V02b's eligible count (the population
   less its ClinVar variants) plus the eligible ClinVar variants, which S15a classifies with V02b's code on the same
   files (data/v2_gnomad_eligible_clinvar.csv). The numerator is the population's variants that are P/LP with >= 1
   review star in the training release (2025-08-31), which S15a found to be exactly the >= 1-star P/LP of the training
   calibration cohort with 0 < AF < 0.01; this script checks the cohort's variants against the file again. Pathogenic
   variants that are not ClinVar P/LP with >= 1 star in the training release are not counted, so the share is a lower
   bound on the share of pathogenic variants in the population. The later release is not used, so that the prior, like
   the thresholds, rests on training data only.
   Sensitivity (reported only), same population: all review statuses, >= 2 stars, AF < 1e-4 (numerator variants rare
   enough to be unlikely misclassifications), and each region on its own (the region's numerator over its population).
2. Thresholds at the lower bound (calib.calibrate: windows of >= 100 labelled variants and >= 25 of each class, 10,000
   bootstrap iterations): MetaSplice (out-of-fold) and SpliceAI (precomputed masked) on the derivation cohort, >= 1 star
   (main) and all review statuses. The requirements follow calib.lr_levels(prior) (c from the prior).
3. Pooled temporal test set, calibration cohort (>= 1 star and all review statuses) and its resolved-VUS subgroup
   (uncertain or conflicting in the training release): the derived intervals with exact 95% CIs and whether each meets
   the lower-bound requirement; the Supporting rule (PP3 at Supporting for MetaSplice >= the PP3 Supporting threshold,
   BP4 at Supporting for <= the BP4 Supporting threshold, >= 1 star thresholds; no other level) against ClinGen's
   SpliceAI rule (>= 0.2 / <= 0.1; Walker et al. 2023) with raw SpliceAI scores (first record, and all genes) and
   precomputed masked scores, with paired bootstrap differences (class-stratified and gene-clustered, 1,000
   iterations, seed 42); the remaining VUS (uncertain or conflicting on 2026-04-15, not P/LP or B/LB in the pooled set):
   MetaSplice score shares and the codes of each rule.
4. Population: the share of the uniform gnomAD reference given each code (gene-matched fold score and full model), the
   rule's likelihood ratios with the reference as the benign denominator, and the rule within each region against the
   region's own bound.
Inputs (md5) and the cohort denominators are checked first. Needs data/v2_gnomad_eligible_clinvar.csv from
S15a_prior_population.py.
Writes results/S15_lower_bound_prior.txt, S15_thresholds.tsv and S15_pooled.tsv.
"""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
from common import CLINVAR_2025, OUT, Tee
from calib import (ZERO_STAR, LEVELS, assign, calibrate, in_cohort, interval_lr, load_derivation_cohort, lr_ci,
                   lr_levels, pathogenic_genes, thresholds, verdicts, was_vus_2025, fmt_t)
from paper_cohorts import load_vus_cohort, pooled_cohort, verify_locks

V02B = f"{OUT}/results/V02b_gnomad_reference.txt"
S15A = f"{OUT}/data/v2_gnomad_eligible_clinvar.csv"
TWO_STAR = {"criteria_provided,_multiple_submitters,_no_conflicts", "reviewed_by_expert_panel", "practice_guideline"}
REGIONS = ["Donor Region", "Acceptor Region", "PPT Region", "Branch Point Region", "Deep Intronic"]
RULES = ["MetaSplice", "SpliceAI raw, first record", "SpliceAI raw, all genes", "SpliceAI masked"]
KEY = ["chr", "pos", "ref", "alt"]
SAI_PP3, SAI_BP4 = 0.2, 0.1                                     # ClinGen's SpliceAI thresholds (Walker et al. 2023)
SHARES = [("P/LP given PP3", 1, "PP3"), ("B/LB given PP3", 0, "PP3"), ("B/LB given BP4", 0, "BP4"),
          ("P/LP given BP4", 1, "BP4")]
# raw SpliceAI scores (V11, V11b, V11c); paper_cohorts.LOCKS holds the md5 of the first two, TOPUP_MD5 the third's
RAW = ["data/v2_spliceai_raw_d4999.csv", "data/v2_spliceai_raw_d4999_vus.csv", "data/v2_spliceai_raw_d4999_topup.csv"]
TOPUP_MD5 = "52d94eb91cef1ec323409e33d7b36906"
LOCK_FILES = ["models/metasplice_v2_full.joblib", "models/metasplice_v2_features.joblib", "data/v2_features_pooled_test.csv",
              "data/v2_spliceai_raw_d4999.csv", "data/v2_spliceai_raw_d4999_vus.csv", "data/v2_features_vus.csv",
              "data/v2_comparators.csv", "data/v2_pathogenic_genes.csv", "data/v2_variants_training.csv",
              CLINVAR_2025["GRCh38"], CLINVAR_2025["GRCh37"]]
RECORDED = {"calibration cohort, >= 1 star": (37841, 578), "calibration cohort, all statuses": (37860, 595),
            "resolved VUS, >= 1 star": (859, 153), "resolved VUS, all statuses": (861, 153)}     # (n, P/LP)
N_REMAINING_VUS = 60776


def add_raw(d, raw):
    return d.merge(raw, on=KEY, how="left", validate="one_to_one")


def boot_diffs(y, a, b, genes, n_boot=1000, seed=42):
    """Bootstrap percentile CIs for share(a) - share(b) within class y==cls, for both classes at once.

    a, b: boolean arrays (code given by each rule). Returns {cls: ((lo, hi) stratified, (lo, hi) gene-clustered)}."""
    rng = np.random.default_rng(seed)
    out = {}
    for cls in (1, 0):
        idx = np.where(y == cls)[0]
        da = a[idx].astype(float) - b[idx].astype(float)
        diffs = [da[rng.choice(len(idx), len(idx), replace=True)].mean() for _ in range(n_boot)]
        strat = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
        out[cls] = [strat]
    # gene-clustered: resample genes; shares within each class over the resampled variants
    g_codes, g_idx = np.unique(genes, return_inverse=True)
    G = len(g_codes)
    sums = {}
    for cls in (1, 0):
        m = (y == cls)
        sums[cls] = (np.bincount(g_idx[m], minlength=G).astype(float),
                     np.bincount(g_idx[m], weights=a[m].astype(float), minlength=G),
                     np.bincount(g_idx[m], weights=b[m].astype(float), minlength=G))
    rng = np.random.default_rng(seed)
    gd = {1: [], 0: []}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
        for cls in (1, 0):
            n, sa, sb = sums[cls]
            tot = (w * n).sum()
            gd[cls].append(((w * sa).sum() - (w * sb).sum()) / tot if tot else np.nan)
    for cls in (1, 0):
        v = np.array(gd[cls])
        out[cls].append((float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))))
    return out


def v02b_counts():
    txt = open(V02B).read()
    eligible = int(re.search(r"^eligible ([\d,]+);", txt, re.M).group(1).replace(",", ""))
    clinvar = int(re.search(r"in a ClinVar release ([\d,]+)", txt).group(1).replace(",", ""))
    by_region = {r: int(n.replace(",", "")) for r, n in re.findall(r"(Deep Intronic|Branch Point Region|PPT Region|"
                                                                 r"Acceptor Region|Donor Region) ([\d,]+)", txt.split("by region:")[1].split("\n")[0])}
    assert sum(by_region.values()) == eligible, (by_region, eligible)
    return eligible, clinvar, by_region


def _key(d):
    return d.chr.astype(str) + ":" + d.pos.astype(int).astype(str) + ":" + d.ref + ":" + d.alt


def lower_bound(log, der_all):
    eligible, _, by_region = v02b_counts()
    cv = pd.read_csv(S15A, dtype={"chr": str, "ref": str, "alt": str}, low_memory=False)
    cv["key"] = _key(cv)
    if cv.key.duplicated().any() or (cv.gnomad_af <= 0).any() or (cv.gnomad_af >= 0.01).any() or not cv.region.isin(REGIONS).all():
        raise SystemExit(f"{S15A}: duplicate variants or a variant outside the population")
    pop = {r: by_region[r] + int((cv.region == r).sum()) for r in REGIONS}
    N = sum(pop.values())
    pos = der_all[der_all.label == 1].copy()
    pos["key"] = _key(pos)
    rare = (pos.gnomad_af > 0) & (pos.gnomad_af < 0.01)
    one = ~pos.review_status.isin(ZERO_STAR)
    # every cohort P/LP in the AF window must be a member of the population, with the same region, gene and AF
    m = pos[rare].merge(cv[["key", "region", "gene_id", "gnomad_af"]], on="key", how="left", suffixes=("", "_pop"))
    member = m.region_pop.notna() & (m.region == m.region_pop) & (m.gene_id == m.gene_id_pop) & np.isclose(
        m.gnomad_af.astype(float), m.gnomad_af_pop.astype(float), rtol=0, atol=1e-12)
    if not member.all():
        raise SystemExit(f"{int((~member).sum())} cohort P/LP variants with 0 < AF < 0.01 are not population members")
    seen = pos[rare & one]
    if set(seen.key) != set(cv.key[cv.plp_2025_one_star]):
        raise SystemExit("the numerator differs from S15a's (P/LP with >= 1 star in the training release)")
    lb = len(seen) / N
    log("1. LOWER BOUND ON THE PRIOR")
    log("  population (gnomAD v4.1 joint, FILTER PASS, 0 < AF < 0.01, non-canonical regions, genes of the calibration cohort):")
    for r in REGIONS:
        log(f"    {r:<22}{by_region[r]:>13,} not in ClinVar + {pop[r] - by_region[r]:>7,} in ClinVar = {pop[r]:>13,}")
    log(f"    {'all five regions':<22}{eligible:>13,} not in ClinVar + {N - eligible:>7,} in ClinVar = {N:>13,}")
    log(f"  P/LP (>= 1 star) of the training calibration cohort: {int(one.sum()):,}; with 0 < gnomAD AF < 0.01: {len(seen):,} "
        f"(all members of the population with the same region, gene and AF; equal to S15a's direct count from the release)")
    log(f"  lower bound = {len(seen):,} / {N:,} = {lb:.4e}  ({lb!r})")
    log("  sensitivity (reported only; same population):")
    rows = [("all review statuses", int(rare.sum())), (">= 2 stars", int((rare & pos.review_status.isin(TWO_STAR)).sum())),
            (">= 1 star, AF < 1e-4", int((rare & one & (pos.gnomad_af < 1e-4)).sum()))]
    out = []
    for name, n in rows:
        b = n / N
        out.append(dict(section="lower bound sensitivity", name=name, numerator=n, denominator=N, bound=b))
        log(f"    {name:<28} {n:>6,} / {N:>11,} = {b:.3e}   Supporting {lr_levels(b)[1]['Supporting']:.2f}")
    log("  by region (the region's numerator over the region's population):")
    region_lb = {}
    for r in REGIONS:
        n = int((seen.region == r).sum())
        region_lb[r] = n / pop[r]
        out.append(dict(section="lower bound by region", name=r, numerator=n, denominator=pop[r], bound=region_lb[r]))
        log(f"    {r:<22} {n:>5,} / {pop[r]:>11,} = {region_lb[r]:.2e}")
    out.insert(0, dict(section="lower bound", name="main", numerator=len(seen), denominator=N, bound=lb))
    return lb, region_lb, out


def rule_codes(d, t_pp3, t_bp4):
    def code(s, hi, lo):
        c = np.full(len(s), "", dtype=object)
        c[s >= hi] = "PP3"          # NaN compares False: an unscored variant gets no code
        c[s <= lo] = "BP4"
        return c
    return {"MetaSplice": code(d.MetaSplice.values.astype(float), t_pp3, t_bp4),
            "SpliceAI raw, first record": code(d.spliceai_raw_max.values.astype(float), SAI_PP3, SAI_BP4),
            "SpliceAI raw, all genes": code(d.spliceai_raw_max_allgenes.values.astype(float), SAI_PP3, SAI_BP4),
            "SpliceAI masked": code(d.SpliceAI.values.astype(float), SAI_PP3, SAI_BP4)}


def rule_report(log, name, d, tsv, t_pp3, t_bp4, need):
    y = d.label.values
    NP, NB = int(y.sum()), int((y == 0).sum())
    c = rule_codes(d, t_pp3, t_bp4)
    req = need["Supporting"]
    log(f"\n  {name}: n={len(d):,} (P/LP {NP}, B/LB {NB:,}); SpliceAI raw scores {int(d.spliceai_raw_max.notna().sum()):,}")
    log(f"    {'rule':<28}{'P/LP PP3':>16}{'B/LB PP3':>17}{'B/LB BP4':>18}{'P/LP BP4':>15}{'P/LP none':>14}{'B/LB none':>16}")
    for rule in RULES:
        k = {(cls, x): int(((c[rule] == x) & (y == cls)).sum()) for cls in (1, 0) for x in ("PP3", "BP4", "")}
        sh = lambda cls, x: k[(cls, x)] / (NP if cls else NB)
        log(f"    {rule:<28}{k[(1,'PP3')]:>6} ({sh(1,'PP3'):6.1%}){k[(0,'PP3')]:>7,} ({sh(0,'PP3'):5.2%})"
            f"{k[(0,'BP4')]:>8,} ({sh(0,'BP4'):6.1%}){k[(1,'BP4')]:>5} ({sh(1,'BP4'):5.1%})"
            f"{k[(1,'')]:>5} ({sh(1,''):5.1%}){k[(0,'')]:>7,} ({sh(0,''):5.1%})")
        for x in ("PP3", "BP4"):
            lr, lo, hi = lr_ci(k[(1, x)], NP, k[(0, x)], NB)
            clears = (lo >= req) if x == "PP3" else (hi <= 1 / req)
            his = "inf" if hi == np.inf else f"{hi:.3g}"
            log(f"        {x} variants: LR {lr:.3g} [{lo:.3g}, {his}]   Supporting requirement "
                f"{'>= ' + format(req, '.3g') if x == 'PP3' else '<= ' + format(1 / req, '.3g')}: {'clears' if clears else 'INCONCLUSIVE'}")
            tsv.append(dict(section="rule", cohort=name, rule=rule, code=x, P=k[(1, x)], B=k[(0, x)], NP=NP, NB=NB,
                            LR=lr, lo95_exact=lo, hi95_exact=hi, ci="clears" if clears else "inconclusive"))
        tsv.append(dict(section="rule", cohort=name, rule=rule, code="neither", P=k[(1, "")], B=k[(0, "")], NP=NP, NB=NB))
    for comp in RULES[1:]:
        log(f"    MetaSplice minus {comp}: share difference [stratified 95% CI] [gene-clustered 95% CI]")
        for label, cls, x in SHARES:
            a, b = (c["MetaSplice"] == x), (c[comp] == x)
            diff = a[y == cls].mean() - b[y == cls].mean()
            (slo, shi), (glo, ghi) = boot_diffs(y, a, b, d.gene_id.values)[cls]
            log(f"        {label:<16}{diff:+8.2%}  [{slo:+.2%}, {shi:+.2%}]  [{glo:+.2%}, {ghi:+.2%}]")
            tsv.append(dict(section="rule difference", cohort=name, rule=f"MetaSplice minus {comp}", code=label, diff=diff,
                            lo=slo, hi=shi, gene_lo=glo, gene_hi=ghi))


def main():
    log = Tee(f"{OUT}/results/S15_lower_bound_prior.txt")
    log("=" * 104)
    log("S15 -- CALIBRATION AT THE LOWER BOUND ON THE INTRONIC PRIOR (MetaSplice v2.0, SpliceAI)")
    log("=" * 104)
    verify_locks(log, LOCK_FILES, extra={RAW[2]: TOPUP_MD5})
    der_all = load_derivation_cohort(one_star_only=False)
    der_one = der_all[~der_all.review_status.isin(ZERO_STAR)].reset_index(drop=True)
    tsv = []
    lb, region_lb, rows = lower_bound(log, der_all)
    tsv += rows
    c_lb, need = lr_levels(lb)
    log("\n  likelihood ratio required at the lower bound (c = %.4g): " % c_lb
        + ", ".join(f"{k} {v:.4g}" for k, v in need.items()) + "; BP4: the reciprocals")
    for p in (0.1, 0.0441, 0.01, 1e-3, 1e-4, lb):
        s = lr_levels(p)[1]["Supporting"]
        log(f"    prior {p:<10.4g} Supporting {s:6.3f}   reciprocal {1 / s:.4f}")

    log("\n2. THRESHOLDS AT THE LOWER BOUND (derivation cohort; windows of >= 100 labelled variants and >= 25 of each class, "
        "10,000 bootstrap iterations)")
    thr, trows = {}, []
    for cohort, der in [(">= 1 star", der_one), ("all statuses", der_all)]:
        for t in ("MetaSplice", "SpliceAI"):
            res = calibrate(der[t].values.astype(float), der.label.values, t, priors=((lb, None),))
            _, _, pp3, bp4 = res["by_prior"][lb]
            thr[(t, cohort)] = (pp3, bp4)
            log(f"  {t}, {cohort} (n={res['n']:,}, P/LP {res['P']:,})")
            log("    PP3 " + "  ".join(f"{l} >= {fmt_t(pp3[l])}" for l in LEVELS[::-1])
                + "\n    BP4 " + "  ".join(f"{l} <= {fmt_t(bp4[l])}" for l in LEVELS))
            for l in LEVELS:
                trows += [(t, cohort, lb, "PP3_" + l, need[l], pp3[l]), (t, cohort, lb, "BP4_" + l, 1 / need[l], bp4[l])]
            if (t, cohort) == ("MetaSplice", ">= 1 star"):
                log("    Supporting thresholds at the sensitivity bounds (same bootstrap):")
                for r in rows:
                    if r["section"] != "lower bound sensitivity":
                        continue
                    nd = lr_levels(r["bound"])[1]
                    sp, sb = thresholds(res["grid"], res["lower"], res["upper"], nd)
                    log(f"      {r['name']:<52} bound {r['bound']:.3e}: PP3 >= {fmt_t(sp['Supporting'])}, BP4 <= {fmt_t(sb['Supporting'])}")
                    tsv.append(dict(section="sensitivity thresholds", name=r["name"], bound=r["bound"],
                                    pp3=sp["Supporting"], bp4=sb["Supporting"]))
    pd.DataFrame(trows, columns=["tool", "cohort", "prior", "level", "LR_needed", "score_threshold"]).to_csv(
        f"{OUT}/results/S15_thresholds.tsv", sep="\t", index=False)
    t_pp3, t_bp4 = thr[("MetaSplice", ">= 1 star")][0]["Supporting"], thr[("MetaSplice", ">= 1 star")][1]["Supporting"]
    log(f"\n  Supporting rule: PP3 for MetaSplice >= {t_pp3!r}, BP4 for <= {t_bp4!r} (>= 1 star thresholds); no other level")

    log("\n3. POOLED TEMPORAL TEST SET")
    p = pooled_cohort()
    pg = pathogenic_genes()
    cal = p[in_cohort(p, pg)].reset_index(drop=True)
    one = cal[~cal.review_status.isin(ZERO_STAR)].reset_index(drop=True)
    cohorts = {"calibration cohort, >= 1 star": one, "calibration cohort, all statuses": cal,
               "resolved VUS, >= 1 star": one[was_vus_2025(one).values].reset_index(drop=True),
               "resolved VUS, all statuses": cal[was_vus_2025(cal).values].reset_index(drop=True)}
    for name, w1 in RECORDED.items():
        g1 = (len(cohorts[name]), int(cohorts[name].label.sum()))
        if g1 != w1:
            raise SystemExit(f"{name}: {g1}, recorded {w1}")
    log("  denominators equal the recorded values: " + "; ".join(f"{k} {len(v):,} ({int(v.label.sum())} P/LP)" for k, v in cohorts.items()))
    raw = pd.concat([pd.read_csv(f"{OUT}/{f}", usecols=KEY + ["spliceai_raw_max", "spliceai_raw_max_allgenes"]) for f in RAW])
    dup = raw[raw.duplicated(KEY, keep=False)]
    if len(dup) and (dup.groupby(KEY)[["spliceai_raw_max", "spliceai_raw_max_allgenes"]].nunique(dropna=False) > 1).any().any():
        raise ValueError("a variant has different raw scores in two raw-score files")
    raw = raw.drop_duplicates(KEY)
    cohorts = {k: add_raw(v, raw) for k, v in cohorts.items()}

    log("\n  3a. Derived intervals (thresholds of the matching review-status cohort; exact 95% CI; requirement at the lower bound)")
    for name, tc in [("calibration cohort, >= 1 star", ">= 1 star"), ("calibration cohort, all statuses", "all statuses")]:
        d = cohorts[name]
        for tool, col in [("MetaSplice", "MetaSplice"), ("SpliceAI (precomputed masked)", "SpliceAI")]:
            th = thr[("MetaSplice" if tool == "MetaSplice" else "SpliceAI", tc)]
            codes = assign(d[col].values.astype(float), *th)
            log(f"\n    {name} -- {tool}")
            for cd, kp, kb, lr, lo, hi in interval_lr(codes, d.label.values):
                point, ci, ns = verdicts(cd, lr, lo, hi, need)
                his = "inf" if hi == np.inf else f"{hi:.3g}"
                log(f"      {cd:<16}{kp:>5}{kb:>8,}{lr:>11.3g}   [{lo:.3g}, {his}]".ljust(66) + f"{ns:<11}{point:<7}{ci}")
                tsv.append(dict(section="levels", cohort=name, rule=tool, code=cd, P=kp, B=kb, LR=lr, lo95_exact=lo,
                                hi95_exact=hi, point=point, ci=ci))

    d = cohorts["calibration cohort, >= 1 star"]
    exact = rule_codes(d, t_pp3, t_bp4)["MetaSplice"]
    rounded = rule_codes(d, 0.182, 0.00071)["MetaSplice"]
    n_ch = int((exact != rounded).sum())
    log(f"\n  rounded cutoffs (>= 0.182 / <= 0.00071) change {n_ch} of {len(d):,} assignments in the calibration cohort, >= 1 star "
        f"(PP3 {int(((exact == 'PP3') != (rounded == 'PP3')).sum())}, BP4 {int(((exact == 'BP4') != (rounded == 'BP4')).sum())})")
    tsv.append(dict(section="rounding", cohort="calibration cohort, >= 1 star", count=n_ch))
    log("\n  3b. The Supporting rule against SpliceAI's rule")
    for name in cohorts:
        rule_report(log, name, cohorts[name], tsv, t_pp3, t_bp4, need)

    vus = load_vus_cohort(pg)
    pk = set(map(tuple, p[KEY].astype(str).values))
    rem = vus[[tuple(x) not in pk for x in vus[KEY].astype(str).values]].reset_index(drop=True)
    if len(rem) != N_REMAINING_VUS:
        raise SystemExit(f"remaining VUS {len(rem):,}; recorded {N_REMAINING_VUS:,}")
    rem = add_raw(rem, raw)
    log(f"\n  3c. Remaining VUS (n={len(rem):,}; no labels)")
    s = rem.MetaSplice.values
    log(f"    MetaSplice >= 0.1 {np.mean(s >= 0.1):6.1%}   >= 0.2 {np.mean(s >= 0.2):6.1%}   >= 0.5 {np.mean(s >= 0.5):6.1%}")
    tsv += [dict(section="remaining VUS shares", threshold=t, share=float(np.mean(s >= t)), n=len(rem)) for t in (0.1, 0.2, 0.5)]
    for rule, cc in rule_codes(rem, t_pp3, t_bp4).items():
        k = {x: int((cc == x).sum()) for x in ("PP3", "BP4", "")}
        log(f"    {rule:<28} PP3 {k['PP3']:>7,} ({k['PP3'] / len(rem):5.1%})   BP4 {k['BP4']:>7,} ({k['BP4'] / len(rem):5.1%})   "
            f"neither {k['']:>7,} ({k[''] / len(rem):5.1%})")
        tsv += [dict(section="remaining VUS codes", rule=rule, code=x or "neither", count=v, share=v / len(rem)) for x, v in k.items()]

    log("\n4. POPULATION (uniform gnomAD reference, n = 100,000)")
    ref = pd.read_csv(f"{OUT}/data/v2_gnomad_reference_scores.csv", low_memory=False)
    ref = ref[ref["sample"] == "uniform"].reset_index(drop=True)
    for col, lab in [("MetaSplice_fold", "gene-matched fold score"), ("MetaSplice", "full model")]:
        u = ref[col].values.astype(float)
        a, b = float(np.mean(u >= t_pp3)), float(np.mean(u <= t_bp4))
        log(f"  {lab:<24} PP3 {a:.3%}   BP4 {b:.2%}")
        tsv += [dict(section="population", rule=lab, code="PP3", share=a), dict(section="population", rule=lab, code="BP4", share=b)]
    d = cohorts["calibration cohort, >= 1 star"]
    y, s, u = d.label.values, d.MetaSplice.values, ref.MetaSplice_fold.values.astype(float)
    NP = int(y.sum())
    log("  the rule with the reference as the benign denominator (P/LP: the pooled calibration cohort, >= 1 star):")
    for x, kp, kb in [("PP3", int((s[y == 1] >= t_pp3).sum()), int((u >= t_pp3).sum())),
                      ("BP4", int((s[y == 1] <= t_bp4).sum()), int((u <= t_bp4).sum()))]:
        lr, lo, hi = lr_ci(kp, NP, kb, len(u))
        clears = lo >= need["Supporting"] if x == "PP3" else hi <= 1 / need["Supporting"]
        log(f"    {x}: P/LP {kp}/{NP}, reference {kb:,}/{len(u):,}: LR {lr:.3g} [{lo:.3g}, {hi:.3g}]  {'clears' if clears else 'INCONCLUSIVE'}")
        tsv.append(dict(section="gnomAD benign reference", rule="MetaSplice", code=x, P=kp, B=kb, NP=NP, NB=len(u), LR=lr,
                        lo95_exact=lo, hi95_exact=hi, ci="clears" if clears else "inconclusive"))
    log("  the rule by region, against each region's own bound (pooled calibration cohort, >= 1 star):")
    for r in REGIONS:
        m = (d.region == r).values
        nd = lr_levels(region_lb[r])[1]["Supporting"]
        np_, nb = int(y[m].sum()), int((y[m] == 0).sum())
        cells = []
        for x, sel in [("PP3", s >= t_pp3), ("BP4", s <= t_bp4)]:
            kp, kb = int((sel & m & (y == 1)).sum()), int((sel & m & (y == 0)).sum())
            lr, lo, hi = lr_ci(kp, np_, kb, nb)
            clears = lo >= nd if x == "PP3" else hi <= 1 / nd
            cells.append(f"{x} {lr:.3g} [{lo:.3g}, {hi:.3g}] {'clears' if clears else 'INCONCLUSIVE'}")
            tsv.append(dict(section="region", cohort=r, rule="MetaSplice", code=x, P=kp, B=kb, NP=np_, NB=nb, LR=lr,
                            lo95_exact=lo, hi95_exact=hi, ci="clears" if clears else "inconclusive", bound=region_lb[r]))
        log(f"    {r:<20} P/LP {np_:>4} B/LB {nb:>6,}  bound {region_lb[r]:.2e} (Supporting {nd:.2f})   " + "   ".join(cells))
    pd.DataFrame(tsv).to_csv(f"{OUT}/results/S15_pooled.tsv", sep="\t", index=False)
    log("\nwrote results/S15_lower_bound_prior.txt, S15_thresholds.tsv and S15_pooled.tsv")
    log.close()


if __name__ == "__main__":
    main()
