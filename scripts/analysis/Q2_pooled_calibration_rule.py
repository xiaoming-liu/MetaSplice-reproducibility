"""Q2 -- Evidence intervals and the Supporting rule on the pooled temporal test set (protocol Amendment 9; a pooled
reassessment, descriptive, no verdict; thresholds and the fallback prior 0.0441 remain exploratory).

  1. S6 levels, pooled reassessment: S6's candidate levels for MetaSplice and SpliceAI (precomputed masked) on the pooled
     calibration cohort (>= 1 star primary; all review statuses secondary): interval LRs with exact 95% CIs and whether
     each point estimate and each CI meets the requirement at prior 0.0441. Pooling cannot reinstate the BP4 Strong level
     dropped at Gate G3; its interval is reported as measured.
  2. The Amendment 6 rule (PP3 at Supporting for score >= 0.544; BP4 at Supporting for score <= 0.0017) against SpliceAI's
     rule (>= 0.2 / <= 0.1; Walker et al. 2023), with X3's measures. Primary comparator: SpliceAI 1.3.1 raw over +-4,999 nt,
     first record (V11, V11b, V11c); secondary: the all-genes maximum and the precomputed masked score. Unscored variants
     receive no code. Paired differences: X3's boot_diffs (stratified and gene-clustered, 1,000 iterations, seed 42).
  3. Resolved-VUS subgroup: pooled calibration-cohort variants uncertain or conflicting in the 2025-08-31 release.
  4. Remaining VUS: S5's VUS cohort less the variants classified P/LP or B/LB by 2026-09-28. MetaSplice score shares and
     codes under each rule.
Calibration intervals keep their counts, LRs and exact CIs however few P/LP they contain. Inputs and denominators are
asserted first; --check-only stops there.

Writes results/Q2_pooled_calibration_rule.txt and .tsv.
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
from common import CLINVAR_2025, OUT, Tee
from calib import (ZERO_STAR, assign, in_cohort, interval_lr, load_thresholds, lr_ci, pathogenic_genes, verdicts,
                   was_vus_2025)
from paper_cohorts import load_vus_cohort, pooled_cohort, verify_locks
from X3_supporting_rule import PRIORS, SAI_BP4, SAI_PP3, SHARES, T_BP4, T_PP3, boot_diffs

S6 = f"{OUT}/results/S6_thresholds.tsv"
KEY = ["chr", "pos", "ref", "alt"]
RAW = ["data/v2_spliceai_raw_d4999.csv", "data/v2_spliceai_raw_d4999_vus.csv", "data/v2_spliceai_raw_d4999_topup.csv"]
TOPUP_MD5 = "52d94eb91cef1ec323409e33d7b36906"     # Amendment 9: added after V11c writes the file, before Q2 computes anything
LOCK_FILES = ["models/metasplice_v2_full.joblib", "models/metasplice_v2_features.joblib", "data/v2_features_pooled_test.csv",
              "results/S6_thresholds.tsv", "data/v2_spliceai_raw_d4999.csv", "data/v2_spliceai_raw_d4999_vus.csv",
              "data/v2_features_vus.csv", "data/v2_comparators.csv", "data/v2_pathogenic_genes.csv",
              "data/v2_variants_training.csv", CLINVAR_2025["GRCh38"], CLINVAR_2025["GRCh37"]]
EXPECTED = {   # Amendment 9: (n, P/LP) and SpliceAI-scored (n, P/LP) where recorded
    "calibration cohort, >= 1 star": ((37841, 578), (37537, 576)),
    "calibration cohort, all statuses": ((37860, 595), (37556, 593)),
    "resolved VUS, >= 1 star": ((859, 153), None), "resolved VUS, all statuses": ((861, 153), None),
}
RULES = ["MetaSplice", "SpliceAI raw, first record", "SpliceAI raw, all genes", "SpliceAI masked"]


def code(s, hi, lo):
    c = np.full(len(s), "", dtype=object)
    c[s >= hi] = "PP3"          # NaN compares False: an unscored variant gets no code
    c[s <= lo] = "BP4"
    return c


def rule_codes(d):
    return {"MetaSplice": code(d.MetaSplice.values.astype(float), T_PP3, T_BP4),
            "SpliceAI raw, first record": code(d.spliceai_raw_max.values.astype(float), SAI_PP3, SAI_BP4),
            "SpliceAI raw, all genes": code(d.spliceai_raw_max_allgenes.values.astype(float), SAI_PP3, SAI_BP4),
            "SpliceAI masked": code(d.SpliceAI.values.astype(float), SAI_PP3, SAI_BP4)}


def add_raw(d, raw):
    m = d.merge(raw, on=KEY, how="left", validate="one_to_one")
    return m


def rule_report(log, name, d, tsv):
    y = d.label.values
    NP, NB = int(y.sum()), int((y == 0).sum())
    c = rule_codes(d)
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
            meets = []
            for pname, need in PRIORS:
                req = need["Supporting"]
                meets.append(f"prior {pname}: " + ("clears" if (lo >= req if x == "PP3" else hi <= 1 / req) else "INCONCLUSIVE"))
            his = "inf" if hi == np.inf else f"{hi:.3g}"
            log(f"        {x} variants: LR {lr:.3g} [{lo:.3g}, {his}]   Supporting CI: " + "; ".join(meets))
            tsv.append(dict(section="rule", cohort=name, rule=rule, code=x, P=k[(1, x)], B=k[(0, x)], NP=NP, NB=NB,
                            LR=lr, lo95_exact=lo, hi95_exact=hi))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    log = Tee(os.devnull if args.check_only else f"{OUT}/results/Q2_pooled_calibration_rule.txt")
    log("=" * 104)
    log("Q2 -- EVIDENCE INTERVALS AND THE SUPPORTING RULE ON THE POOLED TEMPORAL TEST SET (Amendment 9)" + ("  [check only]" if args.check_only else ""))
    log("A pooled reassessment, descriptive, no verdict. Thresholds and the fallback prior 0.0441 are exploratory, not clinically")
    log("validated. Pooling cannot reinstate the BP4 Strong level dropped at Gate G3.")
    log("=" * 104)
    verify_locks(log, LOCK_FILES, extra=None if args.check_only else {RAW[2]: TOPUP_MD5})
    p = pooled_cohort(with_scores=not args.check_only)
    pg = pathogenic_genes()
    cal = p[in_cohort(p, pg)].reset_index(drop=True)
    one = cal[~cal.review_status.isin(ZERO_STAR)].reset_index(drop=True)
    cohorts = {"calibration cohort, >= 1 star": one, "calibration cohort, all statuses": cal,
               "resolved VUS, >= 1 star": one[was_vus_2025(one).values].reset_index(drop=True),
               "resolved VUS, all statuses": cal[was_vus_2025(cal).values].reset_index(drop=True)}
    vus = load_vus_cohort(pg) if not args.check_only else None
    log("\nDenominator check against Amendment 9 (n, P/LP; SpliceAI-scored n, P/LP), before anything is computed")
    bad = []
    for name, (w1, w2) in EXPECTED.items():
        d = cohorts[name]
        g1 = (len(d), int(d.label.sum()))
        s = d[d.SpliceAI.notna()]
        g2 = (len(s), int(s.label.sum())) if w2 else None
        ok = g1 == w1 and (w2 is None or g2 == w2)
        log(f"  {name:<34} {g1}" + (f"  SpliceAI {g2}" if w2 else "") + f"   recorded {w1}" + (f" {w2}" if w2 else "") + f"   {'ok' if ok else 'DIFFERS'}")
        bad += [name] if not ok else []
    if bad:
        raise SystemExit(f"denominators differ from Amendment 9; nothing computed: {bad}")
    log("  every denominator equals Amendment 9")
    if args.check_only:
        return
    pk = set(map(tuple, p[KEY].astype(str).values))
    rem = vus[[tuple(x) not in pk for x in vus[KEY].astype(str).values]].reset_index(drop=True)
    if len(rem) != 60776:
        raise SystemExit(f"remaining VUS {len(rem):,}; Amendment 9 recorded 60,776")
    raw = pd.concat([pd.read_csv(f"{OUT}/{f}", usecols=KEY + ["spliceai_raw_max", "spliceai_raw_max_allgenes"]) for f in RAW])
    dup = raw[raw.duplicated(KEY, keep=False)]
    if len(dup) and (dup.groupby(KEY)[["spliceai_raw_max", "spliceai_raw_max_allgenes"]].nunique(dropna=False) > 1).any().any():
        raise ValueError("a variant has different raw scores in two raw-score files")
    raw = raw.drop_duplicates(KEY)
    cohorts = {k: add_raw(v, raw) for k, v in cohorts.items()}
    rem = add_raw(rem, raw)
    tsv = []

    log("\n1. S6 LEVELS, POOLED REASSESSMENT (interval LR, exact 95% CI; does the point estimate / the CI meet the requirement)")
    for name in ["calibration cohort, >= 1 star", "calibration cohort, all statuses"]:
        d = cohorts[name]
        for tool, col in [("MetaSplice", "MetaSplice"), ("SpliceAI (precomputed masked)", "SpliceAI")]:
            th = load_thresholds("MetaSplice" if tool == "MetaSplice" else "SpliceAI", path=S6)
            codes = assign(d[col].values.astype(float), *th)
            log(f"\n  {name} -- {tool}")
            log(f"    {'interval':<16}{'P':>5}{'B':>8}{'LR':>11}   exact 95% CI          needed    point  CI")
            for cd, kp, kb, lr, lo, hi in interval_lr(codes, d.label.values):
                point, ci, ns = verdicts(cd, lr, lo, hi)
                his = "inf" if hi == np.inf else f"{hi:.3g}"
                note = "   (dropped at Gate G3; reported as measured)" if (tool == "MetaSplice" and cd == "BP4_Strong") else ""
                log(f"    {cd:<16}{kp:>5}{kb:>8,}{lr:>11.3g}   [{lo:.3g}, {his}]".ljust(62) + f"{ns:<10}{point:<7}{ci}{note}")
                tsv.append(dict(section="S6 levels", cohort=name, rule=tool, code=cd, P=kp, B=kb, LR=lr, lo95_exact=lo,
                                hi95_exact=hi, point=point, ci=ci))

    log("\n2. THE AMENDMENT 6 RULE AGAINST SPLICEAI'S RULE (calibration cohort)")
    for name in ["calibration cohort, >= 1 star", "calibration cohort, all statuses"]:
        rule_report(log, name, cohorts[name], tsv)
    log("\n3. RESOLVED-VUS SUBGROUP (uncertain or conflicting in 2025-08-31)")
    for name in ["resolved VUS, >= 1 star", "resolved VUS, all statuses"]:
        rule_report(log, name, cohorts[name], tsv)

    log(f"\n4. REMAINING VUS (n={len(rem):,}; no labels)")
    s = rem.MetaSplice.values
    log(f"  MetaSplice >= 0.1 {np.mean(s >= 0.1):6.1%}   >= 0.2 {np.mean(s >= 0.2):6.1%}   >= 0.5 {np.mean(s >= 0.5):6.1%}")
    tsv += [dict(section="remaining VUS shares", threshold=t, share=float(np.mean(s >= t)), n=len(rem)) for t in (0.1, 0.2, 0.5)]
    for rule, cc in rule_codes(rem).items():
        k = {x: int((cc == x).sum()) for x in ("PP3", "BP4", "")}
        log(f"  {rule:<28} PP3 {k['PP3']:>7,} ({k['PP3'] / len(rem):5.1%})   BP4 {k['BP4']:>7,} ({k['BP4'] / len(rem):5.1%})   "
            f"neither {k['']:>7,} ({k[''] / len(rem):5.1%})")
        tsv += [dict(section="remaining VUS codes", rule=rule, code=x or "neither", count=v, share=v / len(rem)) for x, v in k.items()]
    pd.DataFrame(tsv).to_csv(f"{OUT}/results/Q2_pooled_calibration_rule.tsv", sep="\t", index=False)
    log(f"\nwrote results/Q2_pooled_calibration_rule.tsv ({len(tsv)} rows)")


if __name__ == "__main__":
    main()
