"""P3 -- VUS under the Amendment 6 rule (protocol Amendment 8; descriptive, no verdict).

MetaSplice's rule: PP3 at Supporting for score >= 0.544, BP4 at Supporting for score <= 0.0017 (S6's PP3 Strong and BP4
Moderate thresholds). SpliceAI's rule (Walker et al. 2023): PP3 at >= 0.2, BP4 at <= 0.1, with the precomputed masked
scores (V08, +-50 nt) and with SpliceAI 1.3.1 raw scores over +-4,999 nt (V11b; first record, all-genes maximum beside it);
unscored variants receive no code.
  1. VUS cohort: S5's yield cohort (61,016 VUS or conflicting variants of the 2026-04-15 release, calibration-cohort
     filters; no labels). MetaSplice score shares at >= 0.1, 0.2 and 0.5 (also for all 71,267 VUS in scope); share given
     PP3, BP4 or neither under each rule; MetaSplice's codes against SpliceAI's raw-score codes.
  2. Resolved VUS: S7's cohort (659: 107 later P/LP, 552 later B/LB). For each rule the four shares and the share given
     neither, interval LRs with exact CIs (calib.lr_ci), and MetaSplice minus SpliceAI paired differences with
     stratified and gene-clustered bootstraps (X3's boot_diffs; 1,000 iterations, seed 42).
Inputs and denominators are asserted first; --check-only stops there.

Writes results/P3_vus_rule.txt and .tsv.
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
from common import CLINVAR_2025, OUT, Tee, in_training, score
from calib import load_validation_cohort, lr_ci, pathogenic_genes, was_vus_2025
from paper_cohorts import load_vus_cohort, verify_locks
from X3_supporting_rule import SAI_BP4, SAI_PP3, SHARES, T_BP4, T_PP3, boot_diffs

RAW_VUS = "data/v2_spliceai_raw_d4999_vus.csv"
RAW_VUS_MD5 = "5578901fa0ac8a8034fd11cf3986755d"     # Amendment 8: added after V11b writes the file, before P3 computes anything
LOCK_FILES = ["models/metasplice_v2_full.joblib", "models/metasplice_v2_features.joblib", "data/v2_features_vus.csv",
              "data/v2_features_test_candidates.csv", "data/v2_variants_test_candidates.csv", "data/v2_variants_training.csv",
              "data/v2_comparators.csv", "data/v2_pathogenic_genes.csv", "results/S6_thresholds.tsv",
              CLINVAR_2025["GRCh38"], CLINVAR_2025["GRCh37"]]
KEY = ["chr", "pos", "ref", "alt"]


def code(s, hi, lo):
    c = np.full(len(s), "", dtype=object)
    c[s >= hi] = "PP3"          # NaN compares False: an unscored variant gets no code
    c[s <= lo] = "BP4"
    return c


def rules(d):
    return {"MetaSplice": code(d.MetaSplice.values.astype(float), T_PP3, T_BP4),
            "SpliceAI, masked": code(d.SpliceAI.values.astype(float), SAI_PP3, SAI_BP4),
            "SpliceAI, raw (first record)": code(d.spliceai_raw_max.values.astype(float), SAI_PP3, SAI_BP4),
            "SpliceAI, raw (all genes)": code(d.spliceai_raw_max_allgenes.values.astype(float), SAI_PP3, SAI_BP4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    log = Tee(os.devnull if args.check_only else f"{OUT}/results/P3_vus_rule.txt")
    log("=" * 100)
    log("P3 -- VUS UNDER THE AMENDMENT 6 RULE (Amendment 8; descriptive, no verdict)" + ("  [check only]" if args.check_only else ""))
    log(f"MetaSplice: PP3 at Supporting for score >= {T_PP3:.3g}; BP4 at Supporting for score <= {T_BP4:.3g}. "
        f"SpliceAI: PP3 at >= {SAI_PP3}; BP4 at <= {SAI_BP4}")
    log("=" * 100)
    verify_locks(log, LOCK_FILES, extra=None if args.check_only else {RAW_VUS: RAW_VUS_MD5})
    yld = load_vus_cohort(pathogenic_genes())
    val = load_validation_cohort("reclassified", one_star_only=True)   # S7's cohort; the derivation cohort is not loaded
    res = val[was_vus_2025(val)].reset_index(drop=True)
    allvus = pd.read_csv(f"{OUT}/data/v2_features_vus.csv", low_memory=False)
    if in_training(allvus).any():
        raise ValueError("a VUS is in the training data")
    got = (len(yld), len(allvus), len(res), int(res.label.sum()))
    log(f"\nDenominators: VUS cohort {got[0]:,}; all VUS in scope {got[1]:,}; resolved-VUS cohort {got[2]} ({got[3]} later P/LP)")
    if got != (61016, 71267, 659, 107):
        raise SystemExit(f"denominators differ from Amendment 8: {got}")
    log("  every denominator equals Amendment 8")
    if args.check_only:
        return
    raw = pd.read_csv(f"{OUT}/{RAW_VUS}", low_memory=False)
    yld = yld.merge(raw[KEY + ["spliceai_raw_max", "spliceai_raw_max_allgenes"]], on=KEY, how="left", validate="one_to_one")
    res = res.merge(raw[KEY + ["spliceai_raw_max", "spliceai_raw_max_allgenes"]], on=KEY, how="left", validate="one_to_one")
    allvus["MetaSplice"] = score(allvus)
    tsv = []

    log(f"\n1. VUS COHORT (n={len(yld):,}; no labels)")
    for name, d in [("VUS cohort", yld), ("all VUS in scope", allvus)]:
        s = d.MetaSplice.values
        log(f"  {name:<18} MetaSplice >= 0.1 {np.mean(s >= 0.1):6.1%}   >= 0.2 {np.mean(s >= 0.2):6.1%}   >= 0.5 {np.mean(s >= 0.5):6.1%}")
        tsv += [dict(section="score shares", set=name, threshold=t, share=float(np.mean(s >= t)), n=len(d)) for t in (0.1, 0.2, 0.5)]
    c = rules(yld)
    for rule, cc in c.items():
        k = {x: int((cc == x).sum()) for x in ("PP3", "BP4", "")}
        log(f"  {rule:<30} PP3 {k['PP3']:>7,} ({k['PP3'] / len(yld):5.1%})   BP4 {k['BP4']:>7,} ({k['BP4'] / len(yld):5.1%})   "
            f"neither {k['']:>7,} ({k[''] / len(yld):5.1%})")
        tsv += [dict(section="VUS codes", set="VUS cohort", rule=rule, code=x or "neither", count=v, share=v / len(yld)) for x, v in k.items()]
    xt = pd.crosstab(pd.Series(c["MetaSplice"], name="MetaSplice").replace("", "neither"),
                     pd.Series(c["SpliceAI, raw (first record)"], name="SpliceAI raw").replace("", "neither"))
    log("\n  MetaSplice's codes (rows) against SpliceAI's raw-score codes, first record (columns):\n" + xt.to_string())

    log(f"\n2. RESOLVED VUS (n={len(res)}; later P/LP {int(res.label.sum())}, later B/LB {int((res.label == 0).sum())})")
    y = res.label.values
    NP, NB = int(y.sum()), int((y == 0).sum())
    c = rules(res)
    for rule, cc in c.items():
        k = {(cls, x): int(((cc == x) & (y == cls)).sum()) for cls in (1, 0) for x in ("PP3", "BP4", "")}
        log(f"  {rule:<30} P/LP PP3 {k[(1,'PP3')]:>3} ({k[(1,'PP3')] / NP:5.1%})  B/LB PP3 {k[(0,'PP3')]:>3} ({k[(0,'PP3')] / NB:5.1%})  "
            f"B/LB BP4 {k[(0,'BP4')]:>3} ({k[(0,'BP4')] / NB:5.1%})  P/LP BP4 {k[(1,'BP4')]:>2} ({k[(1,'BP4')] / NP:4.1%})  "
            f"neither P/LP {k[(1,'')]:>2} B/LB {k[(0,'')]:>3}")
        for x in ("PP3", "BP4"):
            lr, lo, hi = lr_ci(k[(1, x)], NP, k[(0, x)], NB)
            his = "inf" if hi == np.inf else f"{hi:.3g}"
            log(f"      {x} variants: LR {lr:.3g} [{lo:.3g}, {his}]")
            tsv.append(dict(section="resolved VUS", rule=rule, code=x, P=k[(1, x)], B=k[(0, x)], NP=NP, NB=NB, LR=lr,
                            lo95_exact=lo, hi95_exact=hi))
    for comp in ["SpliceAI, masked", "SpliceAI, raw (first record)", "SpliceAI, raw (all genes)"]:
        log(f"  MetaSplice minus {comp}: share difference [stratified 95% CI] [gene-clustered 95% CI]")
        for label, cls, x in SHARES:
            a, b = (c["MetaSplice"] == x), (c[comp] == x)
            sel = y == cls
            diff = a[sel].mean() - b[sel].mean()
            (slo, shi), (glo, ghi) = boot_diffs(y, a, b, res.gene_id.values)[cls]
            log(f"      {label:<16}{diff:+8.2%}  [{slo:+.2%}, {shi:+.2%}]  [{glo:+.2%}, {ghi:+.2%}]")
            tsv.append(dict(section="resolved VUS difference", rule=f"MetaSplice minus {comp}", code=label, diff=diff,
                            lo=slo, hi=shi, gene_lo=glo, gene_hi=ghi))
    pd.DataFrame(tsv).to_csv(f"{OUT}/results/P3_vus_rule.tsv", sep="\t", index=False)
    log(f"\nwrote results/P3_vus_rule.tsv ({len(tsv)} rows)")



if __name__ == "__main__":
    main()
