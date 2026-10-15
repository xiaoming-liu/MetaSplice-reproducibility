"""C1 -- Cohorts and comparator coverage of the fresh temporal cohort (ClinVar 2026-09-28): counts only.

Written for protocol Amendment 5 (fresh temporal validation of MetaSplice v2.0), before any v2.0 score is computed
on this cohort. No model is loaded and no score is computed; labels are counted and not otherwise used; comparator
columns are read only for whether a value is present. Sets as in C0, plus the Deep Intronic distance bands (<= 50,
51-500, > 500 bp) of the cohort and of its O5 sensitivity set.

Writes results/C1_fresh_cohorts.txt and .tsv.
"""
import hashlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import NONCANONICAL, REGIONS, ROOT, Tee, deep_intronic_sensitivity, load_fresh
from calib import in_cohort, one_star, pathogenic_genes

PAIRINGS = [("MetaSplice", []), ("MetaSplice & SpliceAI", ["spliceai_max"]), ("MetaSplice & CADD", ["cadd_phred"]),
            ("all three", ["spliceai_max", "cadd_phred"])]
BANDS = [("<= 50 bp", 0, 50), ("51-500 bp", 50, 500), ("> 500 bp", 500, np.inf)]
rows = []


def count(log, cohort, subset, d):
    for pairing, cols in PAIRINGS:
        x = d[d[cols].notna().all(axis=1)] if cols else d
        p, b = int((x.label == 1).sum()), int((x.label == 0).sum())
        rows.append(dict(cohort=cohort, subset=subset, pairing=pairing, n=len(x), P=p, B=b))
        log(f"  {subset:<36} {pairing:<22} n={len(x):>7,}  P/LP={p:>5,}  B/LB={b:>6,}")


def main():
    log = Tee(f"{ROOT}/results/C1_fresh_cohorts.txt")
    log("C1 -- fresh temporal cohort (ClinVar 2026-09-28): counts only; no model loaded, no score computed")
    f = "data/v2_features_fresh_20260928.csv"
    log(f"input: {hashlib.md5(open(f'{ROOT}/{f}', 'rb').read()).hexdigest()}  {f}")
    t = load_fresh()
    pg = pathogenic_genes()
    log(f"cohort: {len(t):,} variants, P/LP {int(t.label.sum()):,}, B/LB {int((t.label == 0).sum()):,}")
    log("\n1. By region")
    for r in REGIONS:
        d = t[t.region == r]
        log(f"  {r:<26} n={len(d):>6,}  P/LP={int(d.label.sum()):>5,}")
    log("\n2. Non-canonical, each region, Deep Intronic bands, the sensitivity set and its bands")
    nc = t[t.region.isin(NONCANONICAL)]
    count(log, "fresh non-canonical", "all non-canonical", nc)
    for r in NONCANONICAL:
        count(log, "fresh non-canonical", r, nc[nc.region == r])
    di, sens = nc[nc.region == "Deep Intronic"], nc[deep_intronic_sensitivity(nc).values]
    for name, d in [("Deep Intronic", di), ("DI sensitivity set", sens)]:
        if name != "Deep Intronic":
            count(log, "fresh non-canonical", name, d)
        a = d.dist_to_boundary.abs()
        for b, lo, hi in BANDS:
            count(log, "fresh non-canonical", f"{name} {b}", d[((a > lo) & (a <= hi)).values])
    log("\n3. Calibration cohorts (non-canonical, gnomAD AF < 0.01, gene with >= 1 P/LP in the training release)")
    val = t[in_cohort(t, pg)]
    count(log, "fresh calibration", "all stars", val)
    count(log, "fresh calibration", ">= 1 star", val[one_star(val)])
    pd.DataFrame(rows).to_csv(f"{ROOT}/results/C1_fresh_cohorts.tsv", sep="\t", index=False)
    log(f"\nwrote results/C1_fresh_cohorts.tsv ({len(rows)} rows)")
    log.close()


if __name__ == "__main__":
    main()
