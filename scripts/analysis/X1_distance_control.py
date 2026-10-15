"""X1 -- EXPLORATORY distance control of the deep-intronic results (docs/METASPLICE_V2_PLAN.md, Task 11 Step 4b).

Not pre-specified in Amendment 3; descriptive only. It changes neither Gate G2, the prior nor any threshold.

Why: ClinVar's B/LB deep-intronic variants lie near exons and its P/LP ones further in (S11_prior_diagnostic), so
distance to the exon boundary alone separates the classes to some degree. This reports how far, and how the tools
compare within fixed distance bands, on the strict test set's Deep Intronic variants and the O5 sensitivity set
(Deep Intronic, neither canonical nor exonic in any GENCODE v50 transcript).
  - distance-only baseline: |dist_to_boundary| (the distance to the nearest exon boundary of the assigned gene, a
    model feature) used as a score, larger = more likely P/LP; auPRC with a stratified bootstrap CI;
  - within the bands <= 50, 51-500 and > 500 bp: counts; each tool's auPRC; paired MetaSplice v2.0 minus SpliceAI
    and minus CADD (common.paired_boot_diff, 1,000 iterations, seed 42). The <= 50 bp band holds 4 P/LP in the
    primary set and is reported with counts only, without performance values.

Writes results/X1_distance_control.txt and .tsv.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import OUT, Tee, add_comparators, auprc, boot_ci, deep_intronic_sensitivity, load_test, paired_boot_diff, score

TOOLS = [("MetaSplice v2.0", "MetaSplice"), ("SpliceAI", "spliceai_max"), ("CADD", "cadd_phred"), ("distance only", "dist")]
BANDS = [("<= 50 bp", lambda d: d <= 50), ("51-500 bp", lambda d: (d > 50) & (d <= 500)), ("> 500 bp", lambda d: d > 500)]
COUNTS_ONLY = {"<= 50 bp"}


def main():
    log = Tee(f"{OUT}/results/X1_distance_control.txt")
    log("=" * 96)
    log("X1 -- EXPLORATORY: DISTANCE CONTROL OF THE DEEP-INTRONIC RESULTS (strict test set; descriptive only)")
    log("Not pre-specified in Amendment 3. No change to Gate G2, the prior or any threshold.")
    log("=" * 96)
    t = add_comparators(load_test("strict"))
    if t.dist_to_boundary.isna().any():
        raise ValueError("a test variant has no distance to the exon boundary")
    t["MetaSplice"], t["dist"] = score(t), t.dist_to_boundary.abs()
    rows = []
    for set_name, sel in [("Deep Intronic", t.region == "Deep Intronic"), ("Deep Intronic, sensitivity set", deep_intronic_sensitivity(t))]:
        d = t[sel.values].reset_index(drop=True)
        y = d.label.values
        log(f"\n{set_name}: n={len(d):,}  P/LP={int(y.sum())}  B/LB={int((y == 0).sum()):,}")
        log(f"  |distance| median: P/LP {np.median(d.dist[y == 1]):,.0f} bp, B/LB {np.median(d.dist[y == 0]):,.0f} bp")
        log(f"  {'tool':<18}{'n':>8}{'P':>5}{'auPRC':>9}{'95% CI':>20}{'AUROC':>9}")
        for name, col in TOOLS:
            s = d[col].values.astype(float)
            ok = ~np.isnan(s)
            ap, lo, hi = boot_ci(y, s)
            ar = roc_auc_score(y[ok], s[ok])
            rows.append(dict(set=set_name, band="all", analysis="marginal", tool=name, n=int(ok.sum()), P=int(y[ok].sum()),
                             auprc=ap, lo=lo, hi=hi, auroc=ar))
            log(f"  {name:<18}{ok.sum():>8,}{int(y[ok].sum()):>5}{ap:>9.4f}   [{lo:.4f}, {hi:.4f}]{ar:>9.4f}")
        dd, lo, hi, n = paired_boot_diff(y, d.MetaSplice.values, d.dist.values.astype(float))
        rows.append(dict(set=set_name, band="all", analysis="paired", tool="MetaSplice v2.0 minus distance only", n=n,
                         P=int(y.sum()), diff=dd, lo=lo, hi=hi))
        log(f"  paired, MetaSplice v2.0 minus distance only: {dd:+.4f} [{lo:+.4f}, {hi:+.4f}]  n={n:,}")

        log(f"\n  Within distance bands (|distance to the exon boundary|):")
        for band, f in BANDS:
            b = d[f(d.dist).values].reset_index(drop=True)
            yb = b.label.values
            log(f"\n  {band}: n={len(b):,}  P/LP={int(yb.sum())}  B/LB={int((yb == 0).sum()):,}  share P/LP {yb.mean():.2%}")
            rows.append(dict(set=set_name, band=band, analysis="counts", n=len(b), P=int(yb.sum()), B=int((yb == 0).sum())))
            if band in COUNTS_ONLY:
                log("    counts only: too few P/LP for performance values")
                continue
            for name, col in TOOLS:
                s = b[col].values.astype(float)
                ok = ~np.isnan(s)
                ap, lo, hi = boot_ci(yb, s)
                rows.append(dict(set=set_name, band=band, analysis="marginal", tool=name, n=int(ok.sum()), P=int(yb[ok].sum()),
                                 auprc=ap, lo=lo, hi=hi))
                log(f"    {name:<18} n={ok.sum():>6,} P={int(yb[ok].sum()):>3}  auPRC {ap:.4f} [{lo:.4f}, {hi:.4f}]")
            for name, col in TOOLS[1:3]:
                dd, lo, hi, n = paired_boot_diff(yb, b.MetaSplice.values, b[col].values)
                ok = b[col].notna().values
                rows.append(dict(set=set_name, band=band, analysis="paired", tool=f"MetaSplice v2.0 minus {name}", n=n,
                                 P=int(yb[ok].sum()), diff=dd, lo=lo, hi=hi))
                log(f"    paired, MetaSplice v2.0 minus {name:<9} n={n:>6,} P={int(yb[ok].sum()):>3}  {dd:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    pd.DataFrame(rows).to_csv(f"{OUT}/results/X1_distance_control.tsv", sep="\t", index=False)
    log(f"\nwrote results/X1_distance_control.tsv ({len(rows)} rows)")
    log.close()


if __name__ == "__main__":
    main()
