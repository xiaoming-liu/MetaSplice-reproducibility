"""S9 -- Candidate evidence by gene: genes with >= 10 P/LP in the >= 1-star validation cohort (Amendment 3).

v1's rule, unchanged. Interval LR (conservative exact 95% CI) for any candidate PP3 and any candidate
BP4, per qualifying gene and pooled over the remaining genes, with the S6 thresholds. A gene falls short
of Supporting when its any-PP3 LR < 2.406 or its any-BP4 LR > 0.4156. One gene qualifies (Amendment 3):
its result is a descriptive single-gene result, not evidence of performance across genes.

Writes results/S9_per_gene.txt and .tsv.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import OUT, Tee
from calib import LEVELS, LR_MAIN, assign, compared_models, load_thresholds, lr_ci, one_star_cohorts

S6 = f"{OUT}/results/S6_thresholds.tsv"
PP3, BP4 = [f"PP3_{l}" for l in LEVELS], [f"BP4_{l}" for l in LEVELS]
MIN_P = 10
EXPECTED_GENES = 1      # Amendment 3


def gene_rows(d, t, gene):
    codes = assign(d[t].values.astype(float), *load_thresholds(t, path=S6))
    y = d.label.values
    NP, NB = int((y == 1).sum()), int((y == 0).sum())
    out = []
    for name, lev, ok in [("any PP3", PP3, lambda lr: lr >= LR_MAIN["Supporting"]),
                          ("any BP4", BP4, lambda lr: lr <= 1 / LR_MAIN["Supporting"])]:
        m = np.isin(codes, lev)
        kp, kb = int(m[y == 1].sum()), int(m[y == 0].sum())
        lr, lo, hi = lr_ci(kp, NP, kb, NB) if NB and NP else (np.nan, np.nan, np.nan)
        out.append((gene, t, name, NP, NB, kp, kb, lr, lo, hi, "meets" if ok(lr) else "SHORT"))
    return out


def main():
    log = Tee(f"{OUT}/results/S9_per_gene.txt")
    log("=" * 88)
    log(f"S9 -- CANDIDATE EVIDENCE BY GENE (>= {MIN_P} P/LP in the >= 1-star validation cohort; S6 thresholds; Amendment 3)")
    log("=" * 88)
    _, val = one_star_cohorts()
    val["gene_label"] = val.gene.astype(str) + " (" + val.gene_id.astype(str) + ")"
    counts = val.groupby("gene_label").label.sum()
    genes = counts[counts >= MIN_P].sort_values(ascending=False).index.tolist()
    if len(genes) != EXPECTED_GENES:
        raise SystemExit(f"{len(genes)} qualifying genes; Amendment 3 recorded {EXPECTED_GENES}")
    log(f"{len(genes)} gene with >= {MIN_P} P/LP, covering {int(counts[genes].sum())} of {int(val.label.sum())} P/LP. "
        f"Reported as a descriptive single-gene result, not as evidence of performance across genes.")
    rows = []
    for t in compared_models() + ["SpliceAI"]:
        d_all = val[val[t].notna()]
        for g in genes + ["(all other genes)"]:
            d = d_all[d_all.gene_label == g] if g != "(all other genes)" else d_all[~d_all.gene_label.isin(genes)]
            rows += gene_rows(d, t, g)
    df = pd.DataFrame(rows, columns=["gene", "tool", "evidence", "P", "B", "P_in", "B_in", "LR", "lo95_exact",
                                     "hi95_exact", "point_verdict"])
    df.to_csv(f"{OUT}/results/S9_per_gene.tsv", sep="\t", index=False)
    log(df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    log.close()


if __name__ == "__main__":
    main()
