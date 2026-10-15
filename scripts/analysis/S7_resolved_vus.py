"""S7 -- Candidate evidence for variants that were uncertain or conflicting at training time (Amendment 3).

The reclassified subgroup (common.load_test(subset="reclassified"): absent from the training data but in
the 2025-08-31 release), restricted to >= 1 star, the calibration cohort and records uncertain or
conflicting in the 2025-08-31 release, that are P/LP or B/LB by 2026-04-15: 659 (107 P/LP, 552 B/LB),
scored against the S6 thresholds. A subgroup reported on its own, not part of the test set.

Writes results/S7_resolved_vus.txt and .tsv.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import OUT, Tee
from calib import LEVELS, assign, compared_models, interval_lr, load_thresholds, one_star_cohorts, verdicts, was_vus_2025

S6 = f"{OUT}/results/S6_thresholds.tsv"
PP3, BP4 = [f"PP3_{l}" for l in LEVELS], [f"BP4_{l}" for l in LEVELS]
EXPECTED = {"all resolved": (659, 107), "SpliceAI-scored": (656, 106)}     # Amendment 3


def main():
    log = Tee(f"{OUT}/results/S7_resolved_vus.txt")
    log("=" * 88)
    log("S7 -- UNCERTAIN OR CONFLICTING AT TRAINING TIME, RESOLVED BY 2026-04-15 (S6 thresholds; Amendment 3)")
    log("=" * 88)
    _, val = one_star_cohorts("reclassified")
    r = val[was_vus_2025(val)]
    subsets = [("all resolved", r), ("SpliceAI-scored", r[r.SpliceAI.notna()])]
    for name, d in subsets:
        got = (len(d), int(d.label.sum()))
        if got != EXPECTED[name]:
            raise SystemExit(f"{name}: {got} differs from Amendment 3's {EXPECTED[name]}")
    tools = compared_models() + ["SpliceAI"]
    rows = []
    for name, d in subsets:
        y = d.label.values
        log(f"\n{name}: n={len(d)}  later P/LP={int(y.sum())}  later B/LB={int((y == 0).sum())}")
        for t in tools:
            codes = assign(d[t].values.astype(float), *load_thresholds(t, path=S6))
            log(f"  {t:<12} later-P/LP assigned candidate PP3 {np.isin(codes[y == 1], PP3).mean():6.1%}   "
                f"later-B/LB assigned candidate BP4 {np.isin(codes[y == 0], BP4).mean():6.1%}   "
                f"later-P/LP assigned candidate BP4 {np.isin(codes[y == 1], BP4).mean():6.1%}")
            for code, kp, kb, lr, lo, hi in interval_lr(codes, y):
                point, ci, _ = verdicts(code, lr, lo, hi)
                rows.append((name, t, code, kp, kb, lr, lo, hi, point, ci))
    df = pd.DataFrame(rows, columns=["subset", "tool", "interval", "P", "B", "LR", "lo95_exact", "hi95_exact",
                                     "point_verdict", "ci_verdict"])
    df.to_csv(f"{OUT}/results/S7_resolved_vus.tsv", sep="\t", index=False)
    log("\nInterval LRs (all resolved):")
    log(df[df.subset == "all resolved"].drop(columns="subset").to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    log.close()


if __name__ == "__main__":
    main()
