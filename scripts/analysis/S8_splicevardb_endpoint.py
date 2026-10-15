"""S8 -- Candidate intervals against measured splicing (SpliceVarDB), the ClinGen splicing endpoint (Amendment 3).

SpliceVarDB at the variant level (common.load_splicevardb, v1's A6 rule), assay-concordant, non-canonical,
gnomAD AF < 0.01; splice-altering (1) against normal (0), low-frequency splice-altering excluded:
7,318 variants (2,688 splice-altering, 4,630 normal). MetaSplice v2.0 and SpliceAI use their S6 candidate
thresholds; SpliceAI is also shown at the ClinGen splicing cut-offs (>= 0.2 PP3, <= 0.1 BP4). Not a
pathogenicity endpoint.

Writes results/S8_splicevardb_endpoint.txt and .tsv.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import NONCANONICAL, OUT, Tee, add_comparators, load_splicevardb, score
from calib import assign, interval_lr, load_thresholds

S6 = f"{OUT}/results/S6_thresholds.tsv"
EXPECTED = (7318, 2688, 4630)      # Amendment 3


def walker_codes(s):
    out = np.full(len(s), "Indeterminate", dtype=object)
    out[np.isnan(s)] = "NoScore"
    out[s >= 0.2] = "PP3_Supporting"
    out[s <= 0.1] = "BP4_Supporting"
    return out


def main():
    log = Tee(f"{OUT}/results/S8_splicevardb_endpoint.txt")
    log("=" * 88)
    log("S8 -- CANDIDATE INTERVALS AGAINST MEASURED SPLICING (SpliceVarDB; Amendment 3)")
    log("=" * 88)
    s = add_comparators(load_splicevardb())
    if s.gnomad_af.isna().any():
        raise ValueError("gnomad_af is missing for some SpliceVarDB variants")
    d = s[s.region.isin(NONCANONICAL) & s.label_strict.isin([0, 1]) & (s.gnomad_af < 0.01)].reset_index(drop=True)
    y = d.label_strict.astype(int).values
    got = (len(d), int(y.sum()), int((y == 0).sum()))
    if got != EXPECTED:
        raise SystemExit(f"S8 cohort {got} differs from Amendment 3's {EXPECTED}")
    d["MetaSplice"] = score(d)
    log(f"non-canonical, AF < 0.01: n={len(d):,}; splice-altering {y.sum():,}; normal {(y == 0).sum():,} (equals Amendment 3)")
    log("LR = (share of splice-altering in the interval) / (share of normal in the interval)")
    rows = []
    schemes = [("MetaSplice v2.0, S6 candidate intervals", assign(d.MetaSplice.values, *load_thresholds("MetaSplice", path=S6))),
               ("SpliceAI, S6 candidate intervals", assign(d.spliceai_max.values.astype(float), *load_thresholds("SpliceAI", path=S6))),
               ("SpliceAI, ClinGen cut-offs 0.2 / 0.1", walker_codes(d.spliceai_max.values.astype(float)))]
    for label, codes in schemes:
        log(f"\n{label}")
        log(f"  {'interval':<16}{'altering':>9}{'normal':>8}{'LR':>9}   exact 95% CI")
        for code, kp, kb, lr, lo, hi in interval_lr(codes, y):
            his = "inf" if np.isinf(hi) else f"{hi:.3g}"
            log(f"  {code:<16}{kp:>9}{kb:>8}{lr:>9.3g}   [{lo:.3g}, {his}]")
            rows.append((label, code, kp, kb, lr, lo, hi))
        pp = np.char.startswith(codes.astype(str), "PP3"); bb = np.char.startswith(codes.astype(str), "BP4")
        log(f"  splice-altering given candidate PP3: {pp[y == 1].mean():.1%}; normal given candidate BP4: "
            f"{bb[y == 0].mean():.1%}; splice-altering given candidate BP4: {bb[y == 1].mean():.1%}")
    pd.DataFrame(rows, columns=["scheme", "interval", "splice_altering", "normal", "LR", "lo95_exact", "hi95_exact"]).to_csv(
        f"{OUT}/results/S8_splicevardb_endpoint.tsv", sep="\t", index=False)
    log.close()


if __name__ == "__main__":
    main()
