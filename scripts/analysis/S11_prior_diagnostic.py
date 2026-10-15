"""S11_prior diagnostic -- why the mixture fit failed (descriptive; computed after the frozen decision).

S11_prior's primary fit was 0.41 against a limit of 0.02, so the estimate was not promoted and the prior
stays 0.0441 (results/S11_prior_frozen.json). This compares the reference with the ClinVar components that
were meant to describe it: score quantiles, and distance to the exon boundary of the Deep Intronic variants
(with the deep-intronic sensitivity set). It describes the mismatch and does not establish its cause;
it changes nothing in the frozen decision.

Writes results/S11_prior_diagnostic.txt.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import KEY, ROOT, Tee, deep_intronic_sensitivity
from calib import load_derivation_cohort

Q = [0.1, 0.25, 0.5, 0.75, 0.9]


def main():
    log = Tee(f"{ROOT}/results/S11_prior_diagnostic.txt")
    log("S11_prior diagnostic -- reference against the ClinVar components (descriptive; does not change the frozen prior)")
    der = load_derivation_cohort(one_star_only=True)
    ref = pd.read_csv(f"{ROOT}/data/v2_gnomad_reference_scores.csv", low_memory=False)
    ref = ref[ref["sample"] == "uniform"].merge(
        pd.read_csv(f"{ROOT}/data/v2_features_gnomad_reference.csv", usecols=KEY + ["dist_to_boundary"]), on=KEY, validate="one_to_one")
    log(f"\nScore quantiles {Q} (MetaSplice v2.0; reference: gene-matched fold score; components: out-of-fold)")
    for name, s in [("gnomAD reference (uniform)", ref.MetaSplice_fold), ("B/LB component", der.MetaSplice[der.label == 0]),
                    ("P/LP component", der.MetaSplice[der.label == 1])]:
        log(f"  {name:<28} " + "  ".join(f"{v:.2e}" for v in np.quantile(s, Q)))
    log("\nDeep Intronic: |distance to the nearest exon boundary of the assigned gene| (bp)")
    for label, d in [("gnomAD reference", ref), ("B/LB component", der[der.label == 0]), ("P/LP component", der[der.label == 1])]:
        for sub, m in [("Deep Intronic", d.region == "Deep Intronic"), ("sensitivity set", deep_intronic_sensitivity(d))]:
            x = np.abs(d.dist_to_boundary[m.values])
            log(f"  {label:<18} {sub:<16} n={len(x):>7,}  median {np.median(x):>8,.0f}  share > 500 bp {(x > 500).mean():6.1%}  "
                f"share <= 50 bp {(x <= 50).mean():6.1%}")
    log("\nReading (descriptive): ClinVar's B/LB deep-intronic variants lie near exons, while its P/LP deep-intronic variants and\n"
        "the population reference lie far into introns, and the reference scores above the B/LB component. The distance\n"
        "imbalance is a plausible contributor to that score shift; this comparison does not establish its cause. Whatever the\n"
        "cause, a mixture of the two ClinVar components does not reproduce the reference's score distribution, which is what\n"
        "the fit criterion tests. The fallback prior 0.0441 is the procedure's default, not a validated population prior.")
    log.close()


if __name__ == "__main__":
    main()
