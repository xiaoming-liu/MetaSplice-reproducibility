"""S10 -- MetaSplice v2.0 against SpliceAI and CADD (docs/AJHG_COMPARISON_PROTOCOL.md, Amendment 3).

Pre-specified in Amendment 3 (2026-10-01), before any v2.0 test-set performance was calculated.
Primary: on the non-canonical variants of the v2.0 strict test set (common.load_test), the auPRC of
MetaSplice v2.0 (common.score, the full model) against SpliceAI and against CADD, each on the variants
both score, with a paired stratified bootstrap (common.paired_boot_diff, 1,000 iterations, seed 42).
Secondary rows carry no verdict. The test set comes from a release v1 analysed (Amendment 3, "What is
already known"), so this is a pre-specified comparison, not a fresh hold-out.

Before any auPRC is computed, the n and P/LP of every pairing must equal the denominators Amendment 3
records (EXPECTED below), or the script stops. With --check-only it runs only that check and loads no
model: MetaSplice scores every in-scope variant, so its pairings are counted from the feature rows.

Writes results/S10_head_to_head.txt and .tsv.
"""
import argparse, hashlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import (FEATURES, NONCANONICAL, OUT, Tee, add_comparators, auprc, deep_intronic_sensitivity,
                    load_test, paired_boot_diff, score)
from calib import in_cohort, one_star, pathogenic_genes

PROTOCOL = os.path.join(OUT, "..", "docs", "AJHG_COMPARISON_PROTOCOL.md")
MODEL = "MetaSplice"
COMPARATORS = [("SpliceAI", "spliceai_max"), ("CADD", "cadd_phred")]
# The denominators recorded in Amendment 3 (from results/C0_cohorts.txt): n and P/LP of each subset,
# then of its pairing with each comparator.
EXPECTED = {
    "all non-canonical":              {"all": (25240, 244), "SpliceAI": (25114, 243), "CADD": (25240, 244)},
    "Donor Region":                   {"all": (1766, 102), "SpliceAI": (1756, 101), "CADD": (1766, 102)},
    "Acceptor Region":                {"all": (4810, 36), "SpliceAI": (4790, 36), "CADD": (4810, 36)},
    "PPT Region":                     {"all": (8359, 21), "SpliceAI": (8319, 21), "CADD": (8359, 21)},
    "Branch Point Region":            {"all": (186, 5), "SpliceAI": (186, 5), "CADD": (186, 5)},
    "Deep Intronic":                  {"all": (10119, 80), "SpliceAI": (10063, 80), "CADD": (10119, 80)},
    "Deep Intronic, sensitivity set": {"all": (8312, 62), "SpliceAI": (8260, 62), "CADD": (8312, 62)},
    ">=1-star calibration cohort":    {"all": (24049, 231), "SpliceAI": (23928, 230), "CADD": (24049, 231)},
    "all three scored":               {"all": (25114, 243), "SpliceAI": (25114, 243), "CADD": (25114, 243)},
}
NOT_INTERPRETED = {"Branch Point Region": "5 P/LP: reported with its denominator, not interpreted"}


def subsets(t):
    """The subsets of Amendment 3, from the strict test set t (comparator columns added)."""
    nc = t[t.region.isin(NONCANONICAL)].reset_index(drop=True)
    out = {"all non-canonical": nc}
    out.update({r: nc[nc.region == r] for r in NONCANONICAL})
    out["Deep Intronic, sensitivity set"] = nc[deep_intronic_sensitivity(nc)]
    val = t[in_cohort(t, pathogenic_genes())]
    out[">=1-star calibration cohort"] = val[one_star(val)]
    out["all three scored"] = nc[nc.spliceai_max.notna() & nc.cadd_phred.notna()]
    return out


def pairing_check(log, subs):
    """Raise unless every subset and pairing has the n and P/LP recorded in Amendment 3."""
    log("\nPairing check against Amendment 3 (n, P/LP), before any auPRC is computed")
    bad = []
    for name, d in subs.items():
        for pairing in ["all"] + [c for c, _ in COMPARATORS]:
            ok = d[MODEL].notna() if pairing == "all" else d[MODEL].notna() & d[dict(COMPARATORS)[pairing]].notna()
            got, want = (int(ok.sum()), int(d.label[ok].sum())), EXPECTED[name][pairing]
            log(f"  {name:<32} {pairing:<9} n={got[0]:>7,} P={got[1]:>4}   recorded n={want[0]:>7,} P={want[1]:>4}"
                f"   {'ok' if got == want else 'DIFFERS'}")
            if got != want:
                bad.append(f"{name} / {pairing}: {got} vs recorded {want}")
    if set(subs) != set(EXPECTED):
        bad.append(f"subsets differ from Amendment 3: {sorted(set(subs) ^ set(EXPECTED))}")
    if bad:
        raise SystemExit("pairing differs from Amendment 3; nothing computed:\n  " + "\n  ".join(bad))
    log("  every pairing equals Amendment 3")


def compare(d, a, b):
    """auPRC of columns a and b on rows both score, and the paired bootstrap difference."""
    ok = d[a].notna() & d[b].notna()
    y = d.label.values[ok]
    diff, lo, hi, n = paired_boot_diff(d.label.values, d[a].values, d[b].values)
    return dict(n=n, P=int(y.sum()), auprc_a=auprc(y, d[a].values[ok]), auprc_b=auprc(y, d[b].values[ok]),
                diff=diff, lo=lo, hi=hi)


def verdict(r):
    if np.isnan(r["lo"]):
        return "not estimable"
    return ("MetaSplice v2.0 better" if r["lo"] > 0 else
            "MetaSplice v2.0 worse" if r["hi"] < 0 else "indistinguishable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="run the pairing check only; load no model")
    args = ap.parse_args()
    log = Tee(os.devnull if args.check_only else f"{OUT}/results/S10_head_to_head.txt")
    log("=" * 96)
    log("S10 -- METASPLICE v2.0 AGAINST SPLICEAI AND CADD: PROTOCOL AMENDMENT 3"
        + ("  [pairing check only]" if args.check_only else ""))
    log("=" * 96)
    log(f"protocol SHA-256: {hashlib.sha256(open(PROTOCOL, 'rb').read()).hexdigest()}" if os.path.exists(PROTOCOL)
        else "protocol SHA-256: not computed (docs/AJHG_COMPARISON_PROTOCOL.md is not present)")
    t = add_comparators(load_test("strict"))
    if args.check_only:
        if t[FEATURES].isna().all(axis=1).any():
            raise ValueError("a test variant has no feature row")
        t[MODEL] = 0.0                      # placeholder: presence only; no model is loaded
    else:
        t[MODEL] = score(t)
        if t[MODEL].isna().any():
            raise ValueError("MetaSplice v2.0 left a test variant unscored")
    subs = subsets(t)
    pairing_check(log, subs)
    if args.check_only:
        return

    log("\npaired stratified bootstrap, 1,000 iterations, seed 42; diff = MetaSplice v2.0 minus comparator")
    rows = []

    def add(role, subset, other, r, v=""):
        note = NOT_INTERPRETED.get(subset, "")
        rows.append(dict(role=role, subset=subset, model="MetaSplice v2.0", comparator=other, **r, verdict=v, note=note))
        log(f"  {subset:<32} vs {other:<9} n={r['n']:>7,} P={r['P']:>4}  {r['auprc_a']:.4f} vs {r['auprc_b']:.4f}  "
            f"diff {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]" + (f"  -> {v}" if v else "") + (f"  ({note})" if note else ""))

    log("\nPRIMARY: non-canonical strict test set, each comparator on the variants both score")
    for name, col in COMPARATORS:
        r = compare(subs["all non-canonical"], MODEL, col)
        add("primary", "all non-canonical", name, r, verdict(r))

    log("\nSECONDARY 1: each non-canonical region (no verdict)")
    for reg in NONCANONICAL:
        for name, col in COMPARATORS:
            add("secondary 1", reg, name, compare(subs[reg], MODEL, col))

    log("\nSECONDARY 2: deep-intronic sensitivity, neither canonical nor exonic in any GENCODE v50 transcript (no verdict)")
    for name, col in COMPARATORS:
        add("secondary 2", "Deep Intronic, sensitivity set", name, compare(subs["Deep Intronic, sensitivity set"], MODEL, col))

    log("\nSECONDARY 3: >= 1-star calibration validation cohort (S6; no verdict)")
    for name, col in COMPARATORS:
        add("secondary 3", ">=1-star calibration cohort", name, compare(subs[">=1-star calibration cohort"], MODEL, col))

    log("\nSECONDARY 4: the variants all three tools score, one denominator (no verdict)")
    for name, col in COMPARATORS:
        add("secondary 4", "all three scored", name, compare(subs["all three scored"], MODEL, col))

    pd.DataFrame(rows).to_csv(f"{OUT}/results/S10_head_to_head.tsv", sep="\t", index=False)
    log(f"\nwrote results/S10_head_to_head.tsv ({len(rows)} rows)")
    log.close()


if __name__ == "__main__":
    main()
