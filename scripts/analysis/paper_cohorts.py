"""Shared inputs for the Amendment 8 analyses (P1-P3): input locks, the two test cohorts and their rows, and the
denominator checks. Nothing here computes a performance metric.
"""
import hashlib, os
import numpy as np
import pandas as pd
from common import (CLINVAR_2025, NONCANONICAL, OUT, ROOT, add_comparators, deep_intronic_sensitivity, in_training,
                    load_fresh, load_test, score)
from calib import in_cohort

LOCKS = {      # Amendment 8; values recorded by C0, F1, X3 and S11_prior_frozen.json (fold models: first recorded here)
    "models/metasplice_v2_full.joblib": "51c6e5d0db100b33b91907fa857cd9a7",
    "models/metasplice_v2_features.joblib": "8be31360848f2cb43c2b02850bd03815",
    "models/metasplice_v2_folds.json": "dca7cf706ac3e6832495d5e8cf8e6e82",
    "models/metasplice_v2_fold0.joblib": "c3258d9cbf71d2b55ab7d5829e46a1c0",
    "models/metasplice_v2_fold1.joblib": "3dd2a124484a40887c1158b8fe44d471",
    "models/metasplice_v2_fold2.joblib": "bd3205ccafc370771802472dbee39e8c",
    "models/metasplice_v2_fold3.joblib": "4d1703f1716021f0e43f192be79040f6",
    "models/metasplice_v2_fold4.joblib": "79b7f6dafae1d9959d7989f7977a59b7",
    "data/v2_features_training.csv": "4de7d586fd8e4e81ceb7cc84d669f2d2",
    "data/v2_oof.csv": "cb80fac8f03f0e4cfdcd2e11e7424884",
    "data/v2_features_test_candidates.csv": "27fad032309f58f4b058b4c6e1ddc4a9",
    "data/v2_variants_test_candidates.csv": "d9e70ef71b2c1ca149e4cf4ed56d59e0",
    "data/v2_variants_training.csv": "5f2a4b045d7e6113d44896babc0d0c47",
    "data/v2_comparators.csv": "fc9013f3b52502ca25dc1d0ff774b7b8",
    "data/v2_features_fresh_20260928.csv": "f9cd3f227b0357a4c022f5d6814e4ab4",
    "data/v2_features_vus.csv": "eae8dc55b16ef10581b3285ea47beab7",
    "data/v2_pathogenic_genes.csv": "c6738b75048c943e9a95f1545c34b752",
    "data/v2_spliceai_raw_d4999.csv": "31b81cc8553720bc33ed7a39e8636206",
    "data/v2_spliceai_raw_d4999_vus.csv": "5578901fa0ac8a8034fd11cf3986755d",
    "data/v2_features_pooled_test.csv": "a1847d63dd5ab2e4021b1f20a01b22c8",          # V12 (Amendment 9)
    "results/S6_thresholds.tsv": "6a009ca0e0d5c837d7b1633c739ceb41",
    CLINVAR_2025["GRCh38"]: "111c595ba957c7fc93b8833188d4b029",
    CLINVAR_2025["GRCh37"]: "3c0116eaa075ee7c658ca5c70ec44763",
}
EXPECTED = {   # Amendment 8: (n, P/LP)
    ("development", "non-canonical"): (25240, 244), ("development", "non-canonical, SpliceAI-scored"): (25114, 243),
    ("september", "non-canonical"): (14749, 275), ("september", "non-canonical, SpliceAI-scored"): (14552, 273),
    ("pooled", "non-canonical"): (40619, 602), ("pooled", "non-canonical, SpliceAI-scored"): (40296, 600),   # Amendment 9
}
MIN_P = 20


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def verify_locks(log, files, extra=None):
    """Assert the md5 of each named input (keys of LOCKS, plus extra {path: md5}) before anything is computed."""
    log("\nInput lock (md5), before anything is computed")
    want = {f: LOCKS[f] for f in files}
    want.update(extra or {})
    bad = []
    for f, w in want.items():
        got = md5(f if os.path.isabs(f) else f"{OUT}/{f}")
        log(f"  {got}  {f}  {'ok' if got == w else 'DIFFERS from ' + w}")
        bad += [f] if got != w else []
    if bad:
        raise SystemExit(f"inputs differ from Amendment 8's locks; nothing computed: {bad}")
    log("  every input equals its locked md5")


def pooled_cohort(with_scores=True):
    """The pooled temporal test set (V12, Amendment 9), labelled by ClinVar 2026-09-28, with MetaSplice (full model),
    SpliceAI (precomputed masked, V08) and CADD (PHRED). Column source: '2026-04-15' or '2026-09-28'."""
    d = pd.read_csv(f"{OUT}/data/v2_features_pooled_test.csv", low_memory=False)
    if in_training(d).any():
        raise ValueError("a pooled test variant is in the training data")
    d["label"] = d["label"].astype(int)
    d["MetaSplice"] = score(d) if with_scores else 0.0
    d["SpliceAI"], d["CADD"] = d.spliceai_max, d.cadd_phred
    return d


def test_cohorts(with_scores=True, kind="separate"):
    """kind='separate': {'development': April strict test set, 'september': fresh cohort}; kind='pooled':
    {'pooled': the Amendment 9 set}. All regions, with MetaSplice (full model), SpliceAI (precomputed masked, V08) and
    CADD (PHRED)."""
    if kind == "pooled":
        return {"pooled": pooled_cohort(with_scores)}
    dev = add_comparators(load_test("strict"))
    sep = load_fresh()
    out = {}
    for name, d in [("development", dev), ("september", sep)]:
        d = d.reset_index(drop=True)
        d["MetaSplice"] = score(d) if with_scores else 0.0
        d["SpliceAI"], d["CADD"] = d.spliceai_max, d.cadd_phred
        out[name] = d
    return out


def check_denominators(log, cohorts):
    amend = "Amendment 9" if "pooled" in cohorts else "Amendment 8"
    log(f"\nDenominator check against {amend} (n, P/LP), before anything is computed")
    bad = []
    for name, d in cohorts.items():
        nc = d[d.region.isin(NONCANONICAL)]
        for tag, x in [("non-canonical", nc), ("non-canonical, SpliceAI-scored", nc[nc.SpliceAI.notna()])]:
            got = (len(x), int(x.label.sum()))
            ok = got == EXPECTED[(name, tag)]
            log(f"  {name:<12} {tag:<32} n={got[0]:>7,} P/LP={got[1]:>4}   recorded {EXPECTED[(name, tag)]}   {'ok' if ok else 'DIFFERS'}")
            bad += [(name, tag)] if not ok else []
    if bad:
        raise SystemExit(f"denominators differ from {amend}; nothing computed: {bad}")
    log(f"  every denominator equals {amend}")


def rows(d):
    """(label, boolean mask): pooled non-canonical; each non-canonical region with >= MIN_P P/LP; Deep Intronic and the
    deep-intronic sensitivity set (always reported together)."""
    nc = d.region.isin(NONCANONICAL).values
    out = [("non-canonical", nc)]
    for r in NONCANONICAL:
        m = (d.region == r).values
        if r == "Deep Intronic" or int(d.label.values[m].sum()) >= MIN_P:
            out.append((r, m))
        if r == "Deep Intronic":
            out.append(("Deep Intronic, sensitivity set", deep_intronic_sensitivity(d).values & m))
    return out


def load_vus_cohort(pgenes):
    """S5's VUS yield cohort (S5_clingen_calibration.load_yield, same logic; S5's module opens its log file when imported,
    so it is not imported): uncertain and conflicting variants of the 2026-04-15 release under the calibration-cohort
    filters, with MetaSplice (full model) and SpliceAI (precomputed masked). S5 logged n=61,016."""
    vus = pd.read_csv(f"{ROOT}/data/v2_features_vus.csv", low_memory=False)
    if in_training(vus).any():
        raise ValueError("a VUS is in the training data")
    vus = add_comparators(vus)
    vus["MetaSplice"], vus["SpliceAI"] = score(vus), vus.spliceai_max
    return vus[in_cohort(vus, pgenes)].reset_index(drop=True)
