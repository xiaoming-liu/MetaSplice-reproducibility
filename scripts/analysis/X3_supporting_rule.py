"""X3 -- EXPLORATORY: a conservative PP3/BP4 rule for MetaSplice against ClinGen's SpliceAI rule (Amendment 6).

The rule (decided 2026-10-08, after both cohorts' interval counts were logged; post hoc):
  PP3 at Supporting for score >= S6's PP3 Strong threshold (0.544); BP4 at Supporting for score <= S6's BP4 Moderate
  threshold (0.0017); neither code in between.
The comparator is the ClinGen SVI Splicing Subgroup's SpliceAI rule (Walker et al. 2023): PP3 at >= 0.2, BP4 at <= 0.1;
variants SpliceAI does not score receive no code.

Cohorts: the calibration cohorts of the April 2026 strict test set (development) and of the September 2026 cohort
(validation); >= 1 star primary, all review statuses secondary. For each rule: the share of P/LP and of B/LB given PP3,
given BP4 and given neither; the interval LR with its exact 95% CI (calib.lr_ci) for the variants given PP3 and for those
given BP4, and whether that CI clears the Supporting requirement at priors 0.0441 (c = 1124), 0.01 and 0.1. Paired
differences (MetaSplice minus SpliceAI) in the four shares, on all cohort variants (primary) and on the variants
SpliceAI scores (secondary): class-stratified bootstrap (1,000 iterations, seed 42, percentile 95% CI) with a
gene-clustered bootstrap CI beside it. No verdict.

Before anything new is computed the script asserts the md5 of every input it reads (the two 2025-08-31 ClinVar
summaries that common.training_release() reads were added after the first run, with V00's md5s), the four cohort denominators and
every interval count already logged in S6 and F1 (Amendment 6), and stops on a difference. --check-only runs those checks
and computes nothing else. It prints the protocol's SHA-256.

--spliceai raw (Amendment 7) replaces the precomputed masked SpliceAI scores (+-50 nt) with SpliceAI 1.3.1 raw scores over
+-4,999 nt (V11, data/v2_spliceai_raw_d4999.csv), the configuration Walker et al. calibrated; everything else is unchanged. --spliceai raw_allgenes uses the maximum over
all of the allele's SpliceAI records instead of the first record (a sensitivity analysis added before the comparison,
because SpliceAI's record order differs from the precomputed files' at overlapping genes).

Writes results/X3_supporting_rule.txt and .tsv (masked) or results/X3_supporting_rule_raw.txt and .tsv (raw).
"""
import argparse, hashlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import CLINVAR_2025, OUT, Tee, load_fresh, score
from calib import in_cohort, load_thresholds, load_validation_cohort, lr_ci, lr_levels, one_star, pathogenic_genes

PROTOCOL = os.path.join(OUT, "..", "docs", "AJHG_COMPARISON_PROTOCOL.md")
LOCKS = {      # Amendment 6: md5 of every input X3 reads (values recorded by F1 and C0)
    "models/metasplice_v2_full.joblib": "51c6e5d0db100b33b91907fa857cd9a7",
    "models/metasplice_v2_features.joblib": "8be31360848f2cb43c2b02850bd03815",
    "results/S6_thresholds.tsv": "6a009ca0e0d5c837d7b1633c739ceb41",
    "data/v2_features_fresh_20260928.csv": "f9cd3f227b0357a4c022f5d6814e4ab4",
    "data/v2_features_test_candidates.csv": "27fad032309f58f4b058b4c6e1ddc4a9",
    "data/v2_variants_test_candidates.csv": "d9e70ef71b2c1ca149e4cf4ed56d59e0",
    "data/v2_variants_training.csv": "5f2a4b045d7e6113d44896babc0d0c47",
    "data/v2_comparators.csv": "fc9013f3b52502ca25dc1d0ff774b7b8",
    "data/v2_pathogenic_genes.csv": "c6738b75048c943e9a95f1545c34b752",
    # read by common.training_release() for the development cohort; md5s as recorded by V00 (added after the first run)
    CLINVAR_2025["GRCh38"]: "111c595ba957c7fc93b8833188d4b029",
    CLINVAR_2025["GRCh37"]: "3c0116eaa075ee7c658ca5c70ec44763",
}
_S6 = load_thresholds("MetaSplice", path=f"{OUT}/results/S6_thresholds.tsv")
T_PP3, T_BP4 = _S6[0]["Strong"], _S6[1]["Moderate"]             # S6, MetaSplice: PP3 Strong and BP4 Moderate, read as F1 reads them
DOCUMENTED = (0.5439710286928215, 0.0017023810017853637)        # Amendment 6
SAI_PP3, SAI_BP4 = 0.2, 0.1                                     # Walker et al. 2023
COHORTS = ["development, >= 1 star", "validation, >= 1 star", "development, all statuses", "validation, all statuses"]
EXPECTED_N = {"development, >= 1 star": (24049, 231), "validation, >= 1 star": (13186, 265),
              "development, all statuses": (24057, 239), "validation, all statuses": (13194, 273)}
EXPECTED_COUNTS = {   # (cohort, rule, code): (P/LP, B/LB), from S6_review_status.txt and F1_fresh_validation.txt
    ("development, >= 1 star", "MetaSplice", "PP3"): (200, 13),
    ("development, >= 1 star", "MetaSplice", "BP4"): (7, 21183),
    ("validation, >= 1 star", "MetaSplice", "PP3"): (228, 55),
    ("validation, >= 1 star", "MetaSplice", "BP4"): (5, 10999),
    ("validation, all statuses", "MetaSplice", "PP3"): (235, 55),
    ("validation, all statuses", "MetaSplice", "BP4"): (6, 10999),
    ("validation, >= 1 star", "SpliceAI", "PP3"): (237, 279),
    ("validation, >= 1 star", "SpliceAI", "BP4"): (18, 12142),
    ("validation, all statuses", "SpliceAI", "PP3"): (243, 279),
    ("validation, all statuses", "SpliceAI", "BP4"): (20, 12142),
}
RAW = "data/v2_spliceai_raw_d4999.csv"
RAW_MD5 = "31b81cc8553720bc33ed7a39e8636206"     # Amendment 7: added after V11 wrote the file, before the comparison
PRIORS = [("0.0441", lr_levels(0.0441, 1124.0)[1]), ("0.01", lr_levels(0.01)[1]), ("0.1", lr_levels(0.1)[1])]
SHARES = [("P/LP given PP3", 1, "PP3"), ("B/LB given PP3", 0, "PP3"), ("B/LB given BP4", 0, "BP4"),
          ("P/LP given BP4", 1, "BP4")]


def verify_locks(log, raw=False):
    log("\nInput lock (md5), before anything is computed")
    bad = []
    for f, want in (dict(LOCKS, **{RAW: RAW_MD5}) if raw else LOCKS).items():
        got = hashlib.md5(open(f if os.path.isabs(f) else f"{OUT}/{f}", "rb").read()).hexdigest()
        log(f"  {got}  {f}  {'ok' if got == want else 'DIFFERS from ' + want}")
        bad += [f] if got != want else []
    if bad:
        raise SystemExit(f"inputs differ from Amendment 6's locks; nothing computed: {bad}")
    if not np.allclose((T_PP3, T_BP4), DOCUMENTED, rtol=1e-12, atol=0):
        raise SystemExit(f"S6's thresholds {(T_PP3, T_BP4)} differ from Amendment 6's {DOCUMENTED}")
    log("  every input equals its locked md5; the rule's thresholds equal S6's PP3 Strong and BP4 Moderate")


def cohorts(raw=False, col="spliceai_raw_max"):
    dev_all = load_validation_cohort("strict")
    dev = dev_all[one_star(dev_all)].reset_index(drop=True)
    f = load_fresh()
    val_all = f[in_cohort(f, pathogenic_genes())].reset_index(drop=True)
    val_all["MetaSplice"], val_all["SpliceAI"] = score(val_all), val_all.spliceai_max
    val = val_all[one_star(val_all)].reset_index(drop=True)
    out = dict(zip(COHORTS, [dev, val, dev_all, val_all]))
    if raw:
        r = pd.read_csv(f"{OUT}/{RAW}", usecols=["chr", "pos", "ref", "alt", col], low_memory=False)
        for k, d in out.items():
            m = d[["chr", "pos", "ref", "alt"]].merge(r, on=["chr", "pos", "ref", "alt"], how="left", validate="one_to_one")
            d = d.copy()
            d["SpliceAI"] = m[col].values     # unscored stays NaN and receives no code
            out[k] = d
    return out


def codes(d):
    """Evidence code under each rule: 'PP3', 'BP4' or '' (none; also an unscored variant)."""
    out = {}
    for rule, col, hi, lo in [("MetaSplice", "MetaSplice", T_PP3, T_BP4), ("SpliceAI", "SpliceAI", SAI_PP3, SAI_BP4)]:
        s = d[col].values.astype(float)
        c = np.full(len(s), "", dtype=object)
        c[s >= hi] = "PP3"            # NaN compares False: an unscored variant gets no code
        c[s <= lo] = "BP4"
        out[rule] = c
    return out


def check(log, data, raw=False):
    log("\nChecks against Amendment 6, before anything new is computed")
    bad = []
    for name in COHORTS:
        d = data[name]
        got = (len(d), int(d.label.sum()))
        ok = got == EXPECTED_N[name]
        log(f"  {name:<28} n={got[0]:>7,} P/LP={got[1]:>4}   recorded {EXPECTED_N[name]}   {'ok' if ok else 'DIFFERS'}")
        bad += [name] if not ok else []
        if d.MetaSplice.isna().any():
            bad.append(f"{name}: MetaSplice missing")
    for (name, rule, code), want in EXPECTED_COUNTS.items():
        if raw and rule == "SpliceAI":
            continue                 # logged counts are for the masked scores
        d = data[name]
        c = codes(d)[rule]
        got = (int(((c == code) & (d.label == 1)).sum()), int(((c == code) & (d.label == 0)).sum()))
        ok = got == want
        log(f"  {name:<28} {rule:<10} {code}  P/LP {got[0]:>4}  B/LB {got[1]:>6,}   recorded {want}   {'ok' if ok else 'DIFFERS'}")
        bad += [(name, rule, code)] if not ok else []
    if bad:
        raise SystemExit(f"checks differ from Amendment 6; nothing computed: {bad}")
    log("  every denominator and every logged interval count is reproduced")


def boot_diffs(y, a, b, genes, n_boot=1000, seed=42):
    """Bootstrap percentile CIs for share(a) - share(b) within class y==cls, for both classes at once.

    a, b: boolean arrays (code given by each rule). Returns {cls: ((lo, hi) stratified, (lo, hi) gene-clustered)}."""
    rng = np.random.default_rng(seed)
    out = {}
    for cls in (1, 0):
        idx = np.where(y == cls)[0]
        da = a[idx].astype(float) - b[idx].astype(float)
        diffs = [da[rng.choice(len(idx), len(idx), replace=True)].mean() for _ in range(n_boot)]
        strat = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
        out[cls] = [strat]
    # gene-clustered: resample genes; shares within each class over the resampled variants
    g_codes, g_idx = np.unique(genes, return_inverse=True)
    G = len(g_codes)
    sums = {}
    for cls in (1, 0):
        m = (y == cls)
        sums[cls] = (np.bincount(g_idx[m], minlength=G).astype(float),
                     np.bincount(g_idx[m], weights=a[m].astype(float), minlength=G),
                     np.bincount(g_idx[m], weights=b[m].astype(float), minlength=G))
    rng = np.random.default_rng(seed)
    gd = {1: [], 0: []}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
        for cls in (1, 0):
            n, sa, sb = sums[cls]
            tot = (w * n).sum()
            gd[cls].append(((w * sa).sum() - (w * sb).sum()) / tot if tot else np.nan)
    for cls in (1, 0):
        v = np.array(gd[cls])
        out[cls].append((float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--spliceai", choices=["masked", "raw", "raw_allgenes"], default="masked")
    args = ap.parse_args()
    raw = args.spliceai != "masked"
    col = "spliceai_raw_max_allgenes" if args.spliceai == "raw_allgenes" else "spliceai_raw_max"
    stem = "X3_supporting_rule" + {"masked": "", "raw": "_raw", "raw_allgenes": "_raw_allgenes"}[args.spliceai]
    log = Tee(os.devnull if args.check_only else f"{OUT}/results/{stem}.txt")
    log("=" * 104)
    log("X3 -- EXPLORATORY: A CONSERVATIVE PP3/BP4 RULE FOR METASPLICE AGAINST CLINGEN'S SPLICEAI RULE (Amendment 6)"
        + ("  [check only]" if args.check_only else ""))
    log(f"MetaSplice: PP3 at Supporting for score >= {T_PP3:.3g}; BP4 at Supporting for score <= {T_BP4:.3g} (S6 thresholds; "
        f"rule chosen after the interval counts were seen)")
    log(f"SpliceAI (Walker et al. 2023): PP3 at >= {SAI_PP3}; BP4 at <= {SAI_BP4}; unscored variants receive no code; scores: "
        + ((f"SpliceAI 1.3.1 raw, +-4,999 nt (V11; the calibrated configuration, Amendment 7); value {col}") if raw
           else "precomputed masked, +-50 nt (V08)"))
    log("=" * 104)
    log(f"protocol SHA-256 (live file): {hashlib.sha256(open(PROTOCOL, 'rb').read()).hexdigest()}" if os.path.exists(PROTOCOL)
        else "protocol SHA-256: not computed (docs/AJHG_COMPARISON_PROTOCOL.md is not present)")
    verify_locks(log, raw)
    data = cohorts(raw, col)
    check(log, data, raw)
    if args.check_only:
        return

    rows = []
    for name in COHORTS:
        d = data[name]
        y = d.label.values
        NP, NB = int(y.sum()), int((y == 0).sum())
        c = codes(d)
        log(f"\n{name}: n={len(d):,} (P/LP {NP}, B/LB {NB:,}); SpliceAI scores {int(d.SpliceAI.notna().sum()):,}")
        log(f"  {'rule':<11}{'P/LP PP3':>16}{'B/LB PP3':>16}{'B/LB BP4':>18}{'P/LP BP4':>14}{'P/LP none':>14}{'B/LB none':>16}")
        for rule in ["MetaSplice", "SpliceAI"]:
            k = {(cls, code): int(((c[rule] == code) & (y == cls)).sum()) for cls in (1, 0) for code in ("PP3", "BP4", "")}
            sh = lambda cls, code: k[(cls, code)] / (NP if cls else NB)
            log(f"  {rule:<11}{k[(1,'PP3')]:>6} ({sh(1,'PP3'):6.1%}){k[(0,'PP3')]:>7,} ({sh(0,'PP3'):5.2%})"
                f"{k[(0,'BP4')]:>8,} ({sh(0,'BP4'):6.1%}){k[(1,'BP4')]:>4} ({sh(1,'BP4'):5.1%})"
                f"{k[(1,'')]:>4} ({sh(1,''):5.1%}){k[(0,'')]:>7,} ({sh(0,''):5.1%})")
            for code in ("PP3", "BP4"):
                lr, lo, hi = lr_ci(k[(1, code)], NP, k[(0, code)], NB)
                clears = []
                for pname, need in PRIORS:
                    req = need["Supporting"]
                    clears.append(f"prior {pname}: " + ("clears" if (lo >= req if code == "PP3" else hi <= 1 / req) else "INCONCLUSIVE"))
                his = "inf" if hi == np.inf else f"{hi:.3g}"
                log(f"      {code} variants: LR {lr:.3g} [{lo:.3g}, {his}]   Supporting CI: " + "; ".join(clears))
                rows.append(dict(cohort=name, rule=rule, code=code, P=k[(1, code)], B=k[(0, code)], NP=NP, NB=NB,
                                 LR=lr, lo95_exact=lo, hi95_exact=hi, **{f"supporting_prior_{p}": v.split(": ")[1]
                                                                        for p, v in zip([p for p, _ in PRIORS], clears)}))
            rows.append(dict(cohort=name, rule=rule, code="none", P=k[(1, "")], B=k[(0, "")], NP=NP, NB=NB))
        for scope, m in [("all cohort variants", np.ones(len(d), bool)), ("variants SpliceAI scores", d.SpliceAI.notna().values)]:
            ys, genes = y[m], d.gene_id.values[m]
            log(f"  MetaSplice minus SpliceAI, {scope} (n={int(m.sum()):,}; P/LP {int(ys.sum())}): "
                "share difference [stratified 95% CI] [gene-clustered 95% CI]")
            for label, cls, code in SHARES:
                a, b = (c["MetaSplice"][m] == code), (c["SpliceAI"][m] == code)
                sel = ys == cls
                diff = a[sel].mean() - b[sel].mean()
                (slo, shi), (glo, ghi) = boot_diffs(ys, a, b, genes)[cls]
                log(f"      {label:<16}{diff:+8.2%}  [{slo:+.2%}, {shi:+.2%}]  [{glo:+.2%}, {ghi:+.2%}]")
                rows.append(dict(cohort=name, rule="MetaSplice minus SpliceAI", code=label, scope=scope, n=int(m.sum()),
                                 diff=diff, lo=slo, hi=shi, gene_lo=glo, gene_hi=ghi))
    pd.DataFrame(rows).to_csv(f"{OUT}/results/{stem}.tsv", sep="\t", index=False)
    log(f"\nwrote results/{stem}.tsv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
