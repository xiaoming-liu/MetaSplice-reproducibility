"""F1 -- Fresh temporal validation of MetaSplice v2.0 on ClinVar 2026-09-28 (protocol Amendment 5).

Pre-specified in Amendment 5 before any v2.0 score was computed on this cohort. The model (frozen v2.0 full model) and
every threshold are frozen; nothing is refit or changed afterwards. Thresholds are exploratory candidate thresholds
under the fallback prior 0.0441, not clinically validated.
  1. Primary, replication of Amendment 3's comparison on a fresh hold-out: non-canonical variants, auPRC of MetaSplice
     v2.0 against SpliceAI and against CADD on the variants both score; paired stratified bootstrap
     (common.paired_boot_diff, 1,000 iterations, seed 42) with Amendment 3's decision rule; a gene-clustered bootstrap CI
     reported beside it.
  2. Secondary, no verdict: each non-canonical region; Deep Intronic and its distance bands (<= 50, 51-500, > 500 bp);
     the O5 sensitivity set and its bands; with distance alone as a baseline. A row with fewer than 20 P/LP is
     inconclusive: it carries counts and false positives only -- B/LB given candidate PP3 at the frozen S6 thresholds,
     with each tool's scored B/LB as the denominator, and on the variants both tools score.
  3. Calibration: interval LRs with exact CIs and verdicts on the fresh >= 1-star calibration cohort (and all stars),
     MetaSplice v2.0 and SpliceAI at the frozen S6 thresholds and S11 reference-window thresholds. SpliceAI's fixed score
     bands (>= 0.2, between, <= 0.1; the ClinGen SVI cut-offs, derived for spliceogenicity) are a descriptive
     comparison only: counts and LRs per band, no evidence-strength labels and no verdicts.
The primary verdict follows the variant-level bootstrap; when it is "better" but the gene-clustered interval includes 0,
both are reported and the replication is called qualified.
Before anything is computed the script asserts the md5 of every input it reads (LOCKS: the model copy it loads, the
feature list, the fresh-feature file, the threshold tables and the cohort filters), then checks every denominator
against EXPECTED (Amendment 5), and stops on a difference. --check-only runs both checks without loading the model.
The script prints the protocol's SHA-256.

Writes results/F1_fresh_validation.txt and .tsv.
"""
import argparse, hashlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from common import (FEATURES, NONCANONICAL, OUT, Tee, auprc, deep_intronic_sensitivity, load_fresh, paired_boot_diff, score)
from calib import LEVELS, assign, in_cohort, interval_lr, load_thresholds, lr_ci, one_star, pathogenic_genes, verdicts

PROTOCOL = os.path.join(OUT, "..", "docs", "AJHG_COMPARISON_PROTOCOL.md")
COMPARATORS = [("SpliceAI", "spliceai_max"), ("CADD", "cadd_phred")]
BANDS = [("<= 50 bp", 0, 50), ("51-500 bp", 50, 500), ("> 500 bp", 500, np.inf)]
MIN_P = 20
LOCKS = {      # Amendment 5: md5 of every input F1 reads, asserted in both modes
    "models/metasplice_v2_full.joblib": "51c6e5d0db100b33b91907fa857cd9a7",
    "models/metasplice_v2_features.joblib": "8be31360848f2cb43c2b02850bd03815",
    "data/v2_features_fresh_20260928.csv": "f9cd3f227b0357a4c022f5d6814e4ab4",
    "results/S6_thresholds.tsv": "6a009ca0e0d5c837d7b1633c739ceb41",
    "results/S11_thresholds.tsv": "b269aafcfe17d1f6cdbcce71d15dbd2d",
    "data/v2_pathogenic_genes.csv": "c6738b75048c943e9a95f1545c34b752",
    "data/v2_variants_test_candidates.csv": "d9e70ef71b2c1ca149e4cf4ed56d59e0",
    "data/v2_variants_training.csv": "5f2a4b045d7e6113d44896babc0d0c47",
}
PP3 = [f"PP3_{l}" for l in LEVELS]
EXPECTED = {      # Amendment 5, from results/C1_fresh_cohorts.txt: {(subset, pairing): (n, P/LP)}
    ('all non-canonical', 'MetaSplice'): (14749, 275),
    ('all non-canonical', 'SpliceAI'): (14552, 273),
    ('all non-canonical', 'CADD'): (14749, 275),
    ('Donor Region', 'MetaSplice'): (1408, 135),
    ('Donor Region', 'SpliceAI'): (1382, 133),
    ('Donor Region', 'CADD'): (1408, 135),
    ('Acceptor Region', 'MetaSplice'): (2019, 65),
    ('Acceptor Region', 'SpliceAI'): (2001, 65),
    ('Acceptor Region', 'CADD'): (2019, 65),
    ('PPT Region', 'MetaSplice'): (3899, 26),
    ('PPT Region', 'SpliceAI'): (3840, 26),
    ('PPT Region', 'CADD'): (3899, 26),
    ('Branch Point Region', 'MetaSplice'): (204, 3),
    ('Branch Point Region', 'SpliceAI'): (203, 3),
    ('Branch Point Region', 'CADD'): (204, 3),
    ('Deep Intronic', 'MetaSplice'): (7219, 46),
    ('Deep Intronic', 'SpliceAI'): (7126, 46),
    ('Deep Intronic', 'CADD'): (7219, 46),
    ('Deep Intronic <= 50 bp', 'MetaSplice'): (5544, 4),
    ('Deep Intronic <= 50 bp', 'SpliceAI'): (5462, 4),
    ('Deep Intronic <= 50 bp', 'CADD'): (5544, 4),
    ('Deep Intronic 51-500 bp', 'MetaSplice'): (432, 14),
    ('Deep Intronic 51-500 bp', 'SpliceAI'): (430, 14),
    ('Deep Intronic 51-500 bp', 'CADD'): (432, 14),
    ('Deep Intronic > 500 bp', 'MetaSplice'): (1243, 28),
    ('Deep Intronic > 500 bp', 'SpliceAI'): (1234, 28),
    ('Deep Intronic > 500 bp', 'CADD'): (1243, 28),
    ('DI sensitivity set', 'MetaSplice'): (6039, 38),
    ('DI sensitivity set', 'SpliceAI'): (5957, 38),
    ('DI sensitivity set', 'CADD'): (6039, 38),
    ('DI sensitivity set <= 50 bp', 'MetaSplice'): (4626, 4),
    ('DI sensitivity set <= 50 bp', 'SpliceAI'): (4552, 4),
    ('DI sensitivity set <= 50 bp', 'CADD'): (4626, 4),
    ('DI sensitivity set 51-500 bp', 'MetaSplice'): (287, 11),
    ('DI sensitivity set 51-500 bp', 'SpliceAI'): (285, 11),
    ('DI sensitivity set 51-500 bp', 'CADD'): (287, 11),
    ('DI sensitivity set > 500 bp', 'MetaSplice'): (1126, 23),
    ('DI sensitivity set > 500 bp', 'SpliceAI'): (1120, 23),
    ('DI sensitivity set > 500 bp', 'CADD'): (1126, 23),
    ('calibration, all stars', 'MetaSplice'): (13194, 273),
    ('calibration, all stars', 'SpliceAI'): (13011, 271),
    ('calibration, all stars', 'CADD'): (13194, 273),
    ('calibration, >= 1 star', 'MetaSplice'): (13186, 265),
    ('calibration, >= 1 star', 'SpliceAI'): (13003, 263),
    ('calibration, >= 1 star', 'CADD'): (13186, 265),
}


def subsets(t):
    nc = t[t.region.isin(NONCANONICAL)].reset_index(drop=True)
    out = [("all non-canonical", nc)] + [(r, nc[nc.region == r]) for r in NONCANONICAL]
    for name, d in [("Deep Intronic", nc[nc.region == "Deep Intronic"]), ("DI sensitivity set", nc[deep_intronic_sensitivity(nc).values])]:
        if name != "Deep Intronic":
            out.append((name, d))
        a = d.dist_to_boundary.abs()
        out += [(f"{name} {b}", d[((a > lo) & (a <= hi)).values]) for b, lo, hi in BANDS]
    return out


def verify_locks(log):
    log("\nInput lock (md5), before anything is computed")
    bad = []
    for f, want in LOCKS.items():
        got = hashlib.md5(open(f"{OUT}/{f}", "rb").read()).hexdigest()
        log(f"  {got}  {f}  {'ok' if got == want else 'DIFFERS from ' + want}")
        if got != want:
            bad.append(f)
    if bad:
        raise SystemExit(f"inputs differ from Amendment 5's locks; nothing computed: {bad}")
    log("  every input equals its locked md5")


def check(log, subs, cal):
    log("\nDenominator check against Amendment 5 (n, P/LP), before any auPRC is computed")
    bad = []
    got = {}
    for name, d in subs + cal:
        for pairing, cols in [("MetaSplice", []), ("SpliceAI", ["spliceai_max"]), ("CADD", ["cadd_phred"])]:
            x = d[d[cols].notna().all(axis=1)] if cols else d
            got[(name, pairing)] = (len(x), int(x.label.sum()))
    for k, v in got.items():
        w = EXPECTED.get(k)
        ok = w == v
        log(f"  {k[0]:<36} {k[1]:<10} n={v[0]:>7,} P={v[1]:>5}   recorded {w}   {'ok' if ok else 'DIFFERS'}")
        if not ok:
            bad.append(k)
    if bad or set(got) != set(EXPECTED):
        raise SystemExit(f"denominators differ from Amendment 5; nothing computed: {bad[:5]}")
    log("  every denominator equals Amendment 5")


def gene_ci(y, s0, s1, genes, n_boot=1000, seed=42):
    codes, gidx = np.unique(genes, return_inverse=True)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))[gidx].astype(float)
        if (w * y).sum() < 1 or (w * (1 - y)).sum() < 1:
            continue
        diffs.append(average_precision_score(y, s0, sample_weight=w) - average_precision_score(y, s1, sample_weight=w))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    log = Tee(os.devnull if args.check_only else f"{OUT}/results/F1_fresh_validation.txt")
    log("=" * 100)
    log("F1 -- FRESH TEMPORAL VALIDATION OF METASPLICE v2.0, ClinVar 2026-09-28 (Amendment 5)" + ("  [check only]" if args.check_only else ""))
    log("Thresholds are exploratory candidate thresholds under the fallback prior 0.0441, not clinically validated.")
    log("=" * 100)
    log(f"protocol SHA-256 (live file): {hashlib.sha256(open(PROTOCOL, 'rb').read()).hexdigest()}" if os.path.exists(PROTOCOL)
        else "protocol SHA-256: not computed (docs/AJHG_COMPARISON_PROTOCOL.md is not present)")
    verify_locks(log)
    t = load_fresh()
    t["MetaSplice"] = 0.0 if args.check_only else score(t)
    t["SpliceAI"], t["CADD"], t["dist"] = t.spliceai_max, t.cadd_phred, t.dist_to_boundary.abs()
    subs = subsets(t)
    val = t[in_cohort(t, pathogenic_genes())]
    cal = [("calibration, all stars", val), ("calibration, >= 1 star", val[one_star(val)])]
    check(log, subs, cal)
    if args.check_only:
        return
    rows = []
    s6 = {k: load_thresholds(k, path=f"{OUT}/results/S6_thresholds.tsv") for k in ["MetaSplice", "SpliceAI"]}

    def verdict(lo, hi, glo, ghi):
        v = "MetaSplice v2.0 better" if lo > 0 else ("MetaSplice v2.0 worse" if hi < 0 else "indistinguishable")
        if v != "indistinguishable" and glo <= 0 <= ghi:
            v += " by the variant-level bootstrap; the gene-clustered interval includes 0 (qualified)"
        return v

    log("\n1. PRIMARY: non-canonical, each comparator on the variants both score (replication of Amendment 3's comparison)")
    for tag, d in subs:
        y = d.label.values
        if int(y.sum()) < MIN_P:
            pp = {k: np.isin(assign(d[k].values.astype(float), *s6[k]), PP3) for k in ["MetaSplice", "SpliceAI"]}
            scored = {k: d[k].notna().values for k in ["MetaSplice", "SpliceAI"]}
            both = scored["MetaSplice"] & scored["SpliceAI"]
            row = dict(section="2", subset=tag, n=len(d), P=int(y.sum()), note=f"inconclusive (< {MIN_P} P/LP)")
            cells = []
            for k in ["MetaSplice", "SpliceAI"]:
                b = scored[k] & (y == 0)
                row[f"FP_{k}"], row[f"B_scored_{k}"] = int((pp[k] & b).sum()), int(b.sum())
                row[f"FP_{k}_shared"] = int((pp[k] & both & (y == 0)).sum())
                cells.append(f"{k} {row[f'FP_{k}']} of {row[f'B_scored_{k}']:,} scored B/LB")
            row["B_shared"] = int((both & (y == 0)).sum())
            log(f"  {tag:<36} n={len(d):,} P/LP={int(y.sum())}: INCONCLUSIVE (< {MIN_P} P/LP). B/LB given candidate PP3 at S6 thresholds: "
                + "; ".join(cells) + f"; on the {row['B_shared']:,} B/LB both score: MetaSplice {row['FP_MetaSplice_shared']}, "
                f"SpliceAI {row['FP_SpliceAI_shared']}")
            rows.append(row)
            continue
        if tag == "all non-canonical":
            log("")
        elif tag == NONCANONICAL[0]:
            log("\n2. SECONDARY (no verdict): regions, Deep Intronic bands, sensitivity set; distance alone as a baseline")
        for name, col in COMPARATORS + [("distance only", "dist")]:
            ok = d[col].notna().values
            yy = y[ok]
            diff, lo, hi, n = paired_boot_diff(y, d.MetaSplice.values, d[col].values.astype(float))
            glo, ghi = gene_ci(yy, d.MetaSplice.values[ok], d[col].values[ok].astype(float), d.gene_id.values[ok])
            v = verdict(lo, hi, glo, ghi) if tag == "all non-canonical" and name != "distance only" else ""
            rows.append(dict(section="1" if tag == "all non-canonical" else "2", subset=tag, comparator=name, n=n, P=int(yy.sum()),
                             auprc_metasplice=auprc(yy, d.MetaSplice.values[ok]), auprc_comparator=auprc(yy, d[col].values[ok].astype(float)),
                             diff=diff, lo=lo, hi=hi, gene_lo=glo, gene_hi=ghi, verdict=v))
            log(f"  {tag:<36} vs {name:<13} n={n:>6,} P={int(yy.sum()):>4}  {rows[-1]['auprc_metasplice']:.4f} vs {rows[-1]['auprc_comparator']:.4f}  "
                f"diff {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]  gene-clustered [{glo:+.4f}, {ghi:+.4f}]" + (f"  -> {v}" if v else ""))

    log("\n3. CALIBRATION: frozen thresholds, interval LR with exact CI (thresholds not changed afterwards)")
    s11 = pd.read_csv(f"{OUT}/results/S11_thresholds.tsv", sep="\t")
    def s11_thr(tool):
        r = s11[(s11.tool == tool) & (s11.config == "reference")]
        pp3 = {l: (None if pd.isna(v) else float(v)) for l, v in zip(r.level, r.score_threshold) if l.startswith("PP3")}
        bp4 = {l: (None if pd.isna(v) else float(v)) for l, v in zip(r.level, r.score_threshold) if l.startswith("BP4")}
        return {k[4:]: v for k, v in pp3.items()}, {k[4:]: v for k, v in bp4.items()}
    for cname, d in cal:
        y = d.label.values
        log(f"\n  {cname}: n={len(d):,}, P/LP {int(y.sum())}, B/LB {int((y == 0).sum()):,}")
        schemes = [("MetaSplice v2.0, S6 thresholds", assign(d.MetaSplice.values, *s6["MetaSplice"])),
                   ("MetaSplice v2.0, S11 reference-window thresholds", assign(d.MetaSplice.values, *s11_thr("MetaSplice"))),
                   ("SpliceAI, S6 thresholds", assign(d.SpliceAI.values.astype(float), *s6["SpliceAI"])),
                   ("SpliceAI, S11 reference-window thresholds", assign(d.SpliceAI.values.astype(float), *s11_thr("SpliceAI")))]
        for label, codes in schemes:
            log(f"    {label}")
            for code, kp, kb, lr, lo, hi in interval_lr(codes, y):
                point, ci, ns = verdicts(code, lr, lo, hi)
                lrs, his = ("inf" if np.isinf(lr) else f"{lr:.3g}"), ("inf" if np.isinf(hi) else f"{hi:.3g}")
                log(f"      {code:<16}{kp:>5}{kb:>7}{lrs:>10}   [{lo:.3g}, {his}]".ljust(70) + f"{ns:<10}{point:<7}{ci}")
                rows.append(dict(section="3", subset=cname, scheme=label, interval=code, P=kp, B=kb, LR=lr, lo=lo, hi=hi,
                                 point_verdict=point, ci_verdict=ci))
        s = d.SpliceAI.values.astype(float)
        NP, NB = int((y == 1).sum()), int((y == 0).sum())
        log("    SpliceAI, fixed score bands at 0.2 and 0.1 (the ClinGen SVI cut-offs, derived for spliceogenicity): a descriptive"
            "\n    comparison only -- no evidence-strength labels and no verdicts for pathogenicity on this cohort")
        for band, m in [("score >= 0.2", s >= 0.2), ("0.1 < score < 0.2", (s > 0.1) & (s < 0.2)), ("score <= 0.1", s <= 0.1),
                        ("no score", np.isnan(s))]:
            kp, kb = int(m[y == 1].sum()), int(m[y == 0].sum())
            if kp + kb == 0:
                continue
            lr, lo, hi = lr_ci(kp, NP, kb, NB)
            lrs, his = ("inf" if np.isinf(lr) else f"{lr:.3g}"), ("inf" if np.isinf(hi) else f"{hi:.3g}")
            log(f"      {band:<18}{kp:>5}{kb:>7}{lrs:>10}   [{lo:.3g}, {his}]")
            rows.append(dict(section="3", subset=cname, scheme="SpliceAI fixed score bands (descriptive)", interval=band, P=kp, B=kb,
                             LR=lr, lo=lo, hi=hi))
    pd.DataFrame(rows).to_csv(f"{OUT}/results/F1_fresh_validation.tsv", sep="\t", index=False)
    log(f"\nwrote results/F1_fresh_validation.tsv ({len(rows)} rows)")
    log.close()


if __name__ == "__main__":
    main()
