"""Q1 -- Discrimination on the pooled temporal test set (protocol Amendment 9; a pooled reassessment, descriptive, no
verdict).

Pooled set (V12): in-scope SNVs classified P/LP or B/LB in ClinVar 2026-09-28, not in the training data and not classified
P/LP or B/LB in the 2025-08-31 training release; labels from 2026-09-28.
  1. MetaSplice, SpliceAI (precomputed masked, V08) and CADD v1.7 PHRED: auPRC with a stratified bootstrap CI, AUROC and
     coverage, each tool on the variants it scores; MetaSplice minus each comparator on the variants both score, with the
     paired stratified bootstrap (common.paired_boot_diff, 1,000 iterations, seed 42) and F1's gene-clustered bootstrap.
     Rows: all non-exonic; canonical; non-canonical (main); each non-canonical region; Deep Intronic and the deep-intronic
     sensitivity set. A non-canonical regional row with fewer than 20 P/LP carries counts only (Amendment 5's rule).
  2. Gene-disjoint (A3b's design): five gene folds over the union of training and pooled genes (seed 42); each pooled
     variant is scored by a model trained without its gene. The full model minus the gene-disjoint model, and the
     gene-disjoint model minus SpliceAI and minus CADD.
  3. Supplementary breakdown by origin: "April-origin variants, classified as of September 2026" and "September-origin
     variants" (the April-origin component includes reclassified variants and 2026-09-28 labels and review statuses; it does
     not reproduce the original April strict-set analysis).
  4. Precision-recall curve points for the figure (non-canonical and all non-exonic).
Inputs and denominators are asserted first; --check-only stops there.

Writes results/Q1_pooled_comparison.txt, .tsv and results/Q1_pooled_pr_curves.tsv.
"""
import argparse, importlib.util, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score
from common import CANONICAL, FEATURES, NONCANONICAL, OUT, Tee, boot_ci, deep_intronic_sensitivity, load_train, paired_boot_diff
from paper_cohorts import pooled_cohort, verify_locks
from F1_fresh_validation import gene_ci

_spec = importlib.util.spec_from_file_location("v06", os.path.join(HERE, "..", "build", "V06_train.py"))
v06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v06)

LOCK_FILES = ["models/metasplice_v2_full.joblib", "models/metasplice_v2_features.joblib", "data/v2_features_pooled_test.csv",
              "data/v2_features_training.csv", "data/v2_oof.csv", "data/v2_variants_training.csv"]
EXPECTED = {   # Amendment 9: (n, P/LP), (SpliceAI-scored n, P/LP)
    "non-canonical": ((40619, 602), (40296, 600)), "Donor Region": ((3314, 283), (3278, 281)),
    "Acceptor Region": ((7021, 123), (6983, 123)), "PPT Region": ((12410, 61), (12310, 61)),
    "Branch Point Region": ((395, 9), (394, 9)), "Deep Intronic": ((17479, 126), (17331, 126)),
    "Deep Intronic, sensitivity set": ((14459, 99), (14325, 99)), "Canonical Donor Site": ((2951, 2932), (2937, 2918)),
    "Canonical Acceptor Site": ((2542, 2522), (2516, 2497)), "all non-exonic": ((46112, 6056), (45749, 6015)),
}
MIN_P = 20
COMPARATORS = [("SpliceAI", "SpliceAI"), ("CADD", "CADD")]


def masks(d):
    nc = d.region.isin(NONCANONICAL).values
    out = [("all non-exonic", np.ones(len(d), bool)), ("canonical", d.region.isin(CANONICAL).values), ("non-canonical", nc)]
    for r in NONCANONICAL:
        out.append((r, (d.region == r).values))
        if r == "Deep Intronic":
            out.append(("Deep Intronic, sensitivity set", deep_intronic_sensitivity(d).values))
    return out


def check(log, d):
    log("\nDenominator check against Amendment 9 (n, P/LP; SpliceAI-scored n, P/LP), before anything is computed")
    mk = dict(masks(d))
    mk["Canonical Donor Site"] = (d.region == "Canonical Donor Site").values
    mk["Canonical Acceptor Site"] = (d.region == "Canonical Acceptor Site").values
    bad = []
    for name, (w1, w2) in EXPECTED.items():
        m = mk[name]
        s = m & d.SpliceAI.notna().values
        got = ((int(m.sum()), int(d.label.values[m].sum())), (int(s.sum()), int(d.label.values[s].sum())))
        ok = got == (w1, w2)
        log(f"  {name:<32} {got[0]}  SpliceAI {got[1]}   recorded {w1} {w2}   {'ok' if ok else 'DIFFERS'}")
        bad += [name] if not ok else []
    if bad:
        raise SystemExit(f"denominators differ from Amendment 9; nothing computed: {bad}")
    log("  every denominator equals Amendment 9")


def evaluate(log, d, title, tsv, extra=None, gene=True):
    """Marginal auPRC per tool and paired MetaSplice-minus-comparator differences for each row of d."""
    log(f"\n{title}")
    cols = [("MetaSplice", d.MetaSplice.values)] + [(lab, d[c].values.astype(float)) for lab, c in COMPARATORS] + (extra or [])
    for tag, m in masks(d):
        y = d.label.values[m]
        P = int(y.sum())
        if tag in NONCANONICAL and P < MIN_P:
            log(f"  {tag:<32} n={int(m.sum()):,} P/LP={P}: counts only (< {MIN_P} P/LP)")
            tsv.append(dict(section=title, row=tag, n=int(m.sum()), P=P, note=f"counts only (< {MIN_P} P/LP)"))
            continue
        log(f"  {tag:<32} n={int(m.sum()):,} P/LP={P}")
        for lab, s in cols:
            s = np.asarray(s, float)[m]
            ok = ~np.isnan(s)
            a, lo, hi = boot_ci(y[ok], s[ok])
            au = roc_auc_score(y[ok], s[ok]) if 0 < y[ok].sum() < ok.sum() else np.nan
            log(f"      {lab:<22} cov {ok.mean():6.1%}  auPRC {a:.4f} [{lo:.4f}, {hi:.4f}]  AUROC {au:.4f}")
            tsv.append(dict(section=title, row=tag, tool=lab, n=int(m.sum()), P=P, coverage=float(ok.mean()), auprc=a, lo=lo,
                            hi=hi, auroc=au))
        for lab, s in cols[1:]:
            s = np.asarray(s, float)[m]
            ms = d.MetaSplice.values[m]
            dd, dlo, dhi, n = paired_boot_diff(y, ms, s)
            both = ~np.isnan(s)
            glo, ghi = gene_ci(y[both], ms[both], s[both], d.gene_id.values[m][both]) if gene else (np.nan, np.nan)
            log(f"      MetaSplice minus {lab:<22} n={n:,} P/LP={int(y[both].sum())}  {dd:+.4f} [{dlo:+.4f}, {dhi:+.4f}]"
                + (f"  gene-clustered [{glo:+.4f}, {ghi:+.4f}]" if gene else ""))
            tsv.append(dict(section=title, row=tag, tool=f"MetaSplice minus {lab}", n=n, P=int(y[both].sum()), diff=dd,
                            lo=dlo, hi=dhi, gene_lo=glo, gene_hi=ghi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    log = Tee(os.devnull if args.check_only else f"{OUT}/results/Q1_pooled_comparison.txt")
    log("=" * 104)
    log("Q1 -- DISCRIMINATION ON THE POOLED TEMPORAL TEST SET (Amendment 9; pooled reassessment, descriptive, no verdict)"
        + ("  [check only]" if args.check_only else ""))
    log("=" * 104)
    verify_locks(log, LOCK_FILES)
    d = pooled_cohort(with_scores=not args.check_only)
    check(log, d)
    if args.check_only:
        return
    tsv = []
    evaluate(log, d, "1. POOLED SET: each tool on the variants it scores; paired differences on the variants both score", tsv)

    # 2. gene-disjoint
    train = load_train()
    genes = sorted(set(train.gene_id.dropna()) | set(d.gene_id.dropna()))
    fold_of = {g: int(f) for g, f in zip(genes, np.random.default_rng(42).integers(0, 5, len(genes)))}
    trf, tef = train.gene_id.map(fold_of).values, d.gene_id.map(fold_of).values
    gd = np.full(len(d), np.nan)
    for f in range(5):
        idx = np.where(tef == f)[0]
        m = v06.fit_one(train[trf != f][FEATURES].astype(np.float32).values, train[trf != f].label.astype(int).values)
        gd[idx] = m.predict_proba(d.iloc[idx][FEATURES].astype(np.float32).values)[:, 1]
    if np.isnan(gd).any():
        raise ValueError("a pooled variant has no gene-disjoint score")
    d["gene_disjoint"] = gd
    log(f"\n2. GENE-DISJOINT ({len(genes):,} genes in 5 folds, seed 42): full model minus gene-disjoint; gene-disjoint minus comparators")
    for tag, m in masks(d):
        y = d.label.values[m]
        if tag in NONCANONICAL and y.sum() < MIN_P:
            continue
        rec = dict(section="2. gene-disjoint", row=tag, n=int(m.sum()), P=int(y.sum()))
        a, lo, hi = boot_ci(y, gd[m])
        parts = [f"gene-disjoint auPRC {a:.4f} [{lo:.4f}, {hi:.4f}]"]
        rec.update(auprc=a, lo=lo, hi=hi)
        for lab, ref, sign in [("full minus gene-disjoint", d.MetaSplice.values, 1), ("gene-disjoint minus SpliceAI", d.SpliceAI.values, -1),
                               ("gene-disjoint minus CADD", d.CADD.values, -1)]:
            s0, s1 = (ref[m], gd[m]) if sign == 1 else (gd[m], ref[m].astype(float))
            dd, dlo, dhi, n = paired_boot_diff(y, s0, s1)
            parts.append(f"{lab} {dd:+.4f} [{dlo:+.4f}, {dhi:+.4f}] (n={n:,})")
            rec.update({f"{lab} diff": dd, f"{lab} lo": dlo, f"{lab} hi": dhi})
        log(f"  {tag:<32} (P/LP {int(y.sum())}) " + "; ".join(parts))
        tsv.append(rec)

    # 3. by origin (supplement)
    for src, lab in [("2026-04-15", "April-origin variants, classified as of September 2026"), ("2026-09-28", "September-origin variants")]:
        evaluate(log, d[d.source == src].reset_index(drop=True), f"3. SUPPLEMENT -- {lab}", tsv)

    # 4. precision-recall curve points
    pr = []
    for tag, m in masks(d)[:3]:
        if tag == "canonical":
            continue
        for lab, col in [("MetaSplice", "MetaSplice"), ("SpliceAI", "SpliceAI"), ("CADD", "CADD")]:
            s = d[col].values[m].astype(float)
            ok = ~np.isnan(s)
            prec, rec_, _ = precision_recall_curve(d.label.values[m][ok], s[ok])
            pr += [dict(row=tag, tool=lab, precision=p_, recall=r_, coverage=float(ok.mean())) for p_, r_ in zip(prec, rec_)]
    pd.DataFrame(pr).to_csv(f"{OUT}/results/Q1_pooled_pr_curves.tsv", sep="\t", index=False)
    pd.DataFrame(tsv).to_csv(f"{OUT}/results/Q1_pooled_comparison.tsv", sep="\t", index=False)
    log(f"\nwrote results/Q1_pooled_comparison.tsv ({len(tsv)} rows) and results/Q1_pooled_pr_curves.tsv ({len(pr):,} points)")


if __name__ == "__main__":
    main()
