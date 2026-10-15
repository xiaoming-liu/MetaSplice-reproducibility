"""P1 -- Inputs, baselines and ablations (protocol Amendment 8; descriptive, no verdict).

On the non-canonical variants of the development test set (April 2026 strict test set) and the September 2026 cohort,
with rows for each non-canonical region with >= 20 P/LP, Deep Intronic and the deep-intronic sensitivity set:
  1. Standalone inputs: 12 inputs scored alone (Pangolin max, increase, -decrease; SpliceTransformer aggregate;
     -GPN-MSA; DANN; phyloP and phastCons 100-way and 470-way; pLI; -LOEUF), each on the variants it covers: auPRC with a
     stratified bootstrap CI and AUROC; MetaSplice minus the input, paired, on the variants both cover.
  2. Baselines trained on all training variants: XGBoost on the 19 deep-learning splice features (V06's parameters and
     fit_one); logistic regression on SpliceTransformer aggregate, Pangolin max and phyloP 100-way (standardised,
     training-median imputation, class_weight='balanced', lbfgs, max_iter=1000). MetaSplice minus each, paired.
  3. Ablations: the full model's parameters retrained on all training variants without (a) the gene-level features,
     (b) DANN and GPN-MSA, (c) both; auPRC, paired difference against the full model, against SpliceAI (precomputed
     masked) and against CADD. (d) ablation (c) with A3b's gene folds on the development test set, against SpliceAI and
     CADD.
Paired differences: common.paired_boot_diff (1,000 iterations, seed 42). Inputs and denominators are asserted first;
--check-only stops there.

Writes results/P1_inputs_baselines_ablations.txt and .tsv; with --cohort pooled (Amendment 9) the same analyses on the
pooled temporal test set, written to results/P1_inputs_baselines_ablations_pooled.txt and .tsv.
"""
import argparse, importlib.util, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import CLINVAR_2025, FEATURES, OUT, Tee, auprc, boot_ci, load_train, paired_boot_diff
from paper_cohorts import check_denominators, rows, test_cohorts, verify_locks

_spec = importlib.util.spec_from_file_location("v06", os.path.join(HERE, "..", "build", "V06_train.py"))
v06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v06)

INPUTS = [("Pangolin max", "pangolin_max", 1), ("Pangolin increase", "pangolin_inc", 1),
          ("Pangolin decrease (negated)", "pangolin_dec", -1), ("SpliceTransformer", "spt_orig", 1),
          ("GPN-MSA (negated)", "gpn_msa_score", -1), ("DANN", "DANN_score", 1),
          ("phyloP 100-way", "phyloP100way_vertebrate", 1), ("phyloP 470-way", "phyloP470way_mammalian", 1),
          ("phastCons 100-way", "phastCons100way_vertebrate", 1), ("phastCons 470-way", "phastCons470way_mammalian", 1),
          ("pLI", "gnomAD_pLI", 1), ("LOEUF (negated)", "gnomAD_LOEUF", -1)]
DL = [f for f in FEATURES if f.startswith("spt_")] + ["pangolin_inc", "pangolin_dec", "pangolin_max"]
LR_FEATURES = ["spt_orig", "pangolin_max", "phyloP100way_vertebrate"]
GENE_LEVEL = ["gnomAD_pLI", "gnomAD_LOEUF", "ClinGen_Haploinsufficiency_Score"]
CLIN = ["DANN_score", "gpn_msa_score"]
ABLATIONS = [("(a) without gene-level", GENE_LEVEL), ("(b) without DANN and GPN-MSA", CLIN),
             ("(c) without both", GENE_LEVEL + CLIN)]
LOCK_FILES = ["models/metasplice_v2_full.joblib", "models/metasplice_v2_features.joblib", "data/v2_features_training.csv",
              "data/v2_oof.csv", "data/v2_features_test_candidates.csv", "data/v2_variants_test_candidates.csv",
              "data/v2_variants_training.csv", "data/v2_comparators.csv", "data/v2_features_fresh_20260928.csv",
              CLINVAR_2025["GRCh38"], CLINVAR_2025["GRCh37"]]   # read by the development-cohort loader (added after review)


def fit(train, feats):
    return v06.fit_one(train[feats].astype(np.float32).values, train.label.astype(int).values)


def predict(model, d, feats):
    return model.predict_proba(d[feats].astype(np.float32).values)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--cohort", choices=["separate", "pooled"], default="separate",
                    help="pooled: the Amendment 9 pooled test set instead of the two releases")
    args = ap.parse_args()
    stem = "P1_inputs_baselines_ablations" + ("_pooled" if args.cohort == "pooled" else "")
    log = Tee(os.devnull if args.check_only else f"{OUT}/results/{stem}.txt")
    log("=" * 100)
    log("P1 -- INPUTS, BASELINES AND ABLATIONS (Amendment 8; descriptive, no verdict)" + ("  [check only]" if args.check_only else ""))
    log("=" * 100)
    verify_locks(log, LOCK_FILES + (["data/v2_features_pooled_test.csv"] if args.cohort == "pooled" else []))
    coh = test_cohorts(with_scores=not args.check_only, kind=args.cohort)
    check_denominators(log, coh)
    if args.check_only:
        return
    train = load_train()
    if (len(train), int(train.label.sum())) != (413199, 41032):
        raise SystemExit("training denominators differ from Amendment 8")
    tsv = []

    def report(name, d, cols, title):
        """cols: [(label, score array)]; rows of d; auPRC per column and paired MetaSplice minus column."""
        log(f"\n  {title}")
        for tag, m in rows(d):
            y = d.label.values[m]
            if y.sum() < 3:
                continue
            ms = d.MetaSplice.values[m]
            log(f"    {tag} (n={int(m.sum()):,}, P/LP {int(y.sum())}); MetaSplice auPRC {auprc(y, ms):.4f}")
            for lab, s in cols:
                s = np.asarray(s, float)[m]
                ok = ~np.isnan(s)
                if ok.sum() == 0 or y[ok].sum() < 3:
                    log(f"      {lab:<34} too few scored"); continue
                a, lo, hi = boot_ci(y[ok], s[ok])
                au = roc_auc_score(y[ok], s[ok])
                dd, dlo, dhi, n = paired_boot_diff(y, ms, s)
                log(f"      {lab:<34} cov {ok.mean():6.1%}  auPRC {a:.4f} [{lo:.4f}, {hi:.4f}]  AUROC {au:.4f}   "
                    f"MetaSplice minus: {dd:+.4f} [{dlo:+.4f}, {dhi:+.4f}] (n={n:,})")
                tsv.append(dict(cohort=name, section=title, row=tag, comparator=lab, n=int(m.sum()), P=int(y.sum()),
                                coverage=float(ok.mean()), auprc=a, lo=lo, hi=hi, auroc=au, diff=dd, diff_lo=dlo,
                                diff_hi=dhi, n_paired=n))

    # 1. standalone inputs
    for name, d in coh.items():
        log(f"\n{'=' * 100}\n1. STANDALONE INPUTS -- {name}\n{'=' * 100}")
        report(name, d, [(lab, sign * d[col].values.astype(float)) for lab, col, sign in INPUTS], "each input alone")

    # 2. baselines
    log(f"\n{'=' * 100}\n2. BASELINES (trained on all {len(train):,} training variants)\n{'=' * 100}")
    dl = fit(train, DL)
    med = train[LR_FEATURES].median()
    mu, sd = train[LR_FEATURES].fillna(med).mean(), train[LR_FEATURES].fillna(med).std()
    lr = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000).fit(
        ((train[LR_FEATURES].fillna(med) - mu) / sd).values, train.label.values)
    log(f"  logistic regression coefficients (standardised): {dict(zip(LR_FEATURES, np.round(lr.coef_[0], 4)))}")
    for name, d in coh.items():
        report(name, d, [("XGBoost, 19 deep-learning features", predict(dl, d, DL)),
                         ("logistic regression, 3 features", lr.predict_proba(((d[LR_FEATURES].fillna(med) - mu) / sd).values)[:, 1])],
               "baselines")

    # 3. ablations
    log(f"\n{'=' * 100}\n3. ABLATIONS (full model's parameters, retrained on all training variants)\n{'=' * 100}")
    for lab, drop in ABLATIONS:
        feats = [f for f in FEATURES if f not in drop]
        m = fit(train, feats)
        log(f"\n  {lab}: {len(feats)} features (removed {drop})")
        for name, d in coh.items():
            s = predict(m, d, feats)
            for tag, mk in rows(d):
                y = d.label.values[mk]
                a, lo, hi = boot_ci(y, s[mk])
                parts = [f"auPRC {a:.4f} [{lo:.4f}, {hi:.4f}]"]
                rec = dict(cohort=name, section="ablation", row=tag, comparator=lab, n=int(mk.sum()), P=int(y.sum()),
                           auprc=a, lo=lo, hi=hi)
                for ref, col in [("full model", "MetaSplice"), ("SpliceAI", "SpliceAI"), ("CADD", "CADD")]:
                    dd, dlo, dhi, n = paired_boot_diff(y, s[mk], d[col].values[mk])
                    parts.append(f"minus {ref} {dd:+.4f} [{dlo:+.4f}, {dhi:+.4f}] (n={n:,})")
                    rec.update({f"diff_vs_{col}": dd, f"lo_vs_{col}": dlo, f"hi_vs_{col}": dhi, f"n_vs_{col}": n})
                log(f"    {name:<12} {tag:<32} (P/LP {int(y.sum())}) " + "; ".join(parts))
                tsv.append(rec)

    # (d) ablation (c) with A3b's gene folds: the development test set, or the pooled set (Amendment 9)
    dname = "pooled" if args.cohort == "pooled" else "development"
    d = coh[dname]
    feats = [f for f in FEATURES if f not in GENE_LEVEL + CLIN]
    genes = sorted(set(train.gene_id.dropna()) | set(d.gene_id.dropna()))
    fold_of = {g: int(f) for g, f in zip(genes, np.random.default_rng(42).integers(0, 5, len(genes)))}
    trf, tef = train.gene_id.map(fold_of).values, d.gene_id.map(fold_of).values
    gd = np.full(len(d), np.nan)
    for f in range(5):
        idx = np.where(tef == f)[0]
        gd[idx] = predict(fit(train[trf != f], feats), d.iloc[idx], feats)
    if np.isnan(gd).any():
        raise ValueError("a development variant has no gene-disjoint score")
    log(f"\n  (d) ablation (c) with A3b's gene folds ({len(genes):,} genes, seed 42), {dname} test set")
    for tag, mk in rows(d):
        y = d.label.values[mk]
        a, lo, hi = boot_ci(y, gd[mk])
        parts = [f"auPRC {a:.4f} [{lo:.4f}, {hi:.4f}]"]
        rec = dict(cohort=dname, section="ablation", row=tag, comparator="(d) without both, gene-disjoint",
                   n=int(mk.sum()), P=int(y.sum()), auprc=a, lo=lo, hi=hi)
        for ref, col in [("full model", "MetaSplice"), ("SpliceAI", "SpliceAI"), ("CADD", "CADD")]:
            dd, dlo, dhi, n = paired_boot_diff(y, gd[mk], d[col].values[mk])
            parts.append(f"minus {ref} {dd:+.4f} [{dlo:+.4f}, {dhi:+.4f}] (n={n:,})")
            rec.update({f"diff_vs_{col}": dd, f"lo_vs_{col}": dlo, f"hi_vs_{col}": dhi, f"n_vs_{col}": n})
        log(f"    {tag:<32} (P/LP {int(y.sum())}) " + "; ".join(parts))
        tsv.append(rec)
    pd.DataFrame(tsv).to_csv(f"{OUT}/results/{stem}.tsv", sep="\t", index=False)
    log(f"\nwrote results/{stem}.tsv ({len(tsv)} rows)")


if __name__ == "__main__":
    main()
