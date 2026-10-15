"""P2 -- Feature-group importance and within-gene concordance (protocol Amendment 8; descriptive, no verdict).

1. Importance: mean |SHAP| per feature (XGBoost pred_contribs), summed within S12's feature groups, on the training
   out-of-fold set (each variant explained by its own fold model) and, with the full model, on the development test set
   (April 2026 strict test set) and the September 2026 cohort; all variants and non-canonical variants. SHAP values are
   model attributions, not mechanisms.
2. Within-gene concordance (stratified Mann-Whitney): among all P/LP-B/LB pairs of non-canonical test variants in the
   same gene, the share in which the P/LP variant scores higher (ties count one half), pooled over genes. A gene-level-
   only predictor scores 0.5 by construction. MetaSplice, SpliceAI (precomputed masked) and CADD on the variants all
   three score, in both cohorts; gene-clustered bootstrap CIs and paired differences (1,000 iterations, seed 42).
Inputs and denominators are asserted first; --check-only stops there.

Writes results/P2_importance_within_gene.txt and .tsv; with --cohort pooled (Amendment 9) the same analyses on the
pooled temporal test set, written to results/P2_importance_within_gene_pooled.txt and .tsv.
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from common import CLINVAR_2025, FEATURES, MODEL_DIR, NONCANONICAL, OUT, Tee, load_train
from paper_cohorts import check_denominators, test_cohorts, verify_locks
from S12_mechanism import GROUPS

LOCK_FILES = ["models/metasplice_v2_full.joblib", "models/metasplice_v2_features.joblib", "models/metasplice_v2_folds.json"] + \
             [f"models/metasplice_v2_fold{k}.joblib" for k in range(5)] + \
             ["data/v2_features_training.csv", "data/v2_oof.csv", "data/v2_features_test_candidates.csv",
              "data/v2_variants_test_candidates.csv", "data/v2_variants_training.csv", "data/v2_comparators.csv",
              "data/v2_features_fresh_20260928.csv",
              CLINVAR_2025["GRCh38"], CLINVAR_2025["GRCh37"]]   # read by the development-cohort loader (added after review)
if sorted(sum(GROUPS.values(), [])) != sorted(FEATURES):
    raise SystemExit("S12's feature groups do not partition the 38 features")


def mean_abs_shap(booster, X):
    c = booster.predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    return np.abs(c)            # per-variant |SHAP| per feature


def group_table(absshap):
    per_feature = absshap.mean(axis=0)
    return {g: float(sum(per_feature[FEATURES.index(f)] for f in fs)) for g, fs in GROUPS.items()}


def concordance_parts(y, s, genes):
    """Per gene (concordant pairs, total pairs), for genes with both classes."""
    out = {}
    for g, idx in pd.Series(np.arange(len(y))).groupby(genes).groups.items():
        idx = np.asarray(list(idx))
        p, b = s[idx][y[idx] == 1], np.sort(s[idx][y[idx] == 0])
        if len(p) and len(b):
            lt = np.searchsorted(b, p, side="left")
            le = np.searchsorted(b, p, side="right")
            out[g] = (float(lt.sum() + 0.5 * (le - lt).sum()), float(len(p) * len(b)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--cohort", choices=["separate", "pooled"], default="separate",
                    help="pooled: the Amendment 9 pooled test set instead of the two releases")
    args = ap.parse_args()
    stem = "P2_importance_within_gene" + ("_pooled" if args.cohort == "pooled" else "")
    log = Tee(os.devnull if args.check_only else f"{OUT}/results/{stem}.txt")
    log("=" * 100)
    log("P2 -- FEATURE-GROUP IMPORTANCE AND WITHIN-GENE CONCORDANCE (Amendment 8; descriptive, no verdict)"
        + ("  [check only]" if args.check_only else ""))
    log("=" * 100)
    verify_locks(log, LOCK_FILES + (["data/v2_features_pooled_test.csv"] if args.cohort == "pooled" else []))
    coh = test_cohorts(with_scores=not args.check_only, kind=args.cohort)
    check_denominators(log, coh)
    if args.check_only:
        return
    tsv = []

    # 1. importance
    log(f"\n1. FEATURE-GROUP IMPORTANCE: mean |SHAP| per feature, summed within group (log-odds scale)")
    train = load_train()
    if (len(train), int(train.label.sum())) != (413199, 41032):
        raise SystemExit("training denominators differ from Amendment 8")
    X = train[FEATURES].astype(np.float32).values
    absshap = np.zeros_like(X)
    for k in range(5):
        m = (train.fold.values == k)
        booster = joblib.load(f"{MODEL_DIR}/metasplice_v2_fold{k}.joblib").get_booster()
        absshap[m] = mean_abs_shap(booster, X[m])
    sets = [("training, out of fold", absshap, train)]
    full = joblib.load(f"{MODEL_DIR}/metasplice_v2_full.joblib").get_booster()
    for name, d in coh.items():
        sets.append((f"{name}, full model", mean_abs_shap(full, d[FEATURES].astype(np.float32).values), d))
    order = None
    for name, a, d in sets:
        for scope, m in [("all", np.ones(len(d), bool)), ("non-canonical", d.region.isin(NONCANONICAL).values)]:
            g = group_table(a[m])
            order = order or sorted(g, key=g.get, reverse=True)
            log(f"  {name:<28} {scope:<14} n={int(m.sum()):>7,}  " + "  ".join(f"{k} {g[k]:.3f}" for k in order))
            tsv += [dict(section="importance", set=name, scope=scope, n=int(m.sum()), group=k, mean_abs_shap=g[k]) for k in order]

    # 2. within-gene concordance
    log(f"\n2. WITHIN-GENE CONCORDANCE, non-canonical variants all three tools score "
        f"(gene-clustered bootstrap, 1,000 iterations, seed 42)")
    tools = ["MetaSplice", "SpliceAI", "CADD"]
    for name, d in coh.items():
        x = d[d.region.isin(NONCANONICAL) & d[tools].notna().all(axis=1)].reset_index(drop=True)
        y, genes = x.label.values, x.gene_id.values
        parts = {t: concordance_parts(y, x[t].values.astype(float), genes) for t in tools}
        G = sorted(parts["MetaSplice"])
        arr = {t: np.array([parts[t][g] for g in G]) for t in tools}
        est = {t: arr[t][:, 0].sum() / arr[t][:, 1].sum() for t in tools}
        rng = np.random.default_rng(42)
        boots = {t: [] for t in tools}
        for _ in range(1000):
            w = np.bincount(rng.integers(0, len(G), len(G)), minlength=len(G)).astype(float)
            for t in tools:
                boots[t].append((w * arr[t][:, 0]).sum() / (w * arr[t][:, 1]).sum())
        boots = {t: np.array(v) for t, v in boots.items()}
        log(f"\n  {name}: n={len(x):,} (P/LP {int(y.sum())}); {len(G)} genes with both classes; "
            f"{int(arr['MetaSplice'][:, 1].sum()):,} P/LP-B/LB pairs")
        for t in tools:
            lo, hi = np.percentile(boots[t], [2.5, 97.5])
            log(f"    {t:<11} concordance {est[t]:.4f} [{lo:.4f}, {hi:.4f}]")
            tsv.append(dict(section="within-gene concordance", set=name, scope=t, n=len(x), P=int(y.sum()), genes=len(G),
                            pairs=float(arr[t][:, 1].sum()), estimate=est[t], lo=lo, hi=hi))
        for t in ["SpliceAI", "CADD"]:
            dd = boots["MetaSplice"] - boots[t]
            lo, hi = np.percentile(dd, [2.5, 97.5])
            log(f"    MetaSplice minus {t:<9} {est['MetaSplice'] - est[t]:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            tsv.append(dict(section="within-gene concordance", set=name, scope=f"MetaSplice minus {t}", n=len(x),
                            P=int(y.sum()), genes=len(G), estimate=est["MetaSplice"] - est[t], lo=lo, hi=hi))
    pd.DataFrame(tsv).to_csv(f"{OUT}/results/{stem}.tsv", sep="\t", index=False)
    log(f"\nwrote results/{stem}.tsv ({len(tsv)} rows)")


if __name__ == "__main__":
    main()
