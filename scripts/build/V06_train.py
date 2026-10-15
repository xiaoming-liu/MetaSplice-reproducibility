"""V06 -- Train MetaSplice v2.0 (docs/METASPLICE_V2_PLAN.md, Task 6).

The 38 features (score_v2.FEATURES, the G1 clinical model's list and order) and v1's hyperparameters
(XGBoost: 300 trees, depth 6, learning rate 0.1, hist, seed 42), with scale_pos_weight set per fit and
no tuning. Five gene-grouped folds keyed by Ensembl gene ID, by v1's rule (a seeded shuffle of the
genes in order of first appearance, i % 5), give out-of-fold scores; the full model is fitted on every
training variant. Derivation data only: no test, VUS, literature, SpliceVarDB or gnomAD table is read.

Writes models/metasplice_v2_{full,fold0..fold4}.joblib, models/metasplice_v2_features.joblib,
models/metasplice_v2_folds.json, data/v2_oof.csv and results/V06_train.txt.
"""
import json, os, sys, time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from common import FEATURES, NONCANONICAL

V2 = os.path.dirname(os.path.dirname(HERE))
PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist", random_state=42, n_jobs=-1)


def gene_folds(tr, seed=42):
    """Gene-grouped folds by v1's rule (calib.gene_fold_map), keyed by Ensembl gene ID."""
    genes = list(dict.fromkeys(tr.gene_id.fillna("UNKNOWN")))
    np.random.RandomState(seed).shuffle(genes)
    return {g: i % 5 for i, g in enumerate(genes)}


def fit_one(X, y):
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    return XGBClassifier(scale_pos_weight=spw, verbosity=0, **PARAMS).fit(X, y)


def main():
    out = open(f"{V2}/results/V06_train.txt", "w")
    def log(s=""):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    import xgboost
    log(f"V06 -- MetaSplice v2.0 training: 38 features, XGBoost {xgboost.__version__}, {PARAMS}")
    tr = pd.read_csv(f"{V2}/data/v2_features_training.csv", low_memory=False)
    X, y = tr[FEATURES].astype(np.float32).values, tr.label.astype(int).values
    log(f"training: {len(tr):,} variants, P/LP {int(y.sum()):,}, {tr.gene_id.nunique():,} genes")
    folds = gene_folds(tr)
    fold = tr.gene_id.fillna("UNKNOWN").map(folds).values
    oof, models = np.zeros(len(tr)), []
    for k in range(5):
        t0 = time.time()
        m = fit_one(X[fold != k], y[fold != k])
        oof[fold == k] = m.predict_proba(X[fold == k])[:, 1]
        models.append(m)
        log(f"  fold {k}: {int((fold == k).sum()):,} variants ({int(y[fold == k].sum()):,} P/LP) held out; {time.time() - t0:.0f} s")
    full = fit_one(X, y)

    log("\nRefit check: the same fits again give identical predictions")
    again_full = fit_one(X, y).predict_proba(X)[:, 1]
    again_k0 = fit_one(X[fold != 0], y[fold != 0]).predict_proba(X[fold == 0])[:, 1]
    same = np.array_equal(again_full, full.predict_proba(X)[:, 1]) and np.array_equal(again_k0, oof[fold == 0])
    log(f"  full model and fold 0: {'identical' if same else 'DIFFERENT'}")
    if not same:
        raise RuntimeError("refitting does not reproduce the predictions")

    log("\nOut-of-fold results (derivation data only)")
    def row(name, m):
        yy, ss = y[m], oof[m]
        ap = average_precision_score(yy, ss) if 0 < yy.sum() < len(yy) else float("nan")
        auc = roc_auc_score(yy, ss) if 0 < yy.sum() < len(yy) else float("nan")
        log(f"  {name:<44} n={int(m.sum()):>7,}  P/LP={int(yy.sum()):>6,}  auPRC {ap:.4f}  AUROC {auc:.4f}")
    row("all", np.ones(len(tr), bool))
    nc = tr.region.isin(NONCANONICAL).values
    row("non-canonical", nc)
    for r in sorted(tr.region.unique()):
        row(r, (tr.region == r).values)
    di = (tr.region == "Deep Intronic").values
    sens = di & ~tr.canonical_any_gencode.astype(bool).values & ~tr.exonic_any_gencode.astype(bool).values
    row("Deep Intronic, wider sensitivity set (O5)", sens)

    gain = pd.Series(full.get_booster().get_score(importance_type="gain"))
    gain.index = [FEATURES[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k for k in gain.index]
    log("\nFull model, top 12 features by gain: " + ", ".join(f"{k} {v:.0f}" for k, v in gain.sort_values(ascending=False).head(12).items()))

    os.makedirs(f"{V2}/models", exist_ok=True)
    joblib.dump(list(FEATURES), f"{V2}/models/metasplice_v2_features.joblib")
    joblib.dump(full, f"{V2}/models/metasplice_v2_full.joblib")
    for k, m in enumerate(models):
        joblib.dump(m, f"{V2}/models/metasplice_v2_fold{k}.joblib")
    json.dump(folds, open(f"{V2}/models/metasplice_v2_folds.json", "w"))
    tr[["chr", "pos", "ref", "alt", "clinvar_id", "gene_id", "region", "label"]].assign(fold=fold, oof=oof).to_csv(
        f"{V2}/data/v2_oof.csv", index=False)
    log(f"\nwrote models/ (full, 5 folds, feature list, fold map) and data/v2_oof.csv")
    out.close()


if __name__ == "__main__":
    main()
