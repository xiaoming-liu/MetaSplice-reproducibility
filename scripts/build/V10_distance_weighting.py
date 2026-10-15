"""V10 -- Distance-balanced training experiment (docs/AJHG_COMPARISON_PROTOCOL.md Amendment 4; docs/METASPLICE_V2_PLAN.md
Task 11b Step 1). Training only: no performance is computed here.

One controlled comparison on the 413,199-row v2.0 derivation table, with V06's joint 38-feature model unchanged
(features, hyperparameters, five gene-grouped folds, training rows, scale_pos_weight from class counts):
  control    unweighted; must reproduce the saved v2.0 out-of-fold and full-model predictions exactly;
  candidate  identical except for per-row sample_weight. P/LP and every variant outside Deep Intronic keep weight 1.
             Deep Intronic B/LB are binned by |dist_to_boundary| (<= 50, 51-100, 101-250, 251-500, 501-1000, > 1000
             bp) and, within each model fit, a B/LB variant in bin b gets
               w_b = [(DI P/LP in b) / (all DI P/LP)] / [(DI B/LB in b) / (all DI B/LB)],
             counted only from the rows that fit uses (held-out fold labels never enter fold weights). The weighted
             B/LB distance distribution then equals the P/LP one and the total B/LB weight is unchanged. The distance
             feature is kept.
Safeguards: the control reproduces V06; the folds equal the saved fold map; each fit's weights have the stated
distribution and total; every file read is audited and no validation or test table may be among them; bin counts,
weight ranges and effective sample sizes are logged. Weights are not capped, bins not changed, nothing tuned.

Writes models/distance_weighting/ (candidate fold and full models, feature list, weights), data/v2_oof_distance_weighting.csv
and results/V10_distance_weighting.txt. The v2.0 models are never overwritten.
"""
import builtins, hashlib, importlib.util, json, os, sys, time
import numpy as np
import pandas as pd

READS, WRITES = [], []
_open, _read_csv = builtins.open, pd.read_csv


def _audited_open(file, mode="r", *a, **k):
    if isinstance(file, (str, bytes, os.PathLike)):
        (WRITES if any(c in mode for c in "wax+") else READS).append(os.path.abspath(os.fsdecode(file)))
    return _open(file, mode, *a, **k)


def _audited_read_csv(path, *a, **k):
    READS.append(os.path.abspath(str(path)))
    return _read_csv(path, *a, **k)


builtins.open, pd.read_csv = _audited_open, _audited_read_csv

import joblib
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
OUTDIR = f"{V2}/models/distance_weighting"
KEY = ["chr", "pos", "ref", "alt"]
BINS = [("<= 50", 0, 50), ("51-100", 50, 100), ("101-250", 100, 250), ("251-500", 250, 500), ("501-1000", 500, 1000),
        ("> 1000", 1000, np.inf)]
FORBIDDEN = ["test_candidates", "vus", "literature", "splicevardb", "gnomad", "comparators", "clinvar"]
_spec = importlib.util.spec_from_file_location("v06", os.path.join(HERE, "V06_train.py"))
v06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v06)
FEATURES = v06.FEATURES


def bin_index(dist):
    d = np.abs(np.asarray(dist, float))
    out = np.full(len(d), -1)
    for i, (_, lo, hi) in enumerate(BINS):
        out[(d > lo) & (d <= hi)] = i
    if (out < 0).any():
        raise ValueError(f"{int((out < 0).sum())} distances fall in no bin")
    return out


def weights(region, y, dist, log, name):
    """Per-row weights for one fit, from that fit's rows only; checks distribution and total."""
    di = region == "Deep Intronic"
    b = bin_index(dist)
    P = np.array([((y == 1) & di & (b == i)).sum() for i in range(len(BINS))], float)
    B = np.array([((y == 0) & di & (b == i)).sum() for i in range(len(BINS))], float)
    if (B == 0).any() or (P == 0).any():
        raise SystemExit(f"{name}: a distance bin has no DI P/LP or no DI B/LB (P={P}, B={B}); bring back for an amendment")
    wb = (P / P.sum()) / (B / B.sum())
    w = np.ones(len(y))
    m = di & (y == 0)
    w[m] = wb[b[m]]
    tot = w[m].sum()
    share_w = np.array([w[m & (b == i)].sum() for i in range(len(BINS))]) / tot
    if not np.isclose(tot, m.sum(), rtol=1e-9, atol=0):
        raise SystemExit(f"{name}: total DI B/LB weight {tot} differs from the count {m.sum()}")
    if np.abs(share_w - P / P.sum()).max() > 1e-12:
        raise SystemExit(f"{name}: the weighted B/LB distance distribution does not match the P/LP distribution")
    ess = tot ** 2 / (w[m] ** 2).sum()
    log(f"  {name}: DI P/LP {int(P.sum())}, DI B/LB {int(B.sum()):,}; total B/LB weight {tot:,.1f} (= count); "
        f"ESS of DI B/LB {ess:,.0f}; weight range {wb.min():.4f}-{wb.max():.4f}")
    log("    " + "  ".join(f"{lab}: P {int(p)}, B {int(bb):,}, w {x:.4f}" for (lab, _, _), p, bb, x in zip(BINS, P, B, wb)))
    return w, dict(P=P.tolist(), B=B.tolist(), w=wb.tolist(), ess=float(ess))


def fit_weighted(X, y, w):
    spw = (y == 0).sum() / max((y == 1).sum(), 1)          # V06's rule, from class counts
    return XGBClassifier(scale_pos_weight=spw, verbosity=0, **v06.PARAMS).fit(X, y, sample_weight=w)


def md5(path):
    return hashlib.md5(_open(path, "rb").read()).hexdigest()


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    out = _open(f"{V2}/results/V10_distance_weighting.txt", "w")
    def log(s=""):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    log(f"V10 -- distance-balanced training experiment (Amendment 4); XGBoost parameters {v06.PARAMS}")
    tr = pd.read_csv(f"{V2}/data/v2_features_training.csv", low_memory=False)
    X, y = tr[FEATURES].astype(np.float32).values, tr.label.astype(int).values
    region, dist = tr.region.values, tr.dist_to_boundary.values
    log(f"training: {len(tr):,} rows, P/LP {int(y.sum()):,}; Deep Intronic {int((region == 'Deep Intronic').sum()):,}")

    log("\nCheck: folds")
    folds = v06.gene_folds(tr)
    saved = json.load(open(f"{V2}/models/metasplice_v2_folds.json"))
    fold = tr.gene_id.fillna("UNKNOWN").map(folds).values
    oof_saved = pd.read_csv(f"{V2}/data/v2_oof.csv", low_memory=False)
    same_rows = bool((oof_saved[KEY].astype(str).values == tr[KEY].astype(str).values).all())
    ok_folds = folds == saved and same_rows and np.array_equal(fold, oof_saved.fold.values)
    log(f"  fold map equals models/metasplice_v2_folds.json, rows in v2_oof.csv order, folds equal: {ok_folds}")
    if not ok_folds:
        raise SystemExit("fold assignments differ from v2.0")

    log("\nControl: unweighted V06 refit")
    oof_c = np.zeros(len(tr))
    for k in range(5):
        m = v06.fit_one(X[fold != k], y[fold != k])
        oof_c[fold == k] = m.predict_proba(X[fold == k])[:, 1]
    full_c = v06.fit_one(X, y).predict_proba(X)[:, 1]
    saved_full = joblib.load(f"{V2}/models/metasplice_v2_full.joblib").predict_proba(X)[:, 1]
    d_oof, d_full = np.abs(oof_c - oof_saved.oof.values).max(), np.abs(full_c - saved_full).max()
    log(f"  out-of-fold vs v2_oof.csv: max |diff| {d_oof:.2e}; full model vs the saved v2.0 model: max |diff| {d_full:.2e}")
    if d_oof > 1e-12 or d_full > 0:
        raise SystemExit("the unweighted control does not reproduce v2.0")
    log("  the control reproduces v2.0 exactly")

    log("\nCandidate: distance-balanced weights, per fit from that fit's rows only")
    summary, models, oof_w = {}, [], np.zeros(len(tr))
    for k in range(5):
        idx = fold != k
        w, summary[f"fold{k}"] = weights(region[idx], y[idx], dist[idx], log, f"fold {k} (fit on {int(idx.sum()):,} rows)")
        t0 = time.time()
        m = fit_weighted(X[idx], y[idx], w)
        oof_w[fold == k] = m.predict_proba(X[fold == k])[:, 1]
        models.append(m)
        log(f"    fitted in {time.time() - t0:.0f} s")
    w_full, summary["full"] = weights(region, y, dist, log, f"full (fit on {len(tr):,} rows)")
    full_w = fit_weighted(X, y, w_full)
    again = fit_weighted(X[fold != 0], y[fold != 0], weights(region[fold != 0], y[fold != 0], dist[fold != 0], lambda s: None, "refit")[0])
    same = np.array_equal(again.predict_proba(X[fold == 0])[:, 1], oof_w[fold == 0])
    log(f"\nCheck: refitting candidate fold 0 gives identical predictions: {same}")
    if not same:
        raise SystemExit("the candidate refit is not deterministic")

    joblib.dump(list(FEATURES), f"{OUTDIR}/candidate_features.joblib")
    joblib.dump(full_w, f"{OUTDIR}/candidate_full.joblib")
    for k, m in enumerate(models):
        joblib.dump(m, f"{OUTDIR}/candidate_fold{k}.joblib")
    json.dump(dict(bins=[b[0] for b in BINS], fits=summary), _open(f"{OUTDIR}/candidate_weights.json", "w"), indent=1)
    tr[KEY + ["clinvar_id", "gene_id", "region", "label", "dist_to_boundary", "canonical_any_gencode", "exonic_any_gencode"]].assign(
        fold=fold, oof_control=oof_c, oof_candidate=oof_w).to_csv(f"{V2}/data/v2_oof_distance_weighting.csv", index=False)
    log("\nCandidate model files (md5, for locking):")
    for f in ["candidate_full.joblib"] + [f"candidate_fold{k}.joblib" for k in range(5)]:
        log(f"  {md5(f'{OUTDIR}/{f}')}  models/distance_weighting/{f}")

    reads = sorted(set(READS))
    bad = [r for r in reads if r.startswith(V2) and any(p in os.path.basename(r) for p in FORBIDDEN)]
    log("\nCheck: files read during training (audited; writes listed separately):")
    for r in reads:
        if r.startswith(V2) or "/data/" in r:
            log(f"  read   {r}")
    for r in sorted(set(WRITES)):
        if r.startswith(V2):
            log(f"  wrote  {r}")
    log(f"  validation or test tables among them: {len(bad)}")
    if bad:
        raise SystemExit(f"a validation or test table was read: {bad}")
    log("\nwrote models/distance_weighting/, data/v2_oof_distance_weighting.csv; no performance computed here")
    out.close()


if __name__ == "__main__":
    main()
