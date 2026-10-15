"""X2 -- Evaluation of the distance-balanced candidate against the unweighted control (Amendment 4; Task 11b Step 2).

Both models are compared on identical variants, as Amendment 4 fixes:
  1. gene-disjoint out-of-fold predictions on the derivation data (data/v2_oof_distance_weighting.csv, V10): the basis
     for choosing;
  2. the strict test set, EXPLORATORY: it motivated the redesign, so it is not independent for this choice. The control
     there is the frozen v2.0 full model; the candidate is models/distance_weighting/candidate_full.joblib.
Paired auPRC difference (candidate minus control) with a gene-clustered bootstrap 95% CI (Ensembl gene IDs resampled with
replacement, 1,000 iterations, seed 42), for: all; pooled non-canonical; every region; Deep Intronic and its bands
<= 50, 51-500 and > 500 bp; the O5 sensitivity set and its bands; on the derivation data also the six weighting bins.
On the strict test set the <= 50 bp bands carry counts and false positives only (4 and 3 P/LP), and the Branch Point
Region (5 P/LP) is reported but not interpreted.
False-positive burden at matched operating points: for each model, the score threshold reaching 80% and 90% recall of
the evaluation set's P/LP (Deep Intronic, its sensitivity set, pooled non-canonical); the B/LB at or above it, overall
and in each Deep Intronic distance band.

Writes results/X2_distance_weighting_eval.txt and .tsv.
"""
import os, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from common import CANONICAL, NONCANONICAL, REGIONS, ROOT, Tee, deep_intronic_sensitivity, load_test, score

BANDS = [("<= 50 bp", 0, 50), ("51-500 bp", 50, 500), ("> 500 bp", 500, np.inf)]
WBINS = [("bin <= 50", 0, 50), ("bin 51-100", 50, 100), ("bin 101-250", 100, 250), ("bin 251-500", 250, 500),
         ("bin 501-1000", 500, 1000), ("bin > 1000", 1000, np.inf)]
RECALLS = (0.80, 0.90)


def band(d, lo, hi):
    a = np.abs(d)
    return (a > lo) & (a <= hi)


def subsets(df, weighting_bins):
    """(name, mask) pairs in reporting order."""
    out = [("all", np.ones(len(df), bool)), ("pooled non-canonical", df.region.isin(NONCANONICAL).values)]
    out += [(r, (df.region == r).values) for r in REGIONS]
    di = (df.region == "Deep Intronic").values
    out += [(f"Deep Intronic {b}", di & band(df.dist_to_boundary.values, lo, hi)) for b, lo, hi in BANDS]
    sens = deep_intronic_sensitivity(df).values
    out += [("DI sensitivity set", sens)] + [(f"DI sensitivity set {b}", sens & band(df.dist_to_boundary.values, lo, hi)) for b, lo, hi in BANDS]
    if weighting_bins:
        out += [(f"Deep Intronic {b}", di & band(df.dist_to_boundary.values, lo, hi)) for b, lo, hi in WBINS]
    return out


def paired_gene_boot(args):
    """auPRC of control and candidate, the paired difference, and a gene-clustered bootstrap CI of the difference."""
    name, y, s0, s1, genes, n_boot, seed = args
    ap0, ap1 = average_precision_score(y, s0), average_precision_score(y, s1)
    codes, gidx = np.unique(genes, return_inverse=True)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))[gidx].astype(float)
        if (w * y).sum() < 1 or (w * (1 - y)).sum() < 1:
            continue
        diffs.append(average_precision_score(y, s1, sample_weight=w) - average_precision_score(y, s0, sample_weight=w))
    lo, hi = (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)) if diffs else (np.nan, np.nan)
    return name, ap0, ap1, ap1 - ap0, lo, hi, len(diffs), len(codes)


def operating_points(log, df, y, s0, s1, tag, rows):
    """FP burden at matched recall: each model's threshold reaching the recall in the evaluation set."""
    for set_name, m in [("Deep Intronic", (df.region == "Deep Intronic").values), ("DI sensitivity set", deep_intronic_sensitivity(df).values),
                        ("pooled non-canonical", df.region.isin(NONCANONICAL).values)]:
        yy = y[m]
        log(f"\n  {tag}: {set_name} (n={m.sum():,}, P/LP {int(yy.sum())}, B/LB {int((yy == 0).sum()):,})")
        for r in RECALLS:
            k = int(np.ceil(r * yy.sum()))
            cells = []
            for model, s in [("control", s0), ("candidate", s1)]:
                ss = s[m]
                t = np.sort(ss[yy == 1])[::-1][k - 1]
                fp = (ss >= t) & (yy == 0)
                row = dict(data=tag, set=set_name, recall=r, model=model, threshold=float(t), recall_achieved=float((ss[yy == 1] >= t).mean()),
                           FP=int(fp.sum()), FPR=float(fp.sum() / (yy == 0).sum()))
                if set_name != "pooled non-canonical":
                    d = df.dist_to_boundary.values[m]
                    for b, lo, hi in BANDS:
                        bm = band(d, lo, hi)
                        row[f"FP {b}"] = int((fp & bm).sum())
                        row[f"B/LB {b}"] = int(((yy == 0) & bm).sum())
                rows.append(row)
                cells.append(row)
            for row in cells:
                bands = "; ".join(f"{b}: {row[f'FP {b}']:,} of {row[f'B/LB {b}']:,}" for b, _, _ in BANDS) if "FP <= 50 bp" in row else ""
                log(f"    recall {r:.0%}  {row['model']:<9} threshold {row['threshold']:.4g} (recall {row['recall_achieved']:.1%})  "
                    f"FP {row['FP']:,} (FPR {row['FPR']:.3%})" + (f"   by band: {bands}" if bands else ""))


def report(log, tag, df, y, s0, s1, weighting_bins, counts_only, not_interpreted, rows):
    subs = subsets(df, weighting_bins)
    jobs = [(n, y[m], s0[m], s1[m], df.gene_id.values[m], 1000, 42) for n, m in subs
            if n not in counts_only and m.sum() and 0 < y[m].sum() < m.sum()]
    with ProcessPoolExecutor(max_workers=12) as ex:
        res = {r[0]: r for r in ex.map(paired_gene_boot, jobs)}
    log(f"\n  {tag}: paired auPRC, candidate minus control; gene-clustered bootstrap 95% CI (1,000 iterations, seed 42)")
    log(f"  {'subset':<34}{'n':>9}{'P/LP':>7}{'genes':>7}{'control':>9}{'candidate':>11}{'diff':>9}{'95% CI':>22}")
    for n, m in subs:
        P = int(y[m].sum())
        if n in counts_only or n not in res:
            log(f"  {n:<34}{m.sum():>9,}{P:>7}{'':>7}   counts only" + ("" if n in counts_only else " (one class absent)"))
            rows.append(dict(data=tag, subset=n, n=int(m.sum()), P=P, note="counts only"))
            continue
        _, a0, a1, d, lo, hi, nb, ng = res[n]
        note = not_interpreted.get(n, "")
        rows.append(dict(data=tag, subset=n, n=int(m.sum()), P=P, genes=ng, auprc_control=a0, auprc_candidate=a1, diff=d, lo=lo, hi=hi,
                         boot_used=nb, note=note))
        log(f"  {n:<34}{m.sum():>9,}{P:>7}{ng:>7}{a0:>9.4f}{a1:>11.4f}{d:>+9.4f}   [{lo:+.4f}, {hi:+.4f}]" + (f"  ({note})" if note else ""))


def main():
    log = Tee(f"{ROOT}/results/X2_distance_weighting_eval.txt")
    log("=" * 104)
    log("X2 -- DISTANCE-BALANCED CANDIDATE vs UNWEIGHTED CONTROL (Amendment 4). Identical variants; paired.")
    log("=" * 104)
    rows = []

    log("\n1. GENE-DISJOINT OUT-OF-FOLD PREDICTIONS ON THE DERIVATION DATA (the basis for choosing)")
    o = pd.read_csv(f"{ROOT}/data/v2_oof_distance_weighting.csv", low_memory=False)
    y, s0, s1 = o.label.astype(int).values, o.oof_control.values, o.oof_candidate.values
    log(f"  {len(o):,} variants, {int(y.sum()):,} P/LP; control = v2.0 out-of-fold (reproduced exactly by V10)")
    report(log, "derivation, out of fold", o, y, s0, s1, True, set(), {}, rows)
    operating_points(log, o, y, s0, s1, "derivation, out of fold", rows)

    log("\n\n2. STRICT TEST SET -- EXPLORATORY: this set motivated the redesign and is not independent for choosing a model")
    t = load_test("strict")
    names = list(joblib.load(f"{ROOT}/models/distance_weighting/candidate_features.joblib"))
    cand = joblib.load(f"{ROOT}/models/distance_weighting/candidate_full.joblib")
    y, s0 = t.label.astype(int).values, score(t)
    s1 = cand.predict_proba(t[names].astype(np.float32).values)[:, 1]
    log(f"  {len(t):,} variants, {int(y.sum()):,} P/LP; control = the frozen v2.0 full model; candidate = candidate_full.joblib")
    report(log, "strict test (exploratory)", t, y, s0, s1, False, {"Deep Intronic <= 50 bp", "DI sensitivity set <= 50 bp"},
           {"Branch Point Region": "5 P/LP: not interpreted"}, rows)
    operating_points(log, t, y, s0, s1, "strict test (exploratory)", rows)
    pd.DataFrame(rows).to_csv(f"{ROOT}/results/X2_distance_weighting_eval.tsv", sep="\t", index=False)
    log(f"\nwrote results/X2_distance_weighting_eval.tsv ({len(rows)} rows)")
    log.close()


if __name__ == "__main__":
    main()
