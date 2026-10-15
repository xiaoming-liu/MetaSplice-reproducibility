"""V07 -- Equivalence gate (docs/METASPLICE_V2_PLAN.md, Task 7).

Draws up to 200 variants per non-canonical region from the training set and from the strict test set
(seed 7; all of a region that has fewer), and runs them through the production path from a VCF: fresh
SpliceTransformer and Pangolin runs with V03's launcher (TF32 off), then the command-line scorer in
scoring mode with the committed model (MetaSplice/model), split into chromosome groups as in V04.
Comparisons, all model-matched (the scorer always applies the full model):
  features      the fresh feature matrix against the stored tables, for both samples; region, gene and
                gene ID identical
  test          the scorer's score (full model) against the full model applied to the stored features
  training      the scorer's score (full model) against the full model applied to the stored features;
                and each variant's assigned fold model applied to the fresh features against its saved
                out-of-fold score (data/v2_oof.csv), never the scorer's full-model score
Pass, overall and within each region: at least 99% of scores within 1e-4 and none further than 0.01;
every feature matching (within 1e-4 absolute or relative, or missing on both sides) for at least 99%
of variants. Labels are never used.

Writes results/V07_gate.txt; exits 1 if the gate fails.
"""
import importlib.util, json, os, subprocess, sys, time
import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from common import FEATURES, NONCANONICAL

P = "/mnt/Work0/Splice_altering_variant"
V2 = os.path.dirname(os.path.dirname(HERE))
T = f"{P}/tools/v2/gate"
D = f"{P}/docker_data"
KEY = ["chr", "pos", "ref", "alt"]
GROUPS = 8
spec = importlib.util.spec_from_file_location("v03", os.path.join(HERE, "V03_upstream_scores.py"))
v03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v03)
failures = []


def check(log, name, ok, detail=""):
    log(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures.append(name)


def sample(t, seed=7, n=200):
    t = t[t.region.isin(NONCANONICAL)]
    return pd.concat([g.sample(n=min(n, len(g)), random_state=seed) for _, g in t.groupby("region", sort=True)])


def keyset(d):
    return d[KEY].astype(str).apply(tuple, axis=1)


def joined(log, name, left, right, **kw):
    """left joined one-to-one to right on KEY, required to keep every left key."""
    m = left.merge(right, on=KEY, validate="one_to_one", **kw)
    check(log, f"{name}: one-to-one join keeps all {len(left):,} keys", len(m) == len(left) and
          set(keyset(m)) == set(keyset(left)), f"{len(m):,} joined")
    return m


def score_check(log, name, got, ref, region):
    d = pd.Series(np.abs(np.asarray(got, float) - np.asarray(ref, float)))
    if d.isna().any():
        check(log, f"{name}: no score missing", False, f"{int(d.isna().sum())} missing")
        return
    rows = [("all", np.ones(len(d), bool))] + [(r, (region == r).values) for r in sorted(region.unique())]
    for lab, m in rows:
        within, worst = (d[m] <= 1e-4).mean(), d[m].max()
        check(log, f"{name:<44} {lab:<20} n={int(m.sum()):>5}  within 1e-4 {within:7.2%}  max |diff| {worst:.2e}",
              within >= 0.99 and worst <= 0.01)


def feature_check(log, name, fresh, stored):
    m = joined(log, name, stored, fresh, suffixes=("_stored", ""))
    for c in ["region", "gene", "gene_id"]:
        diff = int((m[c].astype(str) != m[f"{c}_stored"].astype(str)).sum())
        check(log, f"{name}: {c} identical", diff == 0, f"{diff} differ")
    worst = []
    for c in FEATURES:
        a, b = m[c].astype(float).values, m[f"{c}_stored"].astype(float).values
        ok = np.isclose(a, b, rtol=1e-4, atol=1e-4, equal_nan=True)
        worst.append((ok.mean(), c, int((~ok).sum())))
    worst.sort()
    exact = sum(np.array_equal(m[c].astype(float).values, m[f"{c}_stored"].astype(float).values, equal_nan=True) for c in FEATURES)
    check(log, f"{name}: every feature matches for >= 99% of {len(m):,} variants", all(r >= 0.99 for r, _, _ in worst),
          f"least matching: {worst[0][1]} {worst[0][0]:.2%}; identical bit for bit: {exact} of {len(FEATURES)} features")


def main():
    os.makedirs(T, exist_ok=True)
    out = open(f"{V2}/results/V07_gate.txt", "w")
    def log(s=""):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    log("V07 -- equivalence gate: production path from a VCF against the stored tables, model-matched")
    tr = pd.read_csv(f"{V2}/data/v2_features_training.csv", low_memory=False)
    te = pd.read_csv(f"{V2}/data/v2_features_test_candidates.csv", low_memory=False)
    te = te[te.subset == "strict"]
    s_tr, s_te = sample(tr), sample(te)
    log(f"training sample: {len(s_tr):,} ({s_tr.region.value_counts().sort_index().to_dict()})")
    log(f"strict test sample: {len(s_te):,} ({s_te.region.value_counts().sort_index().to_dict()})")
    both = pd.concat([s_tr[KEY], s_te[KEY]]).drop_duplicates()
    order = {f"chr{c}": i for i, c in enumerate([str(k) for k in range(1, 23)] + ["X", "Y"])}
    both = both.assign(o=both.chr.map(order)).sort_values(["o", "pos", "ref", "alt"]).drop(columns="o").reset_index(drop=True)

    log("\nFresh SpliceTransformer and Pangolin runs (V03 launcher, TF32 off)")
    paths = {}
    for tool in ("spt", "pangolin"):
        sec, stems = v03.run_parallel(tool, both, f"{T}/{tool}", 4)
        paths[tool] = f"{T}/{tool}.all.csv"
        if tool == "spt":
            pd.concat([pd.read_csv(f"{s}.spt.csv", index_col=0) for s in stems]).to_csv(paths[tool])
        else:
            with open(paths[tool], "w") as fo:
                for k, s in enumerate(stems):
                    with open(f"{s}.pangolin_out.csv") as f:
                        h = f.readline()
                        if k == 0:
                            fo.write(h)
                        fo.writelines(f)
        log(f"  {tool}: {len(both):,} variants in {sec:.0f} s")

    log("\nCommand-line scorer, scoring mode, committed model")
    groups, load = [[] for _ in range(GROUPS)], [0] * GROUPS
    for c, n in both.chr.value_counts().items():
        k = load.index(min(load)); groups[k].append(c); load[k] += n
    procs, t0 = [], time.time()
    for k, chroms in enumerate(groups):
        if not chroms:
            continue
        stem = f"{T}/score.g{k}"
        v03.write_inputs(both[both.chr.isin(chroms)], stem)
        cmd = [sys.executable, f"{P}/MetaSplice/score_v2.py", "--input", f"{stem}.vcf", "--output", f"{stem}.out.tsv",
               "--features-out", f"{stem}.features.tsv",
               "--fasta", f"{P}/hg38/hg38.fa", "--hg19-fasta", f"{P}/hg19/hg19.fa", "--gtf", f"{D}/gencode.v50.annotation.gtf.gz",
               "--reverse-liftover-dir", f"{P}/hg38tohg19_ensembl", "--dann-dir", f"{D}/DANN",
               "--phylop-dir", f"{D}/phyloP_GRCh38", "--phastcons-dir", f"{D}/phastCons_GRCh38",
               "--gnomad-dir", f"{D}/gnomAD", "--gpn-msa", f"{D}/gpn_msa/scores.tsv.bgz",
               "--dbnsfp-gene", f"{D}/dbNSFP5.4_gene.gz", "--spt-scores", paths["spt"], "--pangolin-scores", paths["pangolin"]]
        procs.append((k, subprocess.Popen(cmd, stdout=open(f"{stem}.log", "w"), stderr=subprocess.STDOUT)))
    for k, p in procs:
        if p.wait():
            raise RuntimeError(f"score_v2.py failed for group {k}; see {T}/score.g{k}.log")
    fresh = pd.concat([pd.read_csv(f"{T}/score.g{k}.features.tsv", sep="\t", low_memory=False) for k, _ in procs], ignore_index=True)
    scored = pd.concat([pd.read_csv(f"{T}/score.g{k}.out.tsv", sep="\t", low_memory=False) for k, _ in procs], ignore_index=True)
    fresh = fresh.merge(scored[KEY + ["splice_score"]], on=KEY, validate="one_to_one")
    log(f"  {len(fresh):,} rows for {len(both):,} requested variants, in {(time.time() - t0) / 60:.0f} min")
    want, got = set(keyset(both)), keyset(fresh)
    check(log, "every requested variant scored in scope, exactly once",
          set(got) == want and not got.duplicated().any(),
          f"missing {len(want - set(got))}, unexpected {len(set(got) - want)}, duplicated {int(got.duplicated().sum())}")

    full = joblib.load(f"{P}/MetaSplice/model/metasplice_v2_full.joblib")
    assert list(joblib.load(f"{P}/MetaSplice/model/metasplice_v2_features.joblib")) == FEATURES
    folds = json.load(open(f"{V2}/models/metasplice_v2_folds.json"))
    fold_models = [joblib.load(f"{V2}/models/metasplice_v2_fold{k}.joblib") for k in range(5)]
    oof = pd.read_csv(f"{V2}/data/v2_oof.csv", low_memory=False)

    log("\nFeatures: fresh against the stored tables")
    feature_check(log, "training sample", fresh, s_tr[KEY + ["region", "gene", "gene_id"] + FEATURES])
    feature_check(log, "strict test sample", fresh, s_te[KEY + ["region", "gene", "gene_id"] + FEATURES])

    log("\nScores, model-matched")
    for name, s in [("strict test", s_te), ("training", s_tr)]:
        m = joined(log, f"{name} scores", s[KEY + ["region"] + FEATURES], fresh[KEY + ["splice_score"]])
        ref = full.predict_proba(m[FEATURES].astype(np.float32).values)[:, 1]
        score_check(log, f"{name}: scorer (full model) vs full model, stored features", m.splice_score, ref, m.region)
    m = joined(log, "training, fresh features", s_tr[KEY + ["region", "gene_id"]], fresh[KEY + FEATURES])
    m = joined(log, "training, saved out-of-fold scores", m, oof[KEY + ["fold", "oof"]])
    k = m.gene_id.map(folds)
    check(log, "training: fold from the fold map equals the saved fold", bool((k == m.fold).all()),
          f"{int((k != m.fold).sum())} differ")
    fold_score = np.full(len(m), np.nan)
    X = m[FEATURES].astype(np.float32).values
    for j in range(5):
        sel = (k == j).values
        if sel.any():
            fold_score[sel] = fold_models[j].predict_proba(X[sel])[:, 1]
    score_check(log, "training: assigned fold model, fresh features, vs saved OOF", fold_score, m.oof, m.region)
    log(f"\nGATE: {'PASS' if not failures else 'FAIL: ' + '; '.join(failures)}")
    out.close()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
