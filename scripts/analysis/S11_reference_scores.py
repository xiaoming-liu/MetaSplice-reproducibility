"""S11 (Step 1) -- MetaSplice v2.0 scores of the gnomAD population reference (docs/METASPLICE_V2_PLAN.md, Task 11).

Both gnomAD v4.1 samples of V02b (uniform 100,000, primary; region-stratified 50,000, secondary), with
the features V04/V05 built through the v2.0 scorer:
  MetaSplice       the full model (common.score);
  MetaSplice_fold  the gene-matched fold model: each variant scored by the fold model that excluded its
                   gene (models/metasplice_v2_folds.json, by Ensembl gene ID), the same scheme as the
                   out-of-fold derivation scores the prior's components come from.
Genes outside the fold map (in the pathogenic-gene list but with no in-scope training variant) are out of
sample for every fold model. They get a fold by v1's S11b rule (a seeded shuffle of the sorted gene IDs,
i % 5) and gene_in_training = False, so the prior can also be estimated without them.

Writes data/v2_gnomad_reference_scores.csv and results/S11_reference_scores.txt.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib
import numpy as np
import pandas as pd
from common import FEATURES, KEY, MODEL_DIR, ROOT, Tee, in_training, score
from calib import gene_fold_map

SAMPLES = {"uniform": "v2_features_gnomad_reference.csv", "stratified": "v2_features_gnomad_reference_stratified.csv"}
KEEP = ["region", "gene", "gene_id", "gnomad_af", "canonical_any_gencode", "exonic_any_gencode"]


def main():
    log = Tee(f"{ROOT}/results/S11_reference_scores.txt")
    log("S11 Step 1 -- MetaSplice v2.0 scores of the gnomAD population reference (full and gene-matched fold models)")
    g2f = gene_fold_map()
    folds = [joblib.load(f"{MODEL_DIR}/metasplice_v2_fold{k}.joblib") for k in range(5)]
    parts = []
    for name, f in SAMPLES.items():
        d = pd.read_csv(f"{ROOT}/data/{f}", low_memory=False)
        if in_training(d).any():
            raise ValueError(f"{name}: {int(in_training(d).sum())} reference variants are in the training data")
        if d.gene_id.isna().any() or d.gnomad_af.isna().any():
            raise ValueError(f"{name}: a reference variant lacks a gene ID or an allele frequency")
        d["MetaSplice"] = score(d)
        d["gene_in_training"] = d.gene_id.isin(g2f)
        extra = sorted(set(d.gene_id[~d.gene_in_training]))
        rng = np.random.RandomState(42)
        rng.shuffle(extra)
        fold = d.gene_id.map(g2f).fillna(d.gene_id.map({g: i % 5 for i, g in enumerate(extra)})).astype(int)
        X = d[FEATURES].astype(np.float32).values
        d["fold"], d["MetaSplice_fold"] = fold.values, np.nan
        for k in range(5):
            m = (fold == k).values
            d.loc[m, "MetaSplice_fold"] = folds[k].predict_proba(X[m])[:, 1]
        if d.MetaSplice_fold.isna().any():
            raise ValueError(f"{name}: a reference variant has no fold score")
        log(f"  {name}: {len(d):,} variants; genes {d.gene_id.nunique():,}; outside the fold map {int((~d.gene_in_training).sum()):,} "
            f"variants in {len(extra):,} genes (seeded fold, flagged)")
        log(f"    by region: {d.region.value_counts().to_dict()}")
        parts.append(d[KEY + KEEP + ["gene_in_training", "fold", "MetaSplice", "MetaSplice_fold"]].assign(sample=name))
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(f"{ROOT}/data/v2_gnomad_reference_scores.csv", index=False)
    log(f"wrote data/v2_gnomad_reference_scores.csv ({len(out):,} rows)")
    log.close()


if __name__ == "__main__":
    main()
