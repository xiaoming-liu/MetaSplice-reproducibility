"""A3b -- Gene-disjoint evaluation by retraining (Amendment 3).

25,199 of the 25,240 non-canonical test variants are in genes represented in training, so a gene-disjoint
split of the test set has almost no power. v1's design on v2.0 answers the question by retraining: the
sorted union of training and test gene IDs (Ensembl) is assigned to five folds by
numpy.random.default_rng(42).integers(0, 5); for each fold, a model with v2.0's features and parameters
(V06's fit_one: v1's XGBoost parameters, scale_pos_weight per fit) is trained on the training variants of
the other four folds' genes and scores the test variants of that fold's genes. Every test variant then
has a score from a model that never saw its gene.

Reported: the full model's and the gene-disjoint model's auPRC with the paired drop (all, non-canonical,
canonical, each region, and the deep-intronic sensitivity set), and the gene-disjoint model against
SpliceAI and CADD on the non-canonical set.

Writes results/A3b_gene_heldout_retrain.txt and data/A3b_gene_disjoint_scores.csv.
"""
import importlib.util, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from common import (CANONICAL, FEATURES, NONCANONICAL, OUT, REGIONS, Tee, add_comparators, auprc, boot_ci,
                    deep_intronic_sensitivity, load_test, load_train, paired_boot_diff, score)

_spec = importlib.util.spec_from_file_location("v06", os.path.join(HERE, "..", "build", "V06_train.py"))
v06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v06)


def main():
    log = Tee(f"{OUT}/results/A3b_gene_heldout_retrain.txt")
    log("=" * 88)
    log("A3b -- GENE-DISJOINT EVALUATION BY RETRAINING, 5 GENE FOLDS (MetaSplice v2.0; Amendment 3)")
    log("=" * 88)
    log(f"parameters: {v06.PARAMS}, scale_pos_weight per fit")
    test = add_comparators(load_test("strict"))
    test["MetaSplice"] = score(test)
    train = load_train()
    nc = test.region.isin(NONCANONICAL)
    log(f"non-canonical test variants in genes represented in training: {int(test[nc].gene_id.isin(set(train.gene_id)).sum()):,} "
        f"of {int(nc.sum()):,}")
    genes = sorted(set(train.gene_id.dropna()) | set(test.gene_id.dropna()))
    rng = np.random.default_rng(42)
    fold_of = {g: int(f) for g, f in zip(genes, rng.integers(0, 5, len(genes)))}
    tr_fold, te_fold = train.gene_id.map(fold_of).values, test.gene_id.map(fold_of).values
    log(f"{len(genes):,} genes assigned to 5 folds; training N={len(train):,}; test N={len(test):,}, P={int(test.label.sum()):,}")
    gd = np.full(len(test), np.nan)
    for f in range(5):
        tr, idx = train[tr_fold != f], np.where(te_fold == f)[0]
        m = v06.fit_one(tr[FEATURES].astype(np.float32).values, tr.label.astype(int).values)
        gd[idx] = m.predict_proba(test.iloc[idx][FEATURES].astype(np.float32).values)[:, 1]
        log(f"  fold {f}: trained on {len(tr):,} variants ({tr.gene_id.nunique():,} genes) -> scored {len(idx):,} test variants")
    if np.isnan(gd).any():
        raise ValueError(f"{int(np.isnan(gd).sum())} test variants have no gene-disjoint score")
    test["gene_disjoint"] = gd

    log("\n" + "=" * 88)
    log("FULL MODEL (genes seen in training) vs GENE-DISJOINT MODELS")
    log("=" * 88)
    log(f"{'subset':<32}{'N':>8}{'P':>6}{'full':>9}{'95% CI':>18}{'gene-disj':>11}{'95% CI':>18}   paired drop [95% CI]")
    masks = [("all non-exonic", np.ones(len(test), bool)), ("non-canonical", nc.values),
             ("canonical", test.region.isin(CANONICAL).values)]
    for reg in REGIONS:
        masks.append((reg, (test.region == reg).values))
        if reg == "Deep Intronic":
            masks.append(("Deep Intronic, sensitivity set", deep_intronic_sensitivity(test).values))
    for tag, m in masks:
        y = test.label.values[m]
        if y.sum() < 3 or (y == 0).sum() < 3:
            log(f"{tag:<32}{m.sum():>8,}{int(y.sum()):>6}   too few in one class")
            continue
        a1 = boot_ci(y, test.MetaSplice.values[m])
        a2 = boot_ci(y, gd[m])
        d, lo, hi, _ = paired_boot_diff(y, test.MetaSplice.values[m], gd[m])
        log(f"{tag:<32}{m.sum():>8,}{int(y.sum()):>6}{a1[0]:>9.4f} [{a1[1]:.3f}, {a1[2]:.3f}]{a2[0]:>11.4f} "
            f"[{a2[1]:.3f}, {a2[2]:.3f}]   {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    log("\n" + "=" * 88)
    log("GENE-DISJOINT MetaSplice v2.0 vs COMPARATORS, NON-CANONICAL")
    log("(comparator scores are precomputed and unaffected by this split)")
    log("=" * 88)
    tnc = test[nc]
    a, lo_, hi_ = boot_ci(tnc.label.values, tnc.gene_disjoint.values)
    log(f"  MetaSplice v2.0 (gene-disjoint) auPRC {a:.4f} [{lo_:.4f}, {hi_:.4f}], N={len(tnc):,}")
    for name, col in [("SpliceAI", "spliceai_max"), ("CADD", "cadd_phred")]:
        d, lo, hi, n = paired_boot_diff(tnc.label.values, tnc.gene_disjoint.values, tnc[col].values)
        ok = tnc[col].notna().values
        log(f"  vs {name:<9} gene-disjoint {auprc(tnc.label.values[ok], tnc.gene_disjoint.values[ok]):.4f} vs "
            f"{auprc(tnc.label.values[ok], tnc[col].values[ok]):.4f}  diff {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  N={n:,}")
    test[["chr", "pos", "ref", "alt", "label", "region", "gene_id", "MetaSplice", "gene_disjoint"]].to_csv(
        f"{OUT}/data/A3b_gene_disjoint_scores.csv", index=False)
    log("\nwrote data/A3b_gene_disjoint_scores.csv")
    log.close()


if __name__ == "__main__":
    main()
