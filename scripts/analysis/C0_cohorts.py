"""C0 -- Cohorts and comparator coverage of MetaSplice v2.0 (docs/METASPLICE_V2_PLAN.md, Task 8 Step 3).

Counts only. Until protocol Amendment 3 is committed (Task 9), the labels of the v2.0 evaluation sets
(the test set, the calibration cohorts, the reclassified subgroup, the literature set and SpliceVarDB)
are used to count variants and for nothing else. This script loads no model and computes no score,
rank, auPRC, AUROC, likelihood ratio or score summary; the comparator columns are read only for whether
a value is present.

These are the denominators Amendment 3 records. MetaSplice scores every in-scope variant (each has its
feature row), so the pairing "MetaSplice & SpliceAI" is the SpliceAI-scored subset, "MetaSplice & CADD"
the CADD-scored subset and "all three" the subset both comparators score. Wherever deep-intronic
variants are counted, the wider sensitivity set (Deep Intronic, neither canonical nor exonic in any
GENCODE v50 transcript; decision O5) is counted with them.

Writes results/C0_cohorts.txt and results/C0_cohorts.tsv.
"""
import hashlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from common import (ROOT, OUT, FEATURES, NONCANONICAL, REGIONS, Tee, add_comparators, deep_intronic_sensitivity,
                    in_training, load_literature, load_splicevardb, load_test, load_train)
from calib import in_cohort, one_star, pathogenic_genes, was_vus_2025

PAIRINGS = [("MetaSplice", []), ("MetaSplice & SpliceAI", ["spliceai_max"]), ("MetaSplice & CADD", ["cadd_phred"]),
            ("all three", ["spliceai_max", "cadd_phred"])]
INPUTS = ["v2_variants_test_candidates.csv", "v2_features_test_candidates.csv", "v2_features_training.csv", "v2_oof.csv",
          "v2_features_vus.csv", "v2_features_literature.csv", "v2_features_splicevardb.csv", "v2_pathogenic_genes.csv",
          "v2_comparators.csv"]
rows = []


def count(log, cohort, subset, d, label="label", pairings=True, names=("P/LP", "B/LB")):
    """Log and record n and the two classes of d, for each pairing (or for MetaSplice only)."""
    for pairing, cols in PAIRINGS if pairings else PAIRINGS[:1]:
        x = d[d[cols].notna().all(axis=1)] if cols else d
        p, b = int((x[label] == 1).sum()), int((x[label] == 0).sum())
        rows.append(dict(cohort=cohort, subset=subset, pairing=pairing, n=len(x), P=p, B=b, P_is=names[0], B_is=names[1]))
        log(f"  {subset:<34} {pairing:<22} n={len(x):>8,}  {names[0]}={p:>6,}  {names[1]}={b:>7,}")


def by_region(log, cohort, d, overall="all regions", pairings=True, **kw):
    """d overall, each region present, and the deep-intronic sensitivity set."""
    count(log, cohort, overall, d, pairings=pairings, **kw)
    for r in REGIONS:
        if (d.region == r).any():
            count(log, cohort, r, d[d.region == r], pairings=pairings, **kw)
            if r == "Deep Intronic":
                count(log, cohort, "Deep Intronic, sensitivity set", d[deep_intronic_sensitivity(d)], pairings=pairings, **kw)


def coverage(log, name, d):
    """Comparator coverage by region; no labels."""
    for region, r in [("all", d)] + [(x, d[d.region == x]) for x in REGIONS if (d.region == x).any()]:
        log(f"  {name:<20} {region:<24} {len(r):>8,}  SpliceAI {r.spliceai_max.notna().mean():7.2%}  "
            f"CADD {r.cadd_phred.notna().mean():7.2%}  both {(r.spliceai_max.notna() & r.cadd_phred.notna()).mean():7.2%}")


def main():
    log = Tee(f"{OUT}/results/C0_cohorts.txt")
    log("C0 -- cohorts and comparator coverage of MetaSplice v2.0: counts only")
    log("No model is loaded and no score is computed. Labels are counted and not otherwise used (Task 8 rule).")
    log("Inputs (md5):")
    for f in INPUTS:
        log(f"  {hashlib.md5(open(f'{ROOT}/data/{f}', 'rb').read()).hexdigest()}  data/{f}")
    pg = pathogenic_genes()
    log(f"pathogenic genes (>= 1 P/LP SNV in the 2025-08-31 release, by Ensembl ID): {len(pg):,}")

    log("\n1. Strict test set (the test set), by region")
    st = add_comparators(load_test("strict"))
    if st[FEATURES[-1]].isna().any():
        raise ValueError("a test variant has no feature row")
    by_region(log, "test set", st, pairings=False)

    log("\n2. Non-canonical test set: primary pairings, each region, the deep-intronic sensitivity set")
    nc = st[st.region.isin(NONCANONICAL)]
    by_region(log, "non-canonical test set", nc, overall="all non-canonical")
    train = add_comparators(load_train())
    tr_genes = set(train.gene_id)
    log(f"  genes of the training data: non-canonical test variants in them {int(nc.gene_id.isin(tr_genes).sum()):,}; "
        f"in other genes {int((~nc.gene_id.isin(tr_genes)).sum()):,} (A3)")

    log("\n3. Calibration cohorts: non-canonical, gnomAD AF < 0.01, gene with >= 1 P/LP in the training release")
    der = train[in_cohort(train, pg)]
    count(log, "calibration derivation (training, out of fold)", "derivation, all stars", der)
    count(log, "calibration derivation (training, out of fold)", "derivation, >= 1 star", der[one_star(der)])
    val = st[in_cohort(st, pg)]
    count(log, "calibration validation (test set)", "validation, all stars (S5)", val)
    v1s = val[one_star(val)]
    count(log, "calibration validation (test set)", "validation, >= 1 star (S6-S10)", v1s)
    g = v1s[v1s.label == 1].gene_id.value_counts()
    log(f"  S9: genes with >= 10 P/LP in the >= 1-star validation cohort: {int((g >= 10).sum())} "
        f"(holding {int(v1s.gene_id.isin(g[g >= 10].index).sum()):,} of its {len(v1s):,} variants)")

    log("\n4. Reclassified subgroup (in the 2025-08-31 release with a classification training does not use)")
    rc = add_comparators(load_test("reclassified"))
    by_region(log, "reclassified subgroup", rc, pairings=False)
    log("  S7, resolved VUS: uncertain or conflicting in the 2025-08-31 release, >= 1 star, in the calibration cohort")
    s7 = rc[in_cohort(rc, pg) & one_star(rc) & was_vus_2025(rc)]
    count(log, "resolved VUS (S7)", "resolved VUS (S7)", s7)

    log("\n5. Literature set (all P/LP)")
    lit = add_comparators(load_literature())
    by_region(log, "literature", lit)

    log("\n6. SpliceVarDB, variant level (v1's A6 rule)")
    sv = add_comparators(load_splicevardb())
    log(f"  variants {len(sv):,} from {int(sv.n_assay_records.sum()):,} assay records; assay-discordant (excluded) "
        f"{int((~sv.assay_concordant).sum())}: {sv.svdb_classes[~sv.assay_concordant].value_counts().to_dict()}")
    prim = sv[sv.assay_concordant]
    log("  primary set (assay-concordant), region by class:")
    log("    " + pd.crosstab(prim.region, prim.svdb_classes, margins=True).to_string().replace("\n", "\n    "))
    log("  A6 strict endpoint: splice-altering against normal, low-frequency excluded")
    strict = prim[prim.label_strict >= 0]
    by_region(log, "SpliceVarDB strict endpoint (A6)", strict, label="label_strict", names=("splice-altering", "normal"))
    log("  A6 broad endpoint: splice-altering or low-frequency against normal")
    broad = prim[prim.label_broad >= 0]
    count(log, "SpliceVarDB broad endpoint (A6)", "all regions", broad, label="label_broad",
          names=("splice-altering or low-frequency", "normal"))
    log("  S8: strict endpoint, non-canonical, gnomAD AF < 0.01")
    if strict.gnomad_af.isna().any():          # a missing AF would fail "< 0.01" and drop the variant silently
        raise ValueError(f"gnomad_af is missing for {int(strict.gnomad_af.isna().sum())} SpliceVarDB variants")
    s8 = strict[strict.region.isin(NONCANONICAL) & (strict.gnomad_af < 0.01)]
    by_region(log, "SpliceVarDB S8 (non-canonical, AF < 0.01)", s8, overall="all non-canonical", label="label_strict",
              names=("splice-altering", "normal"))

    log("\n7. Comparator coverage by set and region (no labels)")
    vus = pd.read_csv(f"{ROOT}/data/v2_features_vus.csv", low_memory=False)
    if in_training(vus).any():
        raise ValueError("a VUS is in the training data")
    for name, d in [("training", train), ("test strict", st), ("test reclassified", rc),
                    ("VUS", add_comparators(vus)), ("literature", lit), ("SpliceVarDB", sv)]:
        coverage(log, name, d)

    pd.DataFrame(rows).to_csv(f"{OUT}/results/C0_cohorts.tsv", sep="\t", index=False)
    log(f"\nwrote results/C0_cohorts.tsv ({len(rows)} rows)")
    log.close()


if __name__ == "__main__":
    main()
