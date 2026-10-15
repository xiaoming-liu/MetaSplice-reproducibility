"""V12 -- The pooled temporal test set, labelled by ClinVar 2026-09-28 (counts only; labels used only to count).

Definition (decided with the user on 2026-10-10, option b2): every in-scope SNV classified P/LP or B/LB in the 2026-09-28
release that is not in the training data (by GRCh38 variant or ClinVar ID) and was not classified P/LP or B/LB in the
2025-08-31 training release (by GRCh38 variant or ClinVar ID, either build) -- variants first classified after the
training release. Built with V02's and V09b's functions. Label, CLNSIG and review status come from the 2026-09-28 release.

Features come from the tables already built by the scorer: the 2026-04-15 test candidates (strict and reclassified
subsets) and the 2026-09-28 fresh cohort. Every pooled variant must have exactly one feature row, with the same region,
gene and gene ID as V02's assignment. Comparator scores: SpliceAI (precomputed masked, V08) and CADD v1.7 PHRED.
Source: '2026-04-15' (classified P/LP or B/LB by the April release) or '2026-09-28' (first classified after it).

Writes data/v2_features_pooled_test.csv and results/V12_pooled_test_set.txt. No model score is computed.
"""
import importlib.util, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from common import NONCANONICAL, deep_intronic_sensitivity
from calib import ZERO_STAR, pathogenic_genes


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


v09b = load("v09b", os.path.join(HERE, "V09b_fresh_variants.py"))
v02 = v09b.v02
KEY = ["chr", "pos", "ref", "alt"]


def main():
    out = open(f"{V2}/results/V12_pooled_test_set.txt", "w")
    def log(s=""):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    log("V12 -- pooled temporal test set, labelled by ClinVar 2026-09-28 (counts only; labels used only to count)")
    c = v02.classify(v02.read_clinvar(v09b.NEW38, v02.PATHOGENIC | v02.BENIGN))
    d = v02.dedupe(c[c.in_scope], log, "2026-09-28 in scope")
    tr = pd.read_csv(f"{V2}/data/v2_variants_training.csv", usecols=KEY + ["clinvar_id"], low_memory=False)
    in_tr = v02.key(d).isin(set(v02.key(tr))) | d.clinvar_id.isin(set(tr.clinvar_id))
    keys25, ids25 = v09b.labelled_records([v02.CV38["2025"]], [v02.CV37["2025"]])
    in25 = v02.key(d).isin(keys25) | d.clinvar_id.isin(ids25)
    log(f"  2026-09-28 P/LP or B/LB in scope: {len(d):,}; in training {int(in_tr.sum()):,}; "
        f"P/LP or B/LB in 2025-08-31 (not in training) {int((in25 & ~in_tr).sum()):,}")
    p = d[~in_tr & ~in25].reset_index(drop=True)
    log(f"  pooled test set: {len(p):,} variants (P/LP {int(p.label.sum()):,})")

    te = pd.read_csv(f"{V2}/data/v2_features_test_candidates.csv", low_memory=False)
    fr = pd.read_csv(f"{V2}/data/v2_features_fresh_20260928.csv", low_memory=False)
    cmp_ = pd.read_csv(f"{V2}/data/v2_comparators.csv", usecols=KEY + ["spliceai_max", "cadd_phred"], low_memory=False)
    te = te.merge(cmp_, on=KEY, how="left", validate="one_to_one").assign(source="2026-04-15")
    fr = fr.assign(source="2026-09-28")
    feats = pd.concat([te.drop(columns=["subset"]), fr], ignore_index=True)
    if feats.duplicated(KEY).any():
        raise ValueError("a variant has feature rows in both tables")
    m = p[KEY + ["clinvar_id", "label", "clnsig", "review_status", "region", "gene", "gene_id"]].merge(
        feats.drop(columns=["clinvar_id", "label", "clnsig", "review_status"]), on=KEY, how="left",
        suffixes=("", "_feat"), validate="one_to_one")
    missing = int(m.source.isna().sum())
    if missing:
        raise ValueError(f"{missing} pooled variants have no feature row")
    for col in ["region", "gene", "gene_id"]:
        diff = int((m[col].astype(str) != m[f"{col}_feat"].astype(str)).sum())
        if diff:
            raise ValueError(f"{col} differs between V02's assignment and the feature table for {diff} variants")
    m = m.drop(columns=["region_feat", "gene_feat", "gene_id_feat"])
    log("  every pooled variant has exactly one feature row, with V02's region, gene and gene ID")
    m.to_csv(f"{V2}/data/v2_features_pooled_test.csv", index=False)

    # counts
    nc = m[m.region.isin(NONCANONICAL)]
    pg = pathogenic_genes()
    cal = nc[(nc.gnomad_af < 0.01) & nc.gene_id.isin(pg)]
    one = cal[~cal.review_status.isin(ZERO_STAR)]
    log("\nBy source")
    for s, g in m.groupby("source"):
        log(f"  {s}: {len(g):,} (P/LP {int(g.label.sum()):,}); non-canonical {int(g.region.isin(NONCANONICAL).sum()):,} "
            f"(P/LP {int(g.label[g.region.isin(NONCANONICAL)].sum())})")
    log("\nBy region (n, P/LP), and with a SpliceAI score / a CADD score")
    rows = [(r, m[m.region == r]) for r in sorted(m.region.unique())]
    rows += [("non-canonical", nc), ("Deep Intronic, sensitivity set", nc[deep_intronic_sensitivity(nc).values & (nc.region == "Deep Intronic").values]),
             ("calibration cohort, all statuses", cal), ("calibration cohort, >= 1 star", one), ("all", m)]
    for name, g in rows:
        s, cd = g[g.spliceai_max.notna()], g[g.cadd_phred.notna()]
        log(f"  {name:<34} n={len(g):>7,} P/LP={int(g.label.sum()):>5,}   SpliceAI n={len(s):>7,} P/LP={int(s.label.sum()):>5,}   "
            f"CADD n={len(cd):>7,} P/LP={int(cd.label.sum()):>5,}")
    log(f"\nwrote data/v2_features_pooled_test.csv ({len(m):,} rows); no model score computed")
    out.close()


if __name__ == "__main__":
    main()
