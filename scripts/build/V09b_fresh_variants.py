"""V09b -- The fresh temporal cohort of ClinVar 2026-09-28 (docs/METASPLICE_V2_PLAN.md, Task 11 Step 8). Counts only.

Built with V02's own functions (read_clinvar, classify, dedupe, hg19_by_id, the GENCODE v50 flags, check_ref) from the
2026-09-28 release (summaries written by V09a; NCBI md5s verified at download):
  candidates   in-scope SNVs classified P/LP or B/LB in the 2026-09-28 release (GRCh38), duplicates merged as in V02;
  excluded     (1) the v2.0 training data, by variant and by ClinVar ID;
               (2) anything classified P/LP or B/LB in the 2026-04-15 release -- the examined strict test set, the
                   reclassified subgroup and any other earlier-labelled record -- by GRCh38 variant or ClinVar ID
                   (either build);
               (3) the same for the 2025-08-31 release.
  so the cohort holds variants first classified P/LP or B/LB after 2026-04-15. Each exclusion step is counted.
Labels are used only to count variants here. GRCh37 coordinates come from the 2026-09-28 GRCh37 record of the same
ClinVar ID (for the SpliceAI lookup), as in V02.

Writes data/v2_variants_fresh_20260928.csv and results/V09b_fresh_variants.txt.
"""
import importlib.util, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P = "/mnt/Work0/Splice_altering_variant"
V2 = os.path.dirname(os.path.dirname(HERE))
NEW38 = f"{P}/clinvar20260928/hg38/clinvar_20260928.vcf.gz.summary.gz"
NEW37 = f"{P}/clinvar20260928/hg19/clinvar_20260928.vcf.gz.summary.gz"
_spec = importlib.util.spec_from_file_location("v02", os.path.join(HERE, "V02_variant_sets.py"))
v02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v02)
v02.CV37["2026-09"] = NEW37
NONCANONICAL = ["Donor Region", "Acceptor Region", "PPT Region", "Branch Point Region", "Deep Intronic"]


def labelled_records(paths38, paths37):
    """GRCh38 keys and ClinVar IDs (either build) of every P/LP or B/LB record of a release."""
    keys, ids = set(), set()
    for p in paths38:
        d = v02.read_clinvar(p, v02.PATHOGENIC | v02.BENIGN)
        keys |= set(v02.key(d)); ids |= set(d.clinvar_id)
    for p in paths37:
        d = pd.read_csv(p, sep="\t", usecols=["clinvar_id", "CLNSIG"], low_memory=False)
        ids |= set(d.clinvar_id[d.CLNSIG.isin(v02.PATHOGENIC | v02.BENIGN)])
    return keys, ids


def main():
    out = open(f"{V2}/results/V09b_fresh_variants.txt", "w")
    def log(s=""):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    log("V09b -- fresh temporal cohort, ClinVar 2026-09-28 (counts only; labels used only to count)")
    import pysam
    fa = pysam.FastaFile(v02.HG38)
    c = v02.classify(v02.read_clinvar(NEW38, v02.PATHOGENIC | v02.BENIGN))
    log(f"2026-09-28 P/LP or B/LB SNVs: {len(c):,}; in scope {int(c.in_scope.sum()):,}")
    d = v02.dedupe(c[c.in_scope], log, "fresh candidates")
    log(f"  candidates after merging duplicates: {len(d):,} (P/LP {int(d.label.sum()):,})")

    tr = pd.read_csv(f"{V2}/data/v2_variants_training.csv", usecols=["chr", "pos", "ref", "alt", "clinvar_id"], low_memory=False)
    steps = [("in the v2.0 training data", set(v02.key(tr)), set(tr.clinvar_id)),
             ("P/LP or B/LB in the 2026-04-15 release (includes the strict set and the reclassified subgroup)",
              *labelled_records([v02.CV38["2026"]], [v02.CV37["2026"]])),
             ("P/LP or B/LB in the 2025-08-31 release", *labelled_records([v02.CV38["2025"]], [v02.CV37["2025"]]))]
    te = pd.read_csv(f"{V2}/data/v2_variants_test_candidates.csv", usecols=["chr", "pos", "ref", "alt", "clinvar_id", "subset"])
    for name, keys, ids in steps:
        hit = v02.key(d).isin(keys) | d.clinvar_id.isin(ids)
        log(f"  excluded, {name}: {int(hit.sum()):,} (P/LP {int(d.label[hit].sum()):,})")
        d = d[~hit].copy()
    for sub in ["strict", "reclassified"]:
        t = te[te.subset == sub]
        left = v02.key(d).isin(set(v02.key(t))) | d.clinvar_id.isin(set(t.clinvar_id))
        log(f"  check: fresh variants overlapping the {sub} 2026-04-15 set by key or ID: {int(left.sum())}")
        if left.any():
            raise ValueError(f"the fresh cohort overlaps the {sub} set")
    d = d.reset_index(drop=True)
    d["canonical_any_gencode"], d["exonic_any_gencode"] = v02.canonical_any_gencode(d), v02.exonic_any_gencode(d)
    d = v02.hg19_by_id(d, "2026-09")
    v02.check_ref(d, fa, "fresh cohort")
    cols = v02.KEY + ["clinvar_id", "label", "clnsig", "review_status", "region", "gene", "gene_id", "canonical_any_gencode",
                      "exonic_any_gencode", "chr19", "pos19", "ref19", "alt19"]
    d[cols].to_csv(f"{V2}/data/v2_variants_fresh_20260928.csv", index=False)

    log(f"\nfresh cohort: {len(d):,} variants, P/LP {int(d.label.sum()):,}, B/LB {int((d.label == 0).sum()):,}; "
        f"GRCh37 record found {d.pos19.notna().mean():.2%}")
    log("by region: " + ", ".join(f"{r} {int((d.region == r).sum()):,} (P/LP {int(d.label[d.region == r].sum())})"
                                  for r in v02.CANONICAL | set(NONCANONICAL) if (d.region == r).any()))
    nc = d[d.region.isin(NONCANONICAL)]
    log(f"non-canonical: {len(nc):,} (P/LP {int(nc.label.sum())})")
    di = d[d.region == "Deep Intronic"]
    sens = di[~di.canonical_any_gencode & ~di.exonic_any_gencode]
    dist_note = "distance bands need the scorer's features (V09c)"
    log(f"Deep Intronic: {len(di):,} (P/LP {int(di.label.sum())}); sensitivity set {len(sens):,} (P/LP {int(sens.label.sum())}); {dist_note}")
    fa.close()
    out.close()


if __name__ == "__main__":
    main()
