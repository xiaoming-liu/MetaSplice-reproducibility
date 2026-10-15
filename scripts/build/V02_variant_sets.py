"""V02 -- Variant sets of MetaSplice v2.0 on GRCh38 (docs/METASPLICE_V2_PLAN.md, Task 3).

Every set is classified by the v2.0 scorer's own scope() (MetaSplice/score_v2.py), on the full
GENCODE v50 GTF: the training rule on MANE Select + Plus Clinical transcripts; variants outside their
gene's MANE transcripts out of scope (decision O1); Deep Intronic positions inside an exon of any
GENCODE basic transcript out of scope (decision O5). Sets:

  training         ClinVar 2025-08-31 GRCh38: SNVs, P/LP (1) or B/LB (0), in scope
  test_candidates  ClinVar 2026-04-15 GRCh38, same filters, less every training variant (by variant
                   or ClinVar ID); subset "strict" (ClinVar ID not in the 2025-08-31 release) or
                   "reclassified" (in it, with a classification training does not use)
  vus              ClinVar 2026-04-15: uncertain or conflicting, in scope, less training variants
  literature       the v1 literature set (GRCh37), lifted to GRCh38, in scope, less training variants
  splicevardb      SpliceVarDB records (GRCh37), lifted to GRCh38, in scope, less training variants
  pathogenic_genes genes with >= 1 P/LP SNV of any region in the 2025-08-31 release, inside their MANE
                   transcripts (v1's calib.pathogenic_genes: "P/LP of any type"); for cohort filters

Each row carries chr, pos, ref, alt (GRCh38), region, gene, gene_id, canonical_any_gencode and
exonic_any_gencode (the position is a canonical splice site of, or inside an exon of, any GENCODE v50
transcript; used only for the deep-intronic sensitivity result, O5) and chr19, pos19, ref19, alt19 (ClinVar sets: the GRCh37 release's record for the
same ClinVar ID; lifted sets: their original GRCh37 coordinates), used only for the SpliceAI lookup.
Labels are counted in the log; they are not otherwise used here. Duplicate records of one variant are
merged when their labels agree and dropped when they conflict.

Writes data/v2_variants_*.csv and results/V02_variant_sets.txt.
"""
import importlib.util, os, sys
import numpy as np
import pandas as pd
import pysam

P = "/mnt/Work0/Splice_altering_variant"
V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANE = f"{P}/annotation_v2/gencode.v50.mane.gtf.gz"
FULL = f"{P}/docker_data/gencode.v50.annotation.gtf.gz"
HG38 = f"{P}/hg38/hg38.fa"
LIFT19 = f"{P}/hg19tohg38_ensembl"
CV38 = {"2025": f"{P}/clinvar/hg38/clinvar_20250831.vcf.gz.summary.gz",
        "2026": f"{P}/clinvar20260418/hg38/clinvar_20260415.vcf.gz.summary.gz"}
CV37 = {"2025": f"{P}/reproducibility/raw_inputs/clinvar_20250831.vcf.gz.summary.gz",
        "2026": f"{P}/reproducibility_manuscript/data/clinvar_releases/clinvar_20260415_hg19.vcf.gz.summary.gz"}
LIT = f"{P}/reproducibility_manuscript/data/literature_validation/mapped_variants.csv"
SVDB = f"{P}/SpliceVarDB_replicate_20260316/variants_with_position_classification.fixed.no_sakaguchi.csv"
V1_TRAIN = f"{P}/reproducibility_manuscript/data/features.csv"

PATHOGENIC = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}      # as 01_prepare_clinvar
BENIGN = {"Benign", "Likely_benign", "Benign/Likely_benign"}
UNCERTAIN = {"Uncertain_significance", "Conflicting_classifications_of_pathogenicity"}
CANONICAL = {"Canonical Donor Site", "Canonical Acceptor Site"}
KEY = ["chr", "pos", "ref", "alt"]

spec = importlib.util.spec_from_file_location("score_v2", f"{P}/MetaSplice/score_v2.py")
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)


def key(d, cols=KEY):
    return d[cols[0]].astype(str) + ":" + d[cols[1]].astype(str) + ":" + d[cols[2]].astype(str) + ":" + d[cols[3]].astype(str)


def read_clinvar(path, keep):
    d = pd.read_csv(path, sep="\t", usecols=["#chr", "pos", "clinvar_id", "ref", "alt", "CLNSIG", "CLNREVSTAT"],
                    dtype={"#chr": str, "ref": str, "alt": str}, low_memory=False).rename(columns={"#chr": "chr"})
    d = d[d.ref.isin(list("ACGT")) & d.alt.isin(list("ACGT"))]
    d = d[d.CLNSIG.isin(keep)].copy() if keep is not None else d.copy()
    d["chr"] = "chr" + d.chr
    d["label"] = np.where(d.CLNSIG.isin(PATHOGENIC), 1.0, np.where(d.CLNSIG.isin(BENIGN), 0.0, np.nan))
    return d.rename(columns={"CLNSIG": "clnsig", "CLNREVSTAT": "review_status"}).reset_index(drop=True)


def classify(d):
    """region (in scope only), gene, gene_id, inside (a MANE transcript of the gene), in_scope and
    out_of_scope_reason, from the scorer's scope() on the full GENCODE v50 GTF (rules O1 and O5)."""
    v = list(zip(d.chr, d.pos.astype(int), d.ref, d.alt))
    in_scope, genes, why = ms.scope(v, FULL)
    inside = ms.in_mane_transcript(v, genes, FULL)
    d = d.copy()
    d["region"] = [in_scope.get(i) for i in range(len(d))]
    d["gene"] = [genes.get(i, (None, None))[0] for i in range(len(d))]
    d["gene_id"] = [genes.get(i, (None, None))[1] for i in range(len(d))]
    d["inside"] = [bool(inside.get(i, False)) for i in range(len(d))]
    d["in_scope"] = d.region.notna()
    d["out_of_scope_reason"] = [why.get(i, "") for i in range(len(d))]
    return d


_full_index, _full_exons = {}, {}


def exonic_any_gencode(d):
    """True where the position lies inside an exon of any GENCODE v50 transcript."""
    if not _full_exons:
        iv = {}
        for c, genes in ms.parse_gene_exons(FULL).items():
            iv[c] = [(s, e) for exons in genes.values() for _, s, e in exons]
        _full_exons.update({c: ms._merge(v) for c, v in iv.items()})
    out = []
    for c, p in zip(d.chr, d.pos.astype(int)):
        if c not in _full_exons:
            out.append(False)
            continue
        starts, ends = _full_exons[c]
        k = np.searchsorted(starts, p, side="right") - 1
        out.append(bool(k >= 0 and p <= ends[k]))
    return out


def canonical_any_gencode(d):
    """True where the position is a canonical splice site of any GENCODE v50 transcript. The training
    rule ranks canonical sites first, so it returns a canonical region whenever any exon of any
    transcript makes the position canonical."""
    if not _full_index:
        ge = ms.parse_gene_exons(FULL)            # every transcript
        _full_index.update({c: ms.build_training_index(ge[c]) for c in ge})
    return [c in _full_index and ms.classify_variant_training(int(p), _full_index[c])[0] in CANONICAL
            for c, p in zip(d.chr, d.pos)]


def dedupe(d, log, name):
    """One row per variant: records of one variant with agreeing labels are merged (lowest ClinVar ID
    kept); variants whose records disagree on the label are dropped."""
    k = key(d)
    lab = d.groupby(k)["label"].nunique(dropna=False)
    conflict = set(lab[lab > 1].index)
    n_dup = int(k.duplicated().sum())
    out = d[~k.isin(conflict)].sort_values("clinvar_id") if "clinvar_id" in d else d[~k.isin(conflict)]
    out = out.loc[~key(out).duplicated()].reset_index(drop=True)
    log(f"  {name}: {n_dup:,} duplicate records of one variant; {len(conflict):,} variants dropped for conflicting labels")
    return out


def hg19_by_id(d, release):
    """chr19, pos19, ref19, alt19 from the GRCh37 release's record for the same ClinVar ID."""
    r = pd.read_csv(CV37[release], sep="\t", usecols=["#chr", "pos", "clinvar_id", "ref", "alt"],
                    dtype={"#chr": str, "ref": str, "alt": str}, low_memory=False).rename(
        columns={"#chr": "chr19", "pos": "pos19", "ref": "ref19", "alt": "alt19"})
    r = r[r.ref19.isin(list("ACGT")) & r.alt19.isin(list("ACGT"))].drop_duplicates("clinvar_id")
    r["chr19"] = "chr" + r.chr19
    return d.merge(r, on="clinvar_id", how="left")


def lift_set(d, log, name):
    """Lift a GRCh37 set to GRCh38 with the scorer's function; keep the GRCh37 coordinates."""
    v = list(zip(d.chr, d.pos.astype(int), d.ref, d.alt))
    lifted, failed = ms.lift_variants(v, LIFT19, HG38)
    log(f"  {name}: {len(d):,} records; lifted {len(lifted):,}; not lifted {len(failed):,} "
        f"{pd.Series(list(failed.values()), dtype=str).value_counts().to_dict() if failed else ''}")
    d = d.rename(columns={"chr": "chr19", "pos": "pos19", "ref": "ref19", "alt": "alt19"}).iloc[sorted(lifted)].copy()
    d["chr"], d["pos"], d["ref"], d["alt"] = zip(*[lifted[i][:4] for i in sorted(lifted)])
    return d.reset_index(drop=True)


def check_ref(d, fa, name):
    bad = sum(fa.fetch(c, p - 1, p).upper() != r for c, p, r in zip(d.chr, d.pos.astype(int), d.ref))
    if bad:
        raise ValueError(f"{name}: {bad} REF alleles differ from GRCh38")


def main():
    out = open(f"{V2}/results/V02_variant_sets.txt", "w")
    def log(s=""):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    log("V02 -- MetaSplice v2.0 variant sets on GRCh38 (GENCODE v50 MANE Select + Plus Clinical)")
    fa = pysam.FastaFile(HG38)
    cols = KEY + ["clinvar_id", "label", "clnsig", "review_status", "region", "gene", "gene_id"]

    # ---- training and the pathogenic-gene list ----
    log("\nTraining: ClinVar 2025-08-31 GRCh38, SNVs, P/LP or B/LB")
    c25 = classify(read_clinvar(CV38["2025"], PATHOGENIC | BENIGN))
    log(f"  SNVs {len(c25):,}; in scope {int(c25.in_scope.sum()):,}; out of scope "
        f"{c25.out_of_scope_reason[~c25.in_scope].value_counts().to_dict()}")
    # v1's "P/LP of any type": every region, exonic included, inside the gene's MANE transcripts
    pg = c25[(c25.label == 1) & c25.gene_id.notna() & c25.inside][["gene_id", "gene"]].drop_duplicates("gene_id")
    pg.to_csv(f"{V2}/data/v2_pathogenic_genes.csv", index=False)
    log(f"  pathogenic genes (>= 1 P/LP SNV of any region, inside the gene's MANE transcripts): {len(pg):,}")
    tr = dedupe(c25[c25.in_scope], log, "training")
    tr["canonical_any_gencode"], tr["exonic_any_gencode"] = canonical_any_gencode(tr), exonic_any_gencode(tr)
    tr = hg19_by_id(tr, "2025")
    check_ref(tr, fa, "training")
    tr[cols + ["canonical_any_gencode", "exonic_any_gencode", "chr19", "pos19", "ref19", "alt19"]].to_csv(f"{V2}/data/v2_variants_training.csv", index=False)
    tr_keys, tr_ids = set(key(tr)), set(tr.clinvar_id)
    log(f"  training: {len(tr):,} variants, P/LP {int(tr.label.sum()):,}; GRCh37 record found {tr.pos19.notna().mean():.2%}")
    log("  by region: " + ", ".join(f"{r} {n:,} (P/LP {int(tr.label[tr.region == r].sum()):,})" for r, n in tr.region.value_counts().items()))
    v1 = pd.read_csv(V1_TRAIN, usecols=["clinvar_id", "region", "label"], low_memory=False)
    v1 = v1[~v1.region.str.startswith("Exonic", na=True)]
    log(f"  v1 training (non-exonic, GRCh37, GENCODE v19): {len(v1):,}; ClinVar IDs in both {len(set(v1.clinvar_id) & tr_ids):,}")
    di = tr.region == "Deep Intronic"
    sens = di & ~tr.canonical_any_gencode & ~tr.exonic_any_gencode
    log(f"  Deep Intronic {int(di.sum()):,}; canonical in some GENCODE v50 transcript {int((di & tr.canonical_any_gencode).sum()):,}; "
        f"exonic in some v50 transcript {int((di & tr.exonic_any_gencode).sum()):,}; neither (sensitivity set) {int(sens.sum()):,}")

    # ---- test candidates ----
    log("\nTest candidates: ClinVar 2026-04-15 GRCh38, SNVs, P/LP or B/LB, in scope, less training variants")
    ids25 = set(pd.read_csv(CV38["2025"], sep="\t", usecols=["clinvar_id"]).clinvar_id) | \
            set(pd.read_csv(CV37["2025"], sep="\t", usecols=["clinvar_id"]).clinvar_id)
    c26 = classify(read_clinvar(CV38["2026"], PATHOGENIC | BENIGN))
    te = dedupe(c26[c26.in_scope], log, "test candidates")
    in_tr = key(te).isin(tr_keys) | te.clinvar_id.isin(tr_ids)
    log(f"  in scope {len(te):,}; in the training data {int(in_tr.sum()):,} (dropped)")
    te = te[~in_tr].copy()
    te["subset"] = np.where(te.clinvar_id.isin(ids25), "reclassified", "strict")
    te["canonical_any_gencode"], te["exonic_any_gencode"] = canonical_any_gencode(te), exonic_any_gencode(te)
    te = hg19_by_id(te, "2026")
    check_ref(te, fa, "test candidates")
    te[cols + ["subset", "canonical_any_gencode", "exonic_any_gencode", "chr19", "pos19", "ref19", "alt19"]].to_csv(
        f"{V2}/data/v2_variants_test_candidates.csv", index=False)
    for sub in ["strict", "reclassified"]:
        t = te[te.subset == sub]
        log(f"  {sub}: {len(t):,} variants, P/LP {int(t.label.sum()):,}")
    st = te[te.subset == "strict"]
    log("  strict by region: " + ", ".join(f"{r} {n:,} (P/LP {int(st.label[st.region == r].sum()):,})" for r, n in st.region.value_counts().items()))
    di = st.region == "Deep Intronic"
    sens = di & ~st.canonical_any_gencode & ~st.exonic_any_gencode
    log(f"  strict Deep Intronic {int(di.sum()):,} (P/LP {int(st.label[di].sum())}); sensitivity set (neither canonical nor "
        f"exonic in any GENCODE v50 transcript) {int(sens.sum()):,} (P/LP {int(st.label[sens].sum())})")

    # ---- VUS ----
    log("\nVUS: ClinVar 2026-04-15 GRCh38, uncertain or conflicting, in scope, less training variants")
    cv = classify(read_clinvar(CV38["2026"], UNCERTAIN))
    vu = cv[cv.in_scope].copy()
    vu = vu.loc[~key(vu).duplicated()].reset_index(drop=True)
    in_tr = key(vu).isin(tr_keys) | vu.clinvar_id.isin(tr_ids)
    log(f"  in scope {len(vu):,}; in the training data {int(in_tr.sum()):,} (dropped)")
    vu = vu[~in_tr].copy()
    vu["canonical_any_gencode"], vu["exonic_any_gencode"] = canonical_any_gencode(vu), exonic_any_gencode(vu)
    vu = hg19_by_id(vu, "2026")
    check_ref(vu, fa, "VUS")
    vu[cols + ["canonical_any_gencode", "exonic_any_gencode", "chr19", "pos19", "ref19", "alt19"]].to_csv(f"{V2}/data/v2_variants_vus.csv", index=False)
    log(f"  VUS: {len(vu):,} ({vu.clnsig.value_counts().to_dict()})")

    # ---- literature and SpliceVarDB (GRCh37 sources) ----
    log("\nLiterature and SpliceVarDB: lifted from GRCh37 with the Ensembl chain, in scope, less training variants")
    lit = pd.read_csv(LIT)
    lit = lift_set(lit.rename(columns={"gene": "lit_gene"}), log, "literature")
    lit = classify(lit)
    log(f"  literature: in scope {int(lit.in_scope.sum()):,}; out of scope {lit.out_of_scope_reason[~lit.in_scope].value_counts().to_dict()}")
    lit = lit[lit.in_scope].copy()
    in_tr = key(lit).isin(tr_keys)
    lit = lit[~in_tr].copy()
    lit["label"] = 1.0
    lit["canonical_any_gencode"], lit["exonic_any_gencode"] = canonical_any_gencode(lit), exonic_any_gencode(lit)
    check_ref(lit, fa, "literature")
    lit[KEY + ["label", "region", "gene", "gene_id", "lit_gene", "variant_hgvs", "canonical_any_gencode",
               "exonic_any_gencode", "chr19", "pos19", "ref19", "alt19"]].to_csv(f"{V2}/data/v2_variants_literature.csv", index=False)
    log(f"  literature: in the training data {int(in_tr.sum())} (dropped); {len(lit):,} variants")

    sv = pd.read_csv(SVDB).rename(columns={"Chr": "chr", "Pos": "pos", "Ref": "ref", "Alt": "alt", "Gene": "svdb_gene",
                                           "Classification": "svdb_class", "Dataset": "dataset"})
    sv = sv[sv.ref.isin(list("ACGT")) & sv.alt.isin(list("ACGT"))].reset_index(drop=True)
    sv = lift_set(sv, log, "SpliceVarDB records")
    sv = classify(sv)
    log(f"  SpliceVarDB: in scope {int(sv.in_scope.sum()):,} records; out of scope {sv.out_of_scope_reason[~sv.in_scope].value_counts().to_dict()}")
    sv = sv[sv.in_scope].copy()
    in_tr = key(sv).isin(tr_keys)
    sv = sv[~in_tr].copy()
    sv["canonical_any_gencode"], sv["exonic_any_gencode"] = canonical_any_gencode(sv), exonic_any_gencode(sv)
    check_ref(sv, fa, "SpliceVarDB")
    sv[KEY + ["svdb_class", "dataset", "region", "gene", "gene_id", "svdb_gene", "canonical_any_gencode",
              "exonic_any_gencode", "chr19", "pos19", "ref19", "alt19"]].to_csv(f"{V2}/data/v2_variants_splicevardb.csv", index=False)
    log(f"  SpliceVarDB: in the training data {int(in_tr.sum())} (dropped); {len(sv):,} records, {key(sv).nunique():,} variants; "
        f"classes {sv.svdb_class.value_counts().to_dict()}")

    fa.close()
    log("\nEvery REF allele equals the GRCh38 base; no test, VUS, literature or SpliceVarDB variant is in the training data.")
    out.close()


if __name__ == "__main__":
    main()
