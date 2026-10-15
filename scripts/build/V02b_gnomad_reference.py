"""V02b -- gnomAD v4.1 population reference of MetaSplice v2.0, native GRCh38
(docs/METASPLICE_V2_PLAN.md, Task 3, Step 3).

Eligible population, as S11a but on GRCh38 with the v2.0 scope (no liftover is needed any more):
gnomAD v4.1 joint SNVs on chromosomes 1-22 and X (as S11a), FILTER PASS, 0 < AF < 0.01, not in
either ClinVar release (2025-08-31 or 2026-04-15, GRCh38, any classification), in scope by the v2.0
scorer's scope() on the full GENCODE v50 GTF (MANE regions; inside a MANE transcript of the assigned gene,
O1; Deep Intronic not inside a GENCODE basic exon, O5) in one of the five non-canonical regions, in a
gene with >= 1 P/LP SNV in the training release (data/v2_pathogenic_genes.csv, from V02).

Two vectorized prefilters make the exact classification affordable; they only remove variants the
exact rule would also put out of scope: positions outside every MANE transcript span, and positions
inside a MANE exon (the exon-overlap filter makes those Exonic).

Samples (decision O3):
  primary    uniform, 100,000 variants, seed 42          -> data/v2_variants_gnomad_reference.csv
  secondary  10,000 per non-canonical region, seed 43     -> data/v2_variants_gnomad_reference_stratified.csv
             (all of a region that has fewer)
Writes results/V02b_gnomad_reference.txt with the eligible population by region.
"""
import gzip, importlib.util, os, re, sys, time
import numpy as np
import pandas as pd
import pysam

P = "/mnt/Work0/Splice_altering_variant"
V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANE = f"{P}/annotation_v2/gencode.v50.mane.gtf.gz"
FULL = f"{P}/docker_data/gencode.v50.annotation.gtf.gz"
HG38 = f"{P}/hg38/hg38.fa"
GNOMAD = f"{P}/docker_data/gnomAD/gnomad.joint.v4.1.sites.chr{{}}.vcf.bgz.snp.tsv.gz"
CV38 = [f"{P}/clinvar/hg38/clinvar_20250831.vcf.gz.summary.gz",
        f"{P}/clinvar20260418/hg38/clinvar_20260415.vcf.gz.summary.gz"]
CHROMS = [str(i) for i in range(1, 23)] + ["X"]
NONCANONICAL = ["Donor Region", "Acceptor Region", "PPT Region", "Branch Point Region", "Deep Intronic"]
N_UNIFORM, SEED_UNIFORM, N_PER_REGION, SEED_STRATIFIED = 100_000, 42, 10_000, 43

spec = importlib.util.spec_from_file_location("score_v2", f"{P}/MetaSplice/score_v2.py")
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)
v02spec = importlib.util.spec_from_file_location("v02", os.path.join(os.path.dirname(os.path.abspath(__file__)), "V02_variant_sets.py"))
v02 = importlib.util.module_from_spec(v02spec)
v02spec.loader.exec_module(v02)


def merged(intervals):
    """Sorted, merged (starts, ends) arrays."""
    a = np.array(sorted(intervals), dtype=np.int64)
    starts, ends = [a[0, 0]], [a[0, 1]]
    for s, e in a[1:]:
        if s <= ends[-1] + 1:
            ends[-1] = max(ends[-1], e)
        else:
            starts.append(s)
            ends.append(e)
    return np.array(starts), np.array(ends)


def inside(pos, starts, ends):
    j = np.searchsorted(starts, pos, side="right") - 1
    return (j >= 0) & (pos <= ends[np.clip(j, 0, None)])


def main():
    out = open(f"{V2}/results/V02b_gnomad_reference.txt", "w")
    def log(s=""):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    log("V02b -- gnomAD v4.1 population reference, GRCh38, GENCODE v50 MANE scope")
    g = ms.parse_mane_gtf(FULL)
    spans = {}
    for gid, sp in g["spans"].items():
        for c, s, e in sp:
            spans.setdefault(c, []).append((s, e))
    spans = {c: merged(v) for c, v in spans.items()}
    exons = {c: merged([(s, e) for s, e, _, _ in v]) for c, v in g["exons_by_chr"].items()}
    pgenes = set(pd.read_csv(f"{V2}/data/v2_pathogenic_genes.csv").gene_id)
    clinvar = set()
    for f in CV38:
        c = pd.read_csv(f, sep="\t", usecols=["#chr", "pos", "ref", "alt"], dtype={"#chr": str, "ref": str, "alt": str})
        clinvar |= set("chr" + c["#chr"] + ":" + c.pos.astype(str) + ":" + c.ref + ":" + c.alt)
    log(f"MANE transcript spans on {len(spans)} chromosomes; pathogenic genes {len(pgenes):,}; ClinVar variants (either release) {len(clinvar):,}")
    counts, keep = {}, []
    add = lambda k, n: counts.__setitem__(k, counts.get(k, 0) + int(n))
    for chrom in CHROMS:
        t0, c = time.time(), f"chr{chrom}"
        d = pd.read_csv(GNOMAD.format(chrom), sep="\t", usecols=["pos", "ref", "alt", "gnomAD_joint_flag", "gnomAD_joint_AF"],
                        dtype={"ref": str, "alt": str})
        add("gnomAD SNVs read", len(d))
        d = d[(d.gnomAD_joint_flag == ".") & (d.gnomAD_joint_AF > 0) & (d.gnomAD_joint_AF < 0.01)]
        add("PASS, 0 < AF < 0.01", len(d))
        pos = d.pos.values
        m = inside(pos, *spans[c]) if c in spans else np.zeros(len(d), bool)
        add("outside every MANE transcript span (prefilter)", (~m).sum())
        d, pos = d[m], pos[m]
        m = inside(pos, *exons[c])
        add("inside a MANE exon (prefilter)", m.sum())
        d = d[~m]
        in_cv = (c + ":" + d.pos.astype(str) + ":" + d.ref + ":" + d.alt).isin(clinvar)
        add("in a ClinVar release", in_cv.sum())
        d = d[~in_cv].reset_index(drop=True)
        v = [(c, p, r, a) for p, r, a in zip(d.pos.values.tolist(), d.ref.tolist(), d.alt.tolist())]
        in_scope, genes, why = ms.scope(v, FULL)
        region = pd.Series([in_scope.get(i) for i in range(len(v))], dtype=object)
        gid = pd.Series([genes.get(i, (None, None))[1] for i in range(len(v))], dtype=object)
        gname = pd.Series([genes.get(i, (None, None))[0] for i in range(len(v))], dtype=object)
        for reason, k in pd.Series(list(why.values()), dtype=object).value_counts().items():
            add(f"out of scope: {reason}", k)
        add("in scope, canonical region (not in the non-canonical reference)", region.isin(ms.PRIORITY.keys() - set(NONCANONICAL)).sum())
        elig = region.isin(NONCANONICAL)
        add("in a gene without P/LP in training", (elig & ~gid.isin(pgenes)).sum())
        elig &= gid.isin(pgenes)
        keep.append(pd.DataFrame({"chr": c, "pos": d.pos.values[elig.values], "ref": d.ref.values[elig.values],
                                  "alt": d.alt.values[elig.values], "gnomad_af": d.gnomAD_joint_AF.values[elig.values],
                                  "region": region[elig].values, "gene": gname[elig].values, "gene_id": gid[elig].values}))
        log(f"{c}: eligible {int(elig.sum()):,} ({time.time() - t0:.0f} s)")
    allv = pd.concat(keep, ignore_index=True)
    log("counts: " + "; ".join(f"{k} {v:,}" for k, v in counts.items()))
    log(f"eligible {len(allv):,}; by region: " + ", ".join(f"{r} {n:,}" for r, n in allv.region.value_counts().items()))
    uni = allv.sample(n=min(N_UNIFORM, len(allv)), random_state=SEED_UNIFORM)
    strat = pd.concat([sub.sample(n=min(N_PER_REGION, len(sub)), random_state=SEED_STRATIFIED)
                       for _, sub in allv.groupby("region", sort=True)])
    fa = pysam.FastaFile(HG38)
    for name, s, path in [("uniform", uni, "v2_variants_gnomad_reference.csv"),
                          ("stratified", strat, "v2_variants_gnomad_reference_stratified.csv")]:
        s = s.sort_values(["chr", "pos"]).reset_index(drop=True)
        v02.check_ref(s, fa, f"gnomAD {name} sample")
        s["canonical_any_gencode"], s["exonic_any_gencode"] = v02.canonical_any_gencode(s), v02.exonic_any_gencode(s)
        s.to_csv(f"{V2}/data/{path}", index=False)
        log(f"{name} sample: {len(s):,}; by region: " + ", ".join(f"{r} {n:,}" for r, n in s.region.value_counts().items())
            + f"; Deep Intronic canonical or exonic in some GENCODE v50 transcript "
            + f"{int((s.region.eq('Deep Intronic') & (s.canonical_any_gencode | s.exonic_any_gencode)).sum())}")
    fa.close()
    log("Every REF allele of both samples equals the GRCh38 base.")
    out.close()


if __name__ == "__main__":
    main()
