"""V01b -- Verify SpliceTransformer's MANE annotation (docs/METASPLICE_V2_PLAN.md, Task 1, Step 5).

SpliceTransformer finds the genes at a variant with pyensembl's genes_at_locus, which uses gene
spans (sptransformer.py:68-72), and it names its pyensembl database gencode.v38 whatever GTF it is
given (sptransformer.py:31-35). pyensembl 2.6.2 builds that database next to the GTF and reuses one
it finds there without checking it. V01 therefore resets MANE gene spans to the extent of their
MANE transcripts and removes any stale database from the run directory; this script builds the
database with SpliceTransformer's own call, from the run directory, and checks it:

  1. the database lies inside the run directory, holds every MANE gene, and each gene's span is
     the extent of its MANE transcripts;
  2. genes_at_locus, the call SpliceTransformer makes, returns exactly the genes and strands the
     MANE GTF implies, at
       A. 200 positions 1-50 bp beyond the end of a MANE transcript that lie outside every MANE
          transcript span of any gene (no gene expected);
       B. 200 positions 1-50 bp inside the end of a MANE transcript;
       C. 200 positions where two MANE genes overlap on opposite strands (both strands expected);
  3. SpliceTransformer gives every group A position a score of 0 and all 15 tissue values 0;
  4. positive control, since check 3 would also pass if the run directory could not score at all:
     group B scored in the run directory gives nonzero values, and equals SpliceTransformer's own
     package (GENCODE v38) wherever both annotations give the same strands.

Appends to results/V01_annotation.txt; exits 1 if any check fails.

Usage:  python V01b_spt_annotation_check.py
"""
import gzip, os, random, re, subprocess, sys
import pandas as pd
import pysam

P = "/mnt/Work0/Splice_altering_variant"
V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANNOT = f"{P}/annotation_v2"
RUN = f"{ANNOT}/spt_run"
MANE = f"{ANNOT}/gencode.v50.mane.gtf.gz"
GENE_ID, GENE_NAME = re.compile(r'gene_id "([^"]+)"'), re.compile(r'gene_name "([^"]+)"')
N = 200
failures = []


def check(log, name, ok, detail=""):
    log(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures.append(name)


def mane_transcripts():
    rows = []
    with gzip.open(MANE, "rt") as f:
        for line in f:
            x = line.split("\t", 8)
            if not line.startswith("#") and x[2] == "transcript":
                rows.append((x[0], int(x[3]), int(x[4]), x[6], GENE_ID.search(x[8]).group(1), GENE_NAME.search(x[8]).group(1)))
    return pd.DataFrame(rows, columns=["chr", "start", "end", "strand", "gene_id", "gene"])


def expected(genes, chrom, pos):
    g = genes[(genes.chr == chrom) & (genes.start <= pos) & (genes.end >= pos)]
    return set(zip(g.gene, g.strand))


def pick_groups(tx, genes, rng):
    by_chr = {c: g for c, g in genes.groupby("chr")}
    exp = lambda c, p: expected(by_chr.get(c, genes.iloc[:0]), c, p)
    a, b = [], []
    for r in tx.sample(frac=1, random_state=11).itertuples():
        k = rng.randint(1, 50)
        if len(a) < N:
            pos = r.start - k if rng.random() < 0.5 else r.end + k
            if pos > 0 and not exp(r.chr, pos):
                a.append((r.chr, pos, set()))
        if len(b) < N and r.end - r.start > 100:
            pos = r.start + k if rng.random() < 0.5 else r.end - k
            b.append((r.chr, pos, exp(r.chr, pos)))
        if len(a) >= N and len(b) >= N:
            break
    pairs = []
    for c, g in by_chr.items():
        g = g.sort_values("start").reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if g.start[j] > g.end[i]:
                    break
                if g.strand[i] != g.strand[j]:
                    pairs.append((c, max(g.start[i], g.start[j]), min(g.end[i], g.end[j])))
    rng.shuffle(pairs)
    c_ = [(ch, pos, exp(ch, pos)) for ch, s, e in pairs[:N] for pos in [rng.randint(s, e)]]
    return a, b, c_, len(pairs)


def main():
    out = open(f"{V2}/results/V01_annotation.txt", "a")
    def log(s=""):
        print(s); out.write(s + "\n")
    log("\nV01b -- SpliceTransformer annotation checks (Task 1, Step 5)")
    os.chdir(RUN)
    os.environ["PYENSEMBL_CACHE_DIR"] = f"{RUN}/pyensembl_cache"
    from pyensembl import Genome
    genome = Genome(reference_name="hg38", annotation_name="gencode.v38",          # SpliceTransformer's call
                    gtf_path_or_url="data/data_package/hg38.annotation.gtf.gz")
    genome.index()
    tx = mane_transcripts()
    genes = tx.groupby(["gene_id", "gene", "chr", "strand"]).agg(start=("start", "min"), end=("end", "max")).reset_index()

    log("1. Database")
    path = os.path.realpath(genome.db.local_db_path)
    check(log, "database built inside the run directory", path.startswith(os.path.realpath(RUN)), path)
    db_genes = {g.gene_id: (g.start, g.end, g.strand) for g in genome.genes()}
    check(log, "every MANE gene, and nothing else", set(db_genes) == set(genes.gene_id),
          f"{len(db_genes):,} in the database, {len(genes):,} MANE genes")
    bad = [r.gene_id for r in genes.itertuples() if db_genes.get(r.gene_id, (None,) * 3) != (r.start, r.end, r.strand)]
    check(log, "gene spans equal the extent of their MANE transcripts", not bad, f"{len(bad)} differ")

    log("2. genes_at_locus against the MANE GTF")
    rng = random.Random(11)
    a, b, c, npairs = pick_groups(tx, genes, rng)
    log(f"  opposite-strand overlapping MANE gene pairs: {npairs:,}")
    for label, grp in [("A. 1-50 bp beyond a MANE transcript, outside every MANE transcript span", a),
                       ("B. 1-50 bp inside the end of a MANE transcript", b),
                       ("C. overlap of two MANE genes on opposite strands", c)]:
        got = [{(g.gene_name, g.strand) for g in genome.genes_at_locus(ch, pos)} for ch, pos, _ in grp]
        wrong = [(ch, pos, e, g) for (ch, pos, e), g in zip(grp, got) if e != g]
        extra = f"; two strands at {sum(len({s for _, s in e}) == 2 for _, _, e in grp)}" if label.startswith("C") else ""
        check(log, f"{label}: n={len(grp)}", len(grp) == N and not wrong,
              f"{len(wrong)} differ" + extra + (f"; first {wrong[0]}" if wrong else ""))

    log("3. SpliceTransformer on group A")
    fa = pysam.FastaFile(f"{P}/hg38/hg38.fa")
    vcf = f"{RUN}/groupA.vcf"
    with open(vcf, "w") as f:
        f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for ch, pos, _ in sorted(a):
            ref = fa.fetch(ch, pos - 1, pos).upper()
            f.write(f"{ch}\t{pos}\t.\t{ref}\t{'A' if ref != 'A' else 'C'}\t.\tPASS\t.\n")
    res = subprocess.run([sys.executable, f"{P}/SpliceTransformer/sptransformer.py", "--input", vcf,
                          "--output", f"{RUN}/groupA_spt.csv", "--reference", "hg38", "--vcf", "True",
                          "--raw_score", "True"], capture_output=True, text=True)
    check(log, "SpliceTransformer ran", res.returncode == 0, res.stderr.strip().splitlines()[-1] if res.returncode else "")
    if res.returncode == 0:
        s = pd.read_csv(f"{RUN}/groupA_spt.csv", index_col=0)
        vals = s.drop(columns=["#CHROM", "POS", "ID", "REF", "ALT"]).astype(float)
        check(log, f"group A scored 0 with all tissues 0: n={len(s)}", len(s) == N and (vals == 0).all(axis=None),
              f"{int((vals != 0).any(axis=1).sum())} nonzero")
    log("4. Positive control on group B")
    vcfb = f"{RUN}/groupB.vcf"
    with open(vcfb, "w") as f:
        f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for ch, pos, _ in sorted(b):
            ref = fa.fetch(ch, pos - 1, pos).upper()
            f.write(f"{ch}\t{pos}\t.\t{ref}\t{'A' if ref != 'A' else 'C'}\t.\tPASS\t.\n")
    runs = {}
    for name, cwd in [("MANE run directory", RUN), ("SpliceTransformer package (v38)", f"{P}/SpliceTransformer")]:
        o = f"{RUN}/groupB_spt_{'mane' if cwd == RUN else 'v38'}.csv"
        r = subprocess.run([sys.executable, f"{P}/SpliceTransformer/sptransformer.py", "--input", vcfb, "--output", o,
                            "--reference", "hg38", "--vcf", "True", "--raw_score", "True"], cwd=cwd, capture_output=True, text=True)
        check(log, f"SpliceTransformer ran from the {name}", r.returncode == 0, r.stderr.strip().splitlines()[-1] if r.returncode else "")
        if r.returncode == 0:
            runs[name] = pd.read_csv(o, index_col=0).set_index(["#CHROM", "POS"]).drop(columns=["ID", "REF", "ALT"]).astype(float)
    if len(runs) == 2:
        m, v = runs["MANE run directory"], runs["SpliceTransformer package (v38)"].reindex(runs["MANE run directory"].index)
        check(log, f"group B has nonzero scores in the run directory", (m != 0).any(axis=1).mean() > 0.5,
              f"{int((m != 0).any(axis=1).sum())} of {len(m)} nonzero")
        v38 = Genome(reference_name="hg38", annotation_name="gencode.v38",
                     gtf_path_or_url=f"{P}/SpliceTransformer/data/data_package/hg38.annotation.gtf.gz")
        same = [{g.strand for g in genome.genes_at_locus(ch, pos)} == {g.strand for g in v38.genes_at_locus(ch, pos)}
                for ch, pos in m.index]
        eq = (m.values == v.values).all(axis=1)
        check(log, "equal to the package wherever both annotations give the same strands",
              bool(eq[same].all()), f"same strands at {sum(same)}: identical {int(eq[same].sum())}; "
              f"different strands at {len(same) - sum(same)}: identical {int(eq[[not x for x in same]].sum())}")
    log(f"V01b: {'ALL CHECKS PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    out.close()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
