"""R01 -- Three-column gene-level extract of dbNSFP 5.4 for the MANE genes (Release 2.0; docs/V2_RELEASE_PLAN.md).

The scorer's three gene-level features (gnomAD_pLI, gnomAD_LOEUF, ClinGen_Haploinsufficiency_Score) come from the
dbNSFP 5.4 gene file. Rebuilding them from gnomAD and ClinGen does not reproduce the model's scores (docs/FEATURE_LICENCES.md,
addendum 2026-10-10), so Release 2.0 ships these three columns from dbNSFP 5.4, with the permission the dbNSFP licensor
granted on 2026-10-10.

The extract has one row per Ensembl gene ID of the GENCODE v50 MANE annotation that the gene file lists, holding exactly
what score_v2.compute_gene_level reads: the first ';'-separated value of each column, with the last row winning when an
ID appears in several rows. Values keep their original text ('.' when missing). The file is in the gene file's column
layout, so the scorer reads it unchanged through --dbnsfp-gene.

Checks before writing: the gene file's md5 equals V00's record; and score_v2.compute_gene_level returns identical values
(NaN equal to NaN) from the extract and from the full gene file for every MANE gene.

Writes MetaSplice/data/metasplice_gene_level_dbNSFP5.4.tsv.gz (deterministic gzip) and prints its md5.
"""
import gzip, hashlib, importlib.util, io, os, re
import numpy as np

P = "/mnt/Work0/Splice_altering_variant"
GENE_FILE = f"{P}/reproducibility/raw_inputs/dbNSFP5.4_gene.gz"
GENE_MD5 = "8447df61912ac03fbc0004b753b8bcf3"          # V00_inputs.txt, dbNSFP 5.4 gene file
MANE_GTF = f"{P}/annotation_v2/gencode.v50.mane.gtf.gz"
OUT = f"{P}/MetaSplice/data/metasplice_gene_level_dbNSFP5.4.tsv.gz"
COLS = ["gnomAD_pLI", "gnomAD_LOEUF", "ClinGen_Haploinsufficiency_Score"]

spec = importlib.util.spec_from_file_location("score_v2", f"{P}/MetaSplice/score_v2.py")
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)
assert list(ms.GENE_LEVEL) == COLS


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    got = md5(GENE_FILE)
    if got != GENE_MD5:
        raise SystemExit(f"{GENE_FILE}: md5 {got}, V00 recorded {GENE_MD5}")
    mane = {}
    with gzip.open(MANE_GTF, "rt") as f:
        for line in f:
            if "\tgene\t" in line:
                gid = re.search(r'gene_id "([^"]+)"', line).group(1)
                name = re.search(r'gene_name "([^"]+)"', line).group(1)
                mane[gid.split(".")[0]] = (name, gid)
    # what the scorer reads: first ';' value per column, last row wins per Ensembl ID
    rows = {}
    with gzip.open(GENE_FILE, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        ci = {c: header.index(c) for c in COLS}
        ens, name = header.index("Ensembl_gene"), header.index("Gene_name")
        for line in f:
            fs = line.rstrip("\n").split("\t")
            vals = [fs[ci[c]].split(";")[0] for c in COLS]
            for eid in fs[ens].split(";"):
                if eid not in (".", ""):
                    rows[eid] = (fs[name], vals)
    keep = sorted(e for e in rows if e in mane)
    print(f"MANE genes {len(mane):,}; listed in the gene file {len(keep):,}")
    buf = io.StringIO()
    buf.write("\t".join(["Gene_name", "Ensembl_gene"] + COLS) + "\n")
    for e in keep:
        buf.write("\t".join([rows[e][0], e] + rows[e][1]) + "\n")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(buf.getvalue().encode())
    # identical values through the scorer's own reader, for every MANE gene
    genes = {i: (n, gid) for i, (n, gid) in enumerate(mane.values())}
    a, b = ms.compute_gene_level(genes, GENE_FILE), ms.compute_gene_level(genes, OUT)
    if set(a) != set(b):
        raise SystemExit(f"genes with values differ: {len(set(a) ^ set(b))}")
    bad = [i for i in a for c in COLS
           if not ((np.isnan(a[i][c]) and np.isnan(b[i][c])) or a[i][c] == b[i][c])]
    if bad:
        raise SystemExit(f"{len(bad)} values differ between the extract and the gene file")
    print(f"identical through score_v2.compute_gene_level for all {len(genes):,} MANE genes ({len(a):,} with a row)")
    print(f"wrote {OUT} ({os.path.getsize(OUT):,} bytes; md5 {md5(OUT)})")


if __name__ == "__main__":
    main()
