#!/usr/bin/env python3
"""Build MetaSplice's gene-level values (metasplice_gene_level_dbNSFP5.4.tsv.gz) from their original sources.

  python build_gene_level.py --gtf /path/to/metasplice_v2_data/gencode/gencode.v50.annotation.gtf.gz \\
      [--gnomad-dir DIR] [--out metasplice_gene_level_dbNSFP5.4.tsv.gz]

Three values per gene, all released under CC0 by their sources:
  gnomAD_pLI                        gnomAD v2.1.1 gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz, column pLI
  gnomAD_LOEUF                      gnomAD v4.1 gnomad.v4.1.constraint_metrics.tsv, column lof.oe_ci.upper, from rows
                                    that are the MANE Select transcript, have an Ensembl gene ID and carry no
                                    constraint flags
  ClinGen_Haploinsufficiency_Score  ClinGen gene curation list of 28 July 2026 (sources/), column Haploinsufficiency
                                    Score
Genes are HGNC's (sources/hgnc_ensembl_20260728.tsv.gz: HGNC ID, symbol, HGNC's Ensembl gene ID and Ensembl's, from
HGNC on 28 July 2026). A gene is listed if HGNC gives it an Ensembl gene ID; it is known by both Ensembl IDs. Each
value is matched by gene symbol. Where an Ensembl ID belongs to more than one gene, the later gene in HGNC's order is
kept. The file covers the genes of GENCODE v50 MANE Select and MANE Plus Clinical transcripts (from --gtf). These
are the rules of dbNSFP 5.4's gene file, whose values for these genes the output reproduces.

The two gnomAD files are read from --gnomad-dir, or downloaded there from gnomAD's public bucket. Every input and
the output are checked against their md5; the script stops if an input differs and says so if the output does.
Python 3.6 or newer, standard library only.
"""
import argparse, gzip, hashlib, io, os, re, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = os.path.join(HERE, "sources")
GNOMAD = {
    "gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz": (
        "https://storage.googleapis.com/gcp-public-data--gnomad/release/2.1.1/constraint/"
        "gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz", "967fb59dc509d9ee351b9d35efacf52a"),
    "gnomad.v4.1.constraint_metrics.tsv": (
        "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/constraint/"
        "gnomad.v4.1.constraint_metrics.tsv", "14df4b2acb581fcbbb2a82a3a555fd35"),
}
HGNC = ("hgnc_ensembl_20260728.tsv.gz", "c1e2209cf4eca9dd5643c67b17bb2648")
CLINGEN = ("ClinGen_gene_curation_list_GRCh38_20260728.tsv", "40be4351db8511b53428bc2e56da02d4")
COLS = ["gnomAD_pLI", "gnomAD_LOEUF", "ClinGen_Haploinsufficiency_Score"]
OUT_MD5 = "088ba4d5101c01a8697fbbf503affd06"     # the file shipped with MetaSplice 2.0.0


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()


def checked(path, want):
    got = md5(path)
    if got != want:
        sys.exit(f"{path}: md5 {got}, expected {want}")
    return path


def fields(line, n, default=""):
    """The first n tab-separated fields, padded; an empty field becomes `default`."""
    f = (line.rstrip("\r\n").split("\t") + [""] * n)[:n]
    return [x if x != "" else default for x in f]


def genes_hgnc(path):
    """[(symbol, [Ensembl IDs])] in HGNC's order, for genes HGNC gives an Ensembl gene ID."""
    out = []
    with gzip.open(path, "rt") as f:
        f.readline()
        for line in f:
            _, symbol, hgnc_ens, ens_ens = fields(line, 4, ".")
            if not hgnc_ens.startswith("ENSG"):
                continue
            ids = []
            for x in hgnc_ens.split(",") + ens_ens.split(","):
                x = x.replace('"', "").replace(" ", "")
                if x and x != "." and x not in ids:
                    ids.append(x)
            out.append((symbol.replace('"', "").replace(" ", ""), ids))
    return out


def pli_by_symbol(path):
    """pLI by gene symbol; the file has no empty fields, and a later row of the same symbol wins."""
    out = {}
    with gzip.open(path, "rt") as f:
        idx = f.readline().rstrip("\n").split("\t").index("pLI")
        for line in f:
            t = [x for x in line.rstrip("\r\n").split("\t") if x != ""]
            out[t[0]] = t[idx]
    return out


def loeuf_by_symbol(path):
    """LOEUF by gene symbol, from MANE Select rows with an Ensembl gene ID and no constraint flags."""
    out = {}
    with open(path) as f:
        h = f.readline().rstrip("\n").split("\t")
        i = {c: h.index(c) for c in ("gene", "gene_id", "mane_select", "lof.oe_ci.upper", "constraint_flags")}
        for line in f:
            t = fields(line, len(h), ".")
            if t[i["constraint_flags"]] == "[]" and t[i["mane_select"]] == "true" and t[i["gene_id"]].startswith("ENSG"):
                out[t[i["gene"]]] = t[i["lof.oe_ci.upper"]]
    return out


def clingen_by_symbol(path):
    """Haploinsufficiency score by gene symbol."""
    out = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                t = fields(line, 23)
                out[t[0]] = t[4]
    return out


def mane_genes(gtf):
    """Ensembl gene IDs (no version) of the GENCODE transcripts tagged MANE Select or MANE Plus Clinical."""
    out = set()
    with gzip.open(gtf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 9)
            if p[2] == "transcript" and ('tag "MANE_Select"' in p[8] or 'tag "MANE_Plus_Clinical"' in p[8]):
                out.add(re.search(r'gene_id "([^".]+)', p[8]).group(1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gtf", required=True, help="GENCODE v50 comprehensive GTF (the data bundle's gencode/)")
    ap.add_argument("--gnomad-dir", default=os.path.join(SOURCES, "gnomad"),
                    help="where the two gnomAD files are, or are downloaded to (default sources/gnomad)")
    ap.add_argument("--out", default=os.path.join(HERE, "metasplice_gene_level_dbNSFP5.4.tsv.gz"))
    args = ap.parse_args()

    os.makedirs(args.gnomad_dir, exist_ok=True)
    gnomad = {}
    for name, (url, want) in GNOMAD.items():
        path = os.path.join(args.gnomad_dir, name)
        if not os.path.exists(path):
            print(f"downloading {url}", flush=True)
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
        gnomad[name] = checked(path, want)
    hgnc = checked(os.path.join(SOURCES, HGNC[0]), HGNC[1])
    clingen = checked(os.path.join(SOURCES, CLINGEN[0]), CLINGEN[1])

    genes = genes_hgnc(hgnc)
    pli = pli_by_symbol(gnomad["gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz"])
    loeuf = loeuf_by_symbol(gnomad["gnomad.v4.1.constraint_metrics.tsv"])
    hi = clingen_by_symbol(clingen)
    rows = {}                                        # Ensembl ID -> (symbol, values); a later gene wins
    for symbol, ids in genes:
        h = hi.get(symbol)
        if h is None:                                # dbNSFP also tries the gene's Ensembl IDs as keys
            h = next((hi[e] for e in ids if e in hi), ".")
        vals = [v.split(";")[0] for v in (pli.get(symbol, "."), loeuf.get(symbol, "."), h)]
        for e in ids:
            rows[e] = (symbol, vals)
    mane = mane_genes(args.gtf)
    keep = sorted(e for e in rows if e in mane)

    buf = io.StringIO()
    buf.write("\t".join(["Gene_name", "Ensembl_gene"] + COLS) + "\n")
    for e in keep:
        buf.write("\t".join([rows[e][0], e] + rows[e][1]) + "\n")
    with open(args.out, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(buf.getvalue().encode())
    got = md5(args.out)
    print(f"HGNC genes with an Ensembl ID {len(genes):,}; MANE genes {len(mane):,}, of which listed {len(keep):,}")
    print(f"wrote {args.out} (md5 {got}): "
          + ("identical to the file shipped with MetaSplice 2.0.0" if got == OUT_MD5
             else f"DIFFERS from the file shipped with MetaSplice 2.0.0 (md5 {OUT_MD5})"))
    return 0 if got == OUT_MD5 else 1


if __name__ == "__main__":
    sys.exit(main())
