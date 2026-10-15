# metasplice_gene_level_dbNSFP5.4.tsv.gz

Gene-level inputs of MetaSplice Release 2.0: three values for 19,230 of the 19,299 GENCODE v50 MANE genes. md5
`088ba4d5101c01a8697fbbf503affd06`.

| Column | Meaning | Source (CC0) |
|---|---|---|
| `gnomAD_pLI` | Probability of loss-of-function intolerance | gnomAD v2.1.1, `lof_metrics.by_gene` |
| `gnomAD_LOEUF` | Loss-of-function observed/expected upper bound fraction | gnomAD v4.1, `constraint_metrics` (MANE Select transcript) |
| `ClinGen_Haploinsufficiency_Score` | ClinGen dosage-sensitivity haploinsufficiency score | ClinGen gene curation list of 28 July 2026 |

Values keep their original text, and `.` marks a missing value.

**How it is built.** `build_gene_level.py`, in `data/` of the MetaSplice repository, makes this file from its
sources:

    python build_gene_level.py --gtf /path/to/metasplice_v2_data/gencode/gencode.v50.annotation.gtf.gz

- Genes are HGNC's, from its gene list of 28 July 2026 (`sources/hgnc_ensembl_20260728.tsv.gz`: HGNC ID, symbol, and
  the Ensembl gene IDs given by HGNC and by Ensembl). A gene is listed if HGNC gives it an Ensembl gene ID, and it is
  known by both Ensembl IDs.
- Each value is matched to the gene by its symbol. LOEUF is taken only from MANE Select rows without constraint flags.
- The file keeps the genes of GENCODE v50 MANE Select and MANE Plus Clinical transcripts. The Ensembl IDs of the other
  69 MANE genes are not among those of the listed genes, so they have no row, and the scorer leaves their three values
  missing.

The script downloads the two gnomAD files from gnomAD's public bucket, checks every input against its md5, and reports
whether its output is identical to this file. It needs Python 3.6 or newer and nothing else.

These are the rules of the dbNSFP 5.4 gene file, and the values are its values: this file was first extracted from that
gene file, and building it from the sources gives the same bytes.

**Terms.** gnomAD, ClinGen and HGNC release these data under CC0. The file is distributed as part of MetaSplice
(`LICENSE`).

**Please cite** gnomAD (Karczewski KJ, et al. Nature. 2020;581:434-43, for pLI; Chen S, et al. Nature.
2024;625:92-100, for LOEUF), ClinGen (https://clinicalgenome.org), HGNC (https://www.genenames.org) and dbNSFP (Liu X,
et al. Genome Med. 2020;12:103).

**Use.** The scorer reads this file through `--dbnsfp-gene` (or `METASPLICE_DBNSFP`). The Release 2.0 Docker image uses
it by default.
