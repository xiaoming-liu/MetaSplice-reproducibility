"""V00 -- Manifest of every raw input of MetaSplice v2.0 (docs/METASPLICE_V2_PLAN.md, Task 1).

Records the path, size and md5 of each input file, and of every file inside the input
directories, in results/V00_inputs.txt. A later change to any input shows up as a changed md5.

Usage:  python V00_inputs.py [--workers 8]
"""
import argparse, hashlib, os
from concurrent.futures import ProcessPoolExecutor

P = "/mnt/Work0/Splice_altering_variant"
V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUTS = {
    "GENCODE v50 GTF": f"{P}/docker_data/gencode.v50.annotation.gtf.gz",
    "ClinVar GRCh38 2025-08-31": f"{P}/clinvar/hg38/clinvar_20250831.vcf.gz.summary.gz",
    "ClinVar GRCh38 2026-04-15": f"{P}/clinvar20260418/hg38/clinvar_20260415.vcf.gz.summary.gz",
    "ClinVar GRCh37 2025-08-31": f"{P}/reproducibility/raw_inputs/clinvar_20250831.vcf.gz.summary.gz",
    "ClinVar GRCh37 2026-04-15": f"{P}/reproducibility_manuscript/data/clinvar_releases/clinvar_20260415_hg19.vcf.gz.summary.gz",
    "GRCh38 FASTA": f"{P}/hg38/hg38.fa",
    "GRCh37 FASTA": f"{P}/hg19/hg19.fa",
    "dbNSFP 5.4 gene file": f"{P}/reproducibility/raw_inputs/dbNSFP5.4_gene.gz",
    "GPN-MSA": f"{P}/reproducibility/raw_inputs/gpn_msa/scores.tsv.bgz",
    "phyloP GRCh38": f"{P}/reproducibility/raw_inputs/phyloP_GRCh38",
    "phastCons GRCh38": f"{P}/reproducibility/raw_inputs/phastCons_GRCh38",
    "DANN (hg19)": f"{P}/reproducibility/raw_inputs/DANN",
    "gnomAD v4.1 joint": f"{P}/reproducibility/raw_inputs/gnomAD",
    "Ensembl chain tables hg19->hg38": f"{P}/hg19tohg38_ensembl",
    "Ensembl chain tables hg38->hg19": f"{P}/hg38tohg19_ensembl",
    "SpliceAI v1.3 (hg19, comparator)": f"{P}/spliceAI_v1.3",
    "CADD v1.7 GRCh38 (comparator)": f"{P}/CADDv1.7/hg38/whole_genome_SNVs.tsv.gz",
    "SpliceVarDB (hg19)": f"{P}/SpliceVarDB_replicate_20260316/variants_with_position_classification.fixed.no_sakaguchi.csv",
    "Literature set (hg19)": f"{P}/reproducibility_manuscript/data/literature_validation/mapped_variants.csv",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
    return path, h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    rows, files = [], []
    for name, path in INPUTS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name}: {path}")
        members = ([os.path.join(path, f) for f in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, f))]
                   if os.path.isdir(path) else [path])
        rows += [(name, m) for m in members]
        files += members
    with ProcessPoolExecutor(args.workers) as ex:
        sums = dict(ex.map(md5, files))
    with open(f"{V2}/results/V00_inputs.txt", "w") as out:
        out.write("V00 -- raw inputs of MetaSplice v2.0: md5, size in bytes, path (realpath)\n")
        for name, m in rows:
            out.write(f"{name}\t{sums[m]}\t{os.path.getsize(m)}\t{m}\t{os.path.realpath(m)}\n")
    print(f"{len(files):,} files from {len(INPUTS)} inputs; {sum(os.path.getsize(f) for f in files) / 1e9:,.1f} GB")


if __name__ == "__main__":
    main()
