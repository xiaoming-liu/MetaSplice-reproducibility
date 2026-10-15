"""V01 -- GENCODE v50 MANE annotation for MetaSplice v2.0 and the tool databases built from it
(docs/METASPLICE_V2_PLAN.md, Task 1).

MANE Select and MANE Plus Clinical transcripts, with their gene lines, are the one annotation of
v2.0: regions, gene assignment, exon context, the exon-overlap filter, Pangolin's gene database and
SpliceTransformer's strand lookup all read gencode.v50.mane.gtf.gz.

Writes annotation_v2/ (outside git) and results/V01_annotation.txt. V01b_spt_annotation_check.py
then verifies SpliceTransformer's database, strand and coverage (Task 1, Step 5).

Usage:  python V01_annotation.py
"""
import gzip, hashlib, os, re, subprocess, sys

P = "/mnt/Work0/Splice_altering_variant"
V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = f"{P}/docker_data/gencode.v50.annotation.gtf.gz"
ANNOT = f"{P}/annotation_v2"
MANE = f"{ANNOT}/gencode.v50.mane.gtf.gz"
TAGS = ('tag "MANE_Select"', 'tag "MANE_Plus_Clinical"')
GENE_ID = re.compile(r'gene_id "([^"]+)"')


def write_mane():
    """Gene, transcript and exon lines of MANE transcripts. Each gene line's start and end are reset
    to the extent of the gene's MANE transcripts: GENCODE's gene lines span every transcript of the
    gene, and SpliceTransformer finds genes by their span (sptransformer.py:68-72)."""
    extent = {}
    with gzip.open(SRC, "rt") as f:
        for line in f:
            x = line.split("\t", 8)
            if not line.startswith("#") and x[2] == "transcript" and any(t in x[8] for t in TAGS):
                g = GENE_ID.search(x[8]).group(1)
                s, e = extent.get(g, (int(x[3]), int(x[4])))
                extent[g] = (min(s, int(x[3])), max(e, int(x[4])))
    n = {"gene": 0, "transcript": 0, "exon": 0}
    changed = 0
    # mtime=0 so the output's md5 depends only on its content
    with gzip.open(SRC, "rt") as f, open(MANE, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as o:
        for line in f:
            if line.startswith("#"):
                o.write(line.encode())
                continue
            x = line.split("\t")
            if x[2] == "gene" and GENE_ID.search(x[8]).group(1) in extent:
                new = tuple(map(str, extent[GENE_ID.search(x[8]).group(1)]))
                changed += (x[3], x[4]) != new
                x[3], x[4] = new
            elif not (x[2] in ("transcript", "exon") and any(t in x[8] for t in TAGS)):
                continue
            o.write("\t".join(x).encode())
            n[x[2]] += 1
    return n, changed


def spt_run_dir():
    """SpliceTransformer reads data/data_package/hg38.* relative to its working directory; a run
    directory whose annotation is the MANE GTF makes its strand lookup use MANE genes.

    pyensembl (2.6.2) builds its database next to the GTF it is given and reuses any database it
    finds there without checking it against the GTF, so a stale one is removed here; the new one is
    built and verified by V01b_spt_annotation_check.py."""
    d = f"{ANNOT}/spt_run"
    os.makedirs(f"{d}/data/data_package", exist_ok=True)
    os.makedirs(f"{d}/pyensembl_cache", exist_ok=True)      # PYENSEMBL_CACHE_DIR for every run
    stale = f"{d}/data/data_package/hg38.annotation.gtf.db"
    if os.path.exists(stale):
        os.remove(stale)
    for dst, src in {f"{d}/data/data_package/hg38.fa": f"{P}/hg38/hg38.fa",
                     f"{d}/data/data_package/hg38.fa.fai": f"{P}/hg38/hg38.fa.fai",
                     f"{d}/data/data_package/hg38.annotation.gtf.gz": MANE,
                     f"{d}/model": f"{P}/SpliceTransformer/model"}.items():
        if not os.path.lexists(dst):
            os.symlink(src, dst)
    return d


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(ANNOT, exist_ok=True)
    out = open(f"{V2}/results/V01_annotation.txt", "w")
    def log(s=""):
        print(s); out.write(s + "\n")
    log("V01 -- GENCODE v50 MANE annotation for MetaSplice v2.0")
    log(f"source: {SRC} (md5 {md5(SRC)})")
    n, changed = write_mane()
    log(f"MANE GTF: {n['gene']:,} genes, {n['transcript']:,} transcripts, {n['exon']:,} exons -> {MANE}")
    log(f"  gene lines whose span was reset to the extent of their MANE transcripts: {changed:,}")
    log(f"  md5 {md5(MANE)}")
    subprocess.run([sys.executable, f"{P}/Pangolin/scripts/create_db.py", MANE, "--filter", "None"],
                   check=True, capture_output=True)
    db = MANE[:-len(".gtf.gz")] + ".db"
    log(f"Pangolin database: {db} ({os.path.getsize(db) / 1e6:,.1f} MB)")
    log(f"SpliceTransformer run directory: {spt_run_dir()}")
    out.close()


if __name__ == "__main__":
    main()
