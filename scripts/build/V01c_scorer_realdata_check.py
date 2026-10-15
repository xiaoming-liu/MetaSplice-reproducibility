"""V01c -- The v2.0 scorer on real data (docs/METASPLICE_V2_PLAN.md, Task 2).

The unit tests (MetaSplice/test_v2.py) use synthetic files. This runs score_v2.py on 60 v1
training variants from chr21 and chr22 that v1 did bridge to hg38, and checks:
  1. GRCh38 input: for the variants in scope under MANE (the scorer computes features for those
     only), the features that come from the same source files in v1 and v2 are identical
     to data/features.csv (v1): phyloP and phastCons (GRCh38 wigFix), GPN-MSA (GRCh38), DANN
     (GRCh37; v2 reaches it through the hg38->hg19 chain) and gnomAD AF (reported, not a feature).
     SpliceTransformer, Pangolin, exon context, region and gene-level values are expected to
     differ where the MANE annotation or dbNSFP 5.4 differs, and are only reported.
  2. hg19 input: the same variants at their GRCh37 coordinates lift to the GRCh38 coordinates of
     the ClinVar GRCh38 release, and give identical features.
Writes results/V01c_scorer_realdata_check.txt; exits 1 if a check fails.
"""
import os, subprocess, sys
import numpy as np
import pandas as pd

P = "/mnt/Work0/Splice_altering_variant"
V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W = f"{P}/tools/v2_scorer_check"
PY = sys.executable
D = f"{P}/docker_data"
ANNOT = f"{P}/annotation_v2"
SAME_SOURCE = ["phyloP100way_vertebrate", "phyloP470way_mammalian", "phastCons100way_vertebrate",
               "phastCons470way_mammalian", "gpn_msa_score", "DANN_score"]
failures = []


def check(log, name, ok, detail=""):
    log(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures.append(name)


def run(cmd, cwd=None, env=None):
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd[:2]} failed:\n{r.stderr[-3000:]}")
    return r.stdout


def main():
    os.makedirs(W, exist_ok=True)
    out = open(f"{V2}/results/V01c_scorer_realdata_check.txt", "w")
    def log(s=""):
        print(s); out.write(s + "\n")
    log("V01c -- scorer v2.0 on real data: 60 v1 training variants on chr21 and chr22")

    tr = pd.read_csv(f"{P}/reproducibility_manuscript/data/features.csv", low_memory=False,
                     usecols=["chr", "pos", "ref", "alt", "clinvar_id", "region", "gnomAD_AF"] + SAME_SOURCE)
    tr = tr[tr.chr.isin(["chr21", "chr22"]) & ~tr.region.str.startswith("Exonic", na=True) & tr.phyloP100way_vertebrate.notna()]
    h38 = pd.read_csv(f"{P}/clinvar/hg38/clinvar_20250831.vcf.gz.summary.gz", sep="\t", low_memory=False,
                      usecols=["#chr", "pos", "clinvar_id", "ref", "alt"], dtype={"#chr": str}).rename(
        columns={"#chr": "c38", "pos": "p38", "ref": "r38", "alt": "a38"})
    s = tr.merge(h38.drop_duplicates("clinvar_id"), on="clinvar_id").sample(60, random_state=5).sort_values(["c38", "p38"])
    s["c38"] = "chr" + s.c38
    log(f"sample: {len(s)} variants; regions {s.region.value_counts().to_dict()}")

    v38, v19 = f"{W}/in38.vcf", f"{W}/in19.vcf"
    for path, cols in [(v38, ["c38", "p38", "r38", "a38"]), (v19, ["chr", "pos", "ref", "alt"])]:
        with open(path, "w") as f:
            f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            for c, p, r, a in s[cols].itertuples(index=False):
                f.write(f"{c}\t{p}\t.\t{r}\t{a}\t.\tPASS\t.\n")
    env = dict(os.environ, PYENSEMBL_CACHE_DIR=f"{ANNOT}/spt_run/pyensembl_cache")
    run([PY, f"{P}/SpliceTransformer/sptransformer.py", "--input", v38, "--output", f"{W}/spt.csv",
         "--reference", "hg38", "--vcf", "True", "--raw_score", "True"], cwd=f"{ANNOT}/spt_run", env=env)
    with open(f"{W}/pangolin_in.csv", "w") as f:
        f.write("CHROM,POS,REF,ALT\n")
        for c, p, r, a in s[["c38", "p38", "r38", "a38"]].itertuples(index=False):
            f.write(f"{c.replace('chr', '')},{p},{r},{a}\n")
    run([f"{os.path.dirname(PY)}/pangolin", f"{W}/pangolin_in.csv", f"{P}/hg38/hg38.fa", f"{ANNOT}/gencode.v50.mane.db",
         f"{W}/pangolin", "-c", "CHROM,POS,REF,ALT"])
    common = [PY, f"{P}/MetaSplice/score_v2.py", "--fasta", f"{P}/hg38/hg38.fa", "--hg19-fasta", f"{P}/hg19/hg19.fa",
              "--gtf", f"{D}/gencode.v50.annotation.gtf.gz", "--reverse-liftover-dir", f"{P}/hg38tohg19_ensembl",
              "--dann-dir", f"{D}/DANN", "--phylop-dir", f"{D}/phyloP_GRCh38", "--phastcons-dir", f"{D}/phastCons_GRCh38",
              "--gnomad-dir", f"{D}/gnomAD", "--gpn-msa", f"{D}/gpn_msa/scores.tsv.bgz",
              "--dbnsfp-gene", f"{D}/dbNSFP5.4_gene.gz", "--spt-scores", f"{W}/spt.csv",
              "--pangolin-scores", f"{W}/pangolin.csv", "--include-out-of-scope", "--features-only"]
    run(common + ["--input", v38, "--output", f"{W}/out38.tsv", "--features-out", f"{W}/f38.tsv"])
    run(common + ["--input", v19, "--output", f"{W}/out19.tsv", "--features-out", f"{W}/f19.tsv",
                  "--genome", "hg19", "--liftover-dir", f"{P}/hg19tohg38_ensembl"])

    f38 = pd.read_csv(f"{W}/f38.tsv", sep="\t")
    f19 = pd.read_csv(f"{W}/f19.tsv", sep="\t")
    m = s.merge(f38, left_on=["c38", "p38", "r38", "a38"], right_on=["chr", "pos", "ref", "alt"], suffixes=("_v1", ""))
    log(f"\n1. GRCh38 input ({len(m)} variants; in scope under MANE: {int(m.in_scope.sum())}; "
        f"out of scope: {m.out_of_scope_reason[~m.in_scope].value_counts().to_dict()})")
    check(log, "out-of-scope rows carry no features", bool(m.loc[~m.in_scope, SAME_SOURCE].isna().all(axis=None)))
    m = m[m.in_scope]
    for c in SAME_SOURCE:
        a, b = m[f"{c}_v1"].astype(float), m[c].astype(float)
        # data/features.csv stores DANN rounded to 6 decimals; v2 keeps the DANN file's full value
        same = np.isclose(a, b, rtol=0, atol=5e-7 if c == "DANN_score" else 1e-9, equal_nan=True)
        check(log, f"{c} equals v1", bool(same.all()), f"{int((~same).sum())} differ; v2 missing {int(b.isna().sum())}")
    af = np.isclose(m.gnomAD_AF.astype(float), m.gnomad_af.astype(float), rtol=0, atol=1e-12)
    check(log, "gnomAD AF (reported) equals v1", bool(af.all()), f"{int((~af).sum())} differ")
    log("  reported only (annotation or source differs by design):")
    for c in ["spt_orig", "pangolin_max", "exon_length", "gnomAD_pLI", "gnomAD_LOEUF"]:
        log(f"    {c}: present {int(m[c].notna().sum())}/{len(m)}")

    log(f"\n2. hg19 input")
    k = f19.merge(s, left_on=["chr", "pos", "ref", "alt"], right_on=["chr", "pos", "ref", "alt"])
    same_pos = (k.chr38 == k.c38) & (k.pos38 == k.p38) & (k.ref38 == k.r38) & (k.alt38 == k.a38)
    check(log, "lifted to the ClinVar GRCh38 coordinates", bool(same_pos.all()) and len(k) == len(s),
          f"{int(same_pos.sum())}/{len(s)}; status {k.status.value_counts().to_dict()}")
    feats = [c for c in f38.columns if c not in ("chr", "pos", "ref", "alt", "status")]
    a = f38.set_index(["chr", "pos", "ref", "alt"])[feats].sort_index()
    b = f19.rename(columns={"chr": "c19", "pos": "p19", "ref": "r19", "alt": "a19"}).set_index(
        ["chr38", "pos38", "ref38", "alt38"])[feats].sort_index()
    b.index.names = a.index.names
    num = [c for c in feats if a[c].dtype.kind in "fi"]
    same = np.isclose(a[num].values.astype(float), b.loc[a.index, num].values.astype(float), equal_nan=True).all()
    # fill missing text first: with pandas' string dtype, astype(str) keeps NaN, and NaN != NaN
    txt = all((a[c].astype(str).fillna("") == b.loc[a.index, c].astype(str).fillna("")).all() for c in feats if c not in num)
    check(log, "features identical to the GRCh38-input run", bool(same and txt))
    log(f"\nV01c: {'ALL CHECKS PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    out.close()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
