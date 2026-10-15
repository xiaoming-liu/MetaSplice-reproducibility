"""V09c -- Features and comparator scores of the fresh temporal cohort (docs/METASPLICE_V2_PLAN.md, Task 11 Step 8).

The same code paths as every v2.0 table: SpliceTransformer and Pangolin through V03's launcher (TF32 off), then
score_v2.py --features-only in chromosome groups (as V04); SpliceAI and CADD with V08's functions (SpliceAI through
the GRCh37 record's coordinates with the hg19 REF check; CADD v1.7 on GRCh38). Checks: every variant comes back in
scope with V09b's region, gene and gene ID; gene-level values equal dbNSFP 5.4 (common.check_gene_level_values);
missingness parity with training (common.check_missingness_parity). No model score is computed and no label is used.

Writes data/v2_features_fresh_20260928.csv and results/V09c_fresh_features.txt.
"""
import importlib.util, os, subprocess, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from common import FEATURES, check_gene_level_values, check_missingness_parity

P = "/mnt/Work0/Splice_altering_variant"
V2 = os.path.dirname(os.path.dirname(HERE))
T = f"{P}/tools/v2/fresh"
D = f"{P}/docker_data"
KEY = ["chr", "pos", "ref", "alt"]
K19 = ["chr19", "pos19", "ref19", "alt19"]
GROUPS = 8
ORDER = [str(k) for k in range(1, 23)] + ["X", "Y"]
DBNSFP = f"{P}/reproducibility/raw_inputs/dbNSFP5.4_gene.gz"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m              # so worker processes can unpickle its functions
    spec.loader.exec_module(m)
    return m


v03 = load("v03", os.path.join(HERE, "V03_upstream_scores.py"))
v08 = load("v08", os.path.join(HERE, "V08_comparators.py"))


def keys(d):
    return set(map(tuple, d[KEY].astype(str).values))


def main():
    reuse = "--reuse-upstream" in sys.argv
    os.makedirs(T, exist_ok=True)
    out = open(f"{V2}/results/V09c_fresh_features.txt", "w")
    def log(s=""):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    v = pd.read_csv(f"{V2}/data/v2_variants_fresh_20260928.csv", low_memory=False)
    u = v[KEY].drop_duplicates()
    u = u.assign(o=u.chr.str.replace("chr", "", regex=False).map({k: i for i, k in enumerate(ORDER)})).sort_values(
        ["o", "pos", "ref", "alt"]).drop(columns="o").reset_index(drop=True)
    log(f"V09c -- features of the fresh cohort: {len(u):,} variants")

    paths = {}
    if reuse:
        # reuse this cohort's SpliceTransformer, Pangolin and scorer outputs from an earlier run of this script, only
        # after checking that each covers exactly the cohort's variants, once each
        paths = {"spt": f"{T}/spt.all.csv", "pangolin": f"{T}/pangolin.all.csv"}
        spt = pd.read_csv(paths["spt"], index_col=0).rename(columns={"#CHROM": "chr", "POS": "pos", "REF": "ref", "ALT": "alt"})
        if keys(spt) != keys(u) or spt.duplicated(KEY).any():
            raise RuntimeError("the earlier SpliceTransformer output does not cover exactly the cohort")
        with open(paths["pangolin"]) as fh:
            fh.readline()                      # the annotation field can hold commas: read the first four fields only
            pg = pd.DataFrame([ln.rstrip("\n").split(",", 4)[:4] for ln in fh], columns=KEY)
        pg["chr"] = "chr" + pg.chr.astype(str).str.replace("chr", "", regex=False)
        if keys(pg) != keys(u) or pg.duplicated(KEY).any():
            raise RuntimeError("the earlier Pangolin output does not cover exactly the cohort")
        log(f"  reused SpliceTransformer ({len(spt):,} rows, exactly the cohort) and Pangolin ({len(pg):,} rows, exactly the cohort) from the earlier run")
    for tool in ("spt", "pangolin") if not reuse else ():
        sec, stems = v03.run_parallel(tool, u, f"{T}/{tool}", 4)
        paths[tool] = f"{T}/{tool}.all.csv"
        if tool == "spt":
            pd.concat([pd.read_csv(f"{s}.spt.csv", index_col=0) for s in stems]).to_csv(paths[tool])
        else:
            with open(paths[tool], "w") as fo:
                for k, s in enumerate(stems):
                    with open(f"{s}.pangolin_out.csv") as f:
                        h = f.readline()
                        if k == 0:
                            fo.write(h)
                        fo.writelines(f)
        log(f"  {tool}: {len(u):,} variants in {sec:.0f} s (TF32 off)")

    groups, loads = [[] for _ in range(GROUPS)], [0] * GROUPS
    for c, n in u.chr.value_counts().items():
        k = loads.index(min(loads)); groups[k].append(c); loads[k] += n
    procs, t0 = [], time.time()
    for k, chroms in enumerate(groups):
        if not chroms:
            continue
        stem = f"{T}/features.g{k}"
        if reuse:
            procs.append((k, None))
            continue
        for f in (f"{stem}.features.tsv", f"{stem}.out.tsv"):
            if os.path.exists(f):
                os.remove(f)
        v03.write_inputs(u[u.chr.isin(chroms)], stem)
        cmd = [sys.executable, f"{P}/MetaSplice/score_v2.py", "--input", f"{stem}.vcf", "--output", f"{stem}.out.tsv",
               "--features-out", f"{stem}.features.tsv", "--features-only",
               "--fasta", f"{P}/hg38/hg38.fa", "--hg19-fasta", f"{P}/hg19/hg19.fa", "--gtf", f"{D}/gencode.v50.annotation.gtf.gz",
               "--reverse-liftover-dir", f"{P}/hg38tohg19_ensembl", "--dann-dir", f"{D}/DANN",
               "--phylop-dir", f"{D}/phyloP_GRCh38", "--phastcons-dir", f"{D}/phastCons_GRCh38",
               "--gnomad-dir", f"{D}/gnomAD", "--gpn-msa", f"{D}/gpn_msa/scores.tsv.bgz",
               "--dbnsfp-gene", f"{D}/dbNSFP5.4_gene.gz", "--spt-scores", paths["spt"], "--pangolin-scores", paths["pangolin"]]
        procs.append((k, subprocess.Popen(cmd, stdout=open(f"{stem}.log", "w"), stderr=subprocess.STDOUT)))
    for k, p in procs:
        if p is not None and p.wait():
            raise RuntimeError(f"score_v2.py failed for group {k}; see {T}/features.g{k}.log")
    f = pd.concat([pd.read_csv(f"{T}/features.g{k}.features.tsv", sep="\t", low_memory=False) for k, _ in procs], ignore_index=True)
    log(f"  scorer (features only): {len(f):,} rows" + (" (reused from the earlier run)" if reuse else f" in {(time.time() - t0) / 60:.0f} min"))
    got, want = set(map(tuple, f[KEY].astype(str).values)), set(map(tuple, u.astype(str).values))
    if got != want or f.duplicated(KEY).any():
        raise RuntimeError(f"scorer output differs from the cohort: missing {len(want - got)}, unexpected {len(got - want)}")
    m = v.merge(f.drop(columns=["status"], errors="ignore"), on=KEY, how="left", suffixes=("", "_scorer"), validate="one_to_one")
    for c in ["region", "gene", "gene_id"]:
        diff = int((m[c].astype(str) != m[f"{c}_scorer"].astype(str)).sum())
        if diff:
            raise ValueError(f"{c} differs between V09b and the scorer for {diff} variants")
    m = m.drop(columns=["region_scorer", "gene_scorer", "gene_id_scorer"])
    log("  every variant came back in scope once, with V09b's region, gene and gene ID")

    q = m[m.pos19.notna()].copy()
    q["pos19"] = q.pos19.astype(int)
    fa_ok = v08.hg19_ref_ok(q.assign(pos19=q.pos19.astype("Int64")))
    q = q[fa_ok]
    q["c"] = q.chr19.str.replace("chr", "", regex=False)
    tmp = tempfile.mkdtemp(prefix="v09c_", dir=os.environ.get("TMPDIR", "/tmp"))
    chroms = [c for c in ORDER if (q.c == c).any()]
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(lambda c: v08.spliceai_chrom(c, q.loc[q.c == c, ["pos19", "ref19", "alt19"]].astype(str), tmp), chroms))
    sai = pd.concat([r for _, r, _ in res], ignore_index=True)
    g = m.assign(c=m.chr.str.replace("chr", "", regex=False))
    chroms = [c for c in ORDER if (g.c == c).any()]
    with ProcessPoolExecutor(max_workers=8) as ex:
        cres = list(ex.map(v08.cadd_chrom, chroms, [list(g.loc[g.c == c, ["pos", "ref", "alt"]].itertuples(index=False, name=None)) for c in chroms]))
    cadd = pd.concat([pd.DataFrame(r, columns=["pos", "ref", "alt", "cadd_raw", "cadd_phred"]).assign(chr="chr" + c) for c, r, _ in cres],
                     ignore_index=True)
    m["pos19"] = m.pos19.astype("Int64")
    m = m.merge(sai[K19 + ["spliceai_max"]], on=K19, how="left", validate="many_to_one").merge(
        cadd[KEY + ["cadd_phred"]], on=KEY, how="left", validate="one_to_one")
    m.loc[~m.set_index(KEY).index.isin(q.set_index(KEY).index), "spliceai_max"] = np.nan
    log(f"  SpliceAI scored {m.spliceai_max.notna().mean():.2%} (GRCh37 record {m.pos19.notna().mean():.2%}; hg19 REF equal for "
        f"{len(q):,}); CADD scored {m.cadd_phred.notna().mean():.2%}")

    tr = pd.read_csv(f"{V2}/data/v2_features_training.csv", low_memory=False)
    log("\nGene-level values against dbNSFP 5.4:")
    check_gene_level_values({"fresh": m}, DBNSFP, log=log)
    log("\nMissingness parity with training (5 points; groups >= 100):")
    check_missingness_parity({"training": tr, "fresh": m}, log=log)
    log("  PASS")
    m.to_csv(f"{V2}/data/v2_features_fresh_20260928.csv", index=False)
    log(f"\nwrote data/v2_features_fresh_20260928.csv ({len(m):,} rows); no model score computed, no label used")
    out.close()


if __name__ == "__main__":
    main()
