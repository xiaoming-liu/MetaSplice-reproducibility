"""V04 -- The 38-feature table of every v2.0 variant, built by the v2.0 scorer (docs/METASPLICE_V2_PLAN.md, Task 5).

One feature code path: the union of the V02 variant sets goes through score_v2.py --features-only, the
same code production scoring runs, with every source of the 38 features and the SpliceTransformer and
Pangolin outputs of V03 (TF32 off). The union is split into chromosome groups scored in parallel
processes; a variant's features do not depend on the other variants of a run, so this equals one run.
Every union variant must come back in scope (V02 chose them with the same scope()).

Writes data/v2_features_union.tsv and results/V04_features.txt.
"""
import importlib.util, os, subprocess, sys, time
import pandas as pd

P = "/mnt/Work0/Splice_altering_variant"
V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T = f"{P}/tools/v2"
D = f"{P}/docker_data"
GROUPS = 8
spec = importlib.util.spec_from_file_location("v03", os.path.join(os.path.dirname(os.path.abspath(__file__)), "V03_upstream_scores.py"))
v03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v03)


def concat_parts(tool):
    """One SpliceTransformer or Pangolin output file for the union, from V03's parts."""
    stems = pd.read_csv(f"{T}/union.{tool}.parts.csv").part.tolist()
    path = f"{T}/union.{tool}.all.csv"
    if tool == "spt":
        pd.concat([pd.read_csv(f"{s}.spt.csv", index_col=0) for s in stems]).to_csv(path)
    else:
        with open(path, "w") as out:
            for k, s in enumerate(stems):
                with open(f"{s}.pangolin_out.csv") as f:
                    header = f.readline()
                    if k == 0:
                        out.write(header)
                    out.writelines(f)
    return path


def main():
    out = open(f"{V2}/results/V04_features.txt", "w")
    def log(s=""):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    u = v03.union()
    log(f"V04 -- features of the union of the v2.0 variant sets through score_v2.py: {len(u):,} variants")
    spt, pangolin = concat_parts("spt"), concat_parts("pangolin")
    # chromosome groups of about equal size, scored in parallel
    sizes = u.chr.value_counts()
    groups = [[] for _ in range(GROUPS)]
    load = [0] * GROUPS
    for c, n in sizes.items():
        k = load.index(min(load))
        groups[k].append(c)
        load[k] += n
    procs, t0 = [], time.time()
    for k, chroms in enumerate(groups):
        stem = f"{T}/features.g{k}"
        v03.write_inputs(u[u.chr.isin(chroms)], stem)
        cmd = [sys.executable, f"{P}/MetaSplice/score_v2.py", "--input", f"{stem}.vcf", "--output", f"{stem}.out.tsv",
               "--features-out", f"{stem}.features.tsv", "--features-only",
               "--fasta", f"{P}/hg38/hg38.fa", "--hg19-fasta", f"{P}/hg19/hg19.fa", "--gtf", f"{D}/gencode.v50.annotation.gtf.gz",
               "--reverse-liftover-dir", f"{P}/hg38tohg19_ensembl", "--dann-dir", f"{D}/DANN",
               "--phylop-dir", f"{D}/phyloP_GRCh38", "--phastcons-dir", f"{D}/phastCons_GRCh38",
               "--gnomad-dir", f"{D}/gnomAD", "--gpn-msa", f"{D}/gpn_msa/scores.tsv.bgz",
               "--dbnsfp-gene", f"{D}/dbNSFP5.4_gene.gz", "--spt-scores", spt, "--pangolin-scores", pangolin]
        procs.append((k, chroms, subprocess.Popen(cmd, stdout=open(f"{stem}.log", "w"), stderr=subprocess.STDOUT)))
        log(f"  group {k}: {load[k]:,} variants on {', '.join(sorted(chroms))}")
    for k, chroms, p in procs:
        if p.wait():
            raise RuntimeError(f"score_v2.py failed for group {k}; see {T}/features.g{k}.log")
    log(f"  {GROUPS} scorer processes: {(time.time() - t0) / 3600:.1f} h")
    f = pd.concat([pd.read_csv(f"{T}/features.g{k}.features.tsv", sep="\t", low_memory=False) for k in range(GROUPS)],
                  ignore_index=True)
    key = lambda d: d.chr.astype(str) + ":" + d.pos.astype(str) + ":" + d.ref + ":" + d.alt
    missing = set(key(u)) - set(key(f))
    log(f"  in scope: {len(f):,} of {len(u):,}; not returned {len(missing):,}")
    if missing or len(f) != len(u):
        raise RuntimeError(f"{len(missing):,} union variants did not come back in scope")
    f.to_csv(f"{V2}/data/v2_features_union.tsv", sep="\t", index=False)
    log(f"  wrote data/v2_features_union.tsv ({len(f):,} rows, {f.shape[1]} columns)")
    out.close()


if __name__ == "__main__":
    main()
