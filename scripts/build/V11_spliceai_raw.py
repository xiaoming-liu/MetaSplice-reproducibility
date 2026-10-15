"""V11 -- SpliceAI raw scores in the configuration ClinGen calibrated (protocol Amendment 7).

Walker et al. (2023) calibrated the SpliceAI cut-offs 0.2 and 0.1 on the maximum raw delta score of SpliceAI v1.3.1 over
a 10,000-nt window (+-4,999 nt). The precomputed files V08 reads are masked, with a +-50-nt window. This script scores
X3's four cohorts with SpliceAI 1.3.1 itself (-A grch37 -D 4999 -M 0), using V08's GRCh37 coordinates and hg19 REF check
and V08's value rule (the maximum of DS_AG, DS_AL, DS_DG and DS_DL in the first SpliceAI record for the allele; the
maximum over all records is kept beside it).

Before the raw run, an installation check scores a seeded sample (seed 42; up to 100 variants per non-canonical region,
among variants with a precomputed masked score) in the precomputed configuration (-D 50 -M 1). At least 99% must lie
within 0.01 of V08's value, or the script stops before any raw score is computed. The variant list comes from X3's
cohort loader, which loads labels with the cohorts; labels are not used to select SpliceAI settings or to calculate its
predictions.

Writes data/v2_spliceai_raw_d4999.csv and results/V11_spliceai_raw.txt.
"""
import argparse, hashlib, importlib.util, os, subprocess, sys, tempfile, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from common import NONCANONICAL

ENV = "/home/xliu/miniconda3/envs/spliceai131"
PKG = f"{ENV}/lib/python3.10/site-packages/spliceai"
SPLICEAI_MD5 = {   # Amendment 7
    "annotations/grch37.txt": "cbe7341aaab1dd6678dc9504b0952165",
    "models/spliceai1.h5": "140fc5c24a418d1cac65a5a8d4c543cf",
    "models/spliceai2.h5": "9d5a30fac2a3f11751ef530f44fe2eb2",
    "models/spliceai3.h5": "e44a2ecdf62ec9dc226ebcfed83b3b2f",
    "models/spliceai4.h5": "1449760f82aa8f08bcc4761280811a2c",
    "models/spliceai5.h5": "d35a0e714be8b359c176964e94091b2f",
}
KEY = ["chr", "pos", "ref", "alt"]
K19 = ["chr19", "pos19", "ref19", "alt19"]
TMP = "/mnt/Work0/Splice_altering_variant/tools/v2/spliceai_raw"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def write_vcf(d, path, fai):
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        for c, n in fai:
            f.write(f"##contig=<ID={c},length={n}>\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for r in d.itertuples(index=False):
            f.write(f"{r.chr19}\t{int(r.pos19)}\t.\t{r.ref19}\t{r.alt19}\t.\t.\t.\n")


def parse(path):
    """(chr19, pos19, ref19, alt19) -> (first-record max DS, all-records max DS, n records); unscored alleles absent."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            x = line.rstrip("\n").split("\t")
            if "SpliceAI=" not in x[7]:
                continue
            recs = [r.split("|") for r in x[7].split("SpliceAI=")[1].split(";")[0].split(",")]
            vals = []
            for r in recs:
                if r[0] != x[4]:
                    continue
                try:
                    vals.append(max(float(r[2]), float(r[3]), float(r[4]), float(r[5])))
                except ValueError:          # '.' fields: the gene was skipped
                    pass
            if vals:
                out[(x[0], int(x[1]), x[3], x[4])] = (vals[0], max(vals), len(vals))
    return out


def run_spliceai(d, tag, dist, mask, chunks, fai, log):
    os.makedirs(TMP, exist_ok=True)
    parts = np.array_split(np.arange(len(d)), chunks)
    procs, outs = [], []
    env = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true", TF_CPP_MIN_LOG_LEVEL="2")
    t0 = time.time()
    for k, idx in enumerate(parts):
        vin, vout = f"{TMP}/{tag}.part{k}.vcf", f"{TMP}/{tag}.part{k}.out.vcf"
        write_vcf(d.iloc[idx].sort_values(["chr19", "pos19"]), vin, fai)
        cmd = [f"{ENV}/bin/spliceai", "-I", vin, "-O", vout, "-R", HG19, "-A", "grch37", "-D", str(dist), "-M", str(mask)]
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(f"{TMP}/{tag}.part{k}.log", "w"), stderr=subprocess.STDOUT))
        outs.append(vout)
    for k, p in enumerate(procs):
        if p.wait():
            raise RuntimeError(f"SpliceAI failed on {tag} part {k}; see {TMP}/{tag}.part{k}.log")
    res = {}
    for o in outs:
        res.update(parse(o))
    skipped = {}
    for k in range(chunks):
        for line in open(f"{TMP}/{tag}.part{k}.log"):
            if "Skipping record" in line:
                why = line.split("Skipping record (")[1].split(")")[0]
                skipped[why] = skipped.get(why, 0) + 1
    log(f"  {tag}: {len(d):,} variants in {chunks} processes, {(time.time() - t0) / 60:.1f} min; scored {len(res):,}; "
        f"skipped by SpliceAI {skipped or 0}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--check-only", action="store_true", help="installation check only; no raw score")
    args = ap.parse_args()
    out = open(f"{V2}/results/V11_spliceai_raw.txt", "w") if not args.check_only else open(os.devnull, "w")
    def log(s=""):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    log("V11 -- SpliceAI 1.3.1 raw scores, +-4,999 nt (-A grch37 -D 4999 -M 0), for X3's cohorts (Amendment 7)")
    ver = subprocess.run([f"{ENV}/bin/python", "-c", "import spliceai, tensorflow, keras; "
                          "print(spliceai.__version__, tensorflow.__version__, keras.__version__)"],
                         capture_output=True, text=True, env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")).stdout.split()
    log(f"  SpliceAI {ver[0]}, TensorFlow {ver[1]}, Keras {ver[2]}")
    if ver[0] != "1.3.1":
        raise SystemExit("SpliceAI is not 1.3.1")
    bad = []
    for f, want in SPLICEAI_MD5.items():
        got = md5(f"{PKG}/{f}")
        log(f"  {got}  spliceai/{f}  {'ok' if got == want else 'DIFFERS from ' + want}")
        bad += [f] if got != want else []
    if bad:
        raise SystemExit(f"SpliceAI files differ from Amendment 7: {bad}")
    log(f"  {md5(HG19)}  {HG19}")
    fai = [(l.split("\t")[0], int(l.split("\t")[1])) for l in open(HG19 + ".fai")]

    x3 = load("x3", os.path.join(HERE, "..", "analysis", "X3_supporting_rule.py"))
    coh = x3.cohorts()
    u = pd.concat([d[KEY + K19 + ["region", "SpliceAI"]] for d in coh.values()]).drop_duplicates(KEY).reset_index(drop=True)
    log(f"\nX3's four cohorts: {len(u):,} distinct variants; GRCh37 record {int(u.pos19.notna().sum()):,}")
    ok = v08.hg19_ref_ok(u.assign(pos19=u.pos19.astype("Int64")))
    q = u[ok].copy()
    q["pos19"] = q.pos19.astype(int)
    log(f"  hg19 REF equal: {len(q):,}; precomputed masked score present: {int(q.SpliceAI.notna().sum()):,}")

    # installation check: the precomputed configuration on a seeded sample
    pool = q[q.SpliceAI.notna() & q.region.isin(NONCANONICAL)]
    samp = pd.concat([g.sample(min(100, len(g)), random_state=42) for _, g in pool.groupby("region")])
    log(f"\nInstallation check: -D 50 -M 1 on {len(samp):,} variants (seed 42; up to 100 per non-canonical region)")
    res = run_spliceai(samp, "check_d50_m1", 50, 1, min(args.chunks, 2), fai, log)
    got = samp.apply(lambda r: res.get((r.chr19, r.pos19, r.ref19, r.alt19), (np.nan,))[0], axis=1)
    diff = (got - samp.SpliceAI).abs()
    within = float((diff <= 0.01 + 1e-9).mean())
    log(f"  within 0.01 of V08's precomputed value: {within:.2%} (n={len(samp)}); unscored {int(got.isna().sum())}; "
        f"max |diff| {np.nanmax(diff.values):.3f}; identical {float((diff < 1e-9).mean()):.2%}")
    if within < 0.99:
        raise SystemExit("installation check failed (< 99% within 0.01); no raw score computed")
    log("  PASS")
    if args.check_only:
        return

    log(f"\nRaw scores: -D 4999 -M 0 on {len(q):,} variants")
    res = run_spliceai(q, "raw_d4999_m0", 4999, 0, args.chunks, fai, log)
    vals = q.apply(lambda r: res.get((r.chr19, r.pos19, r.ref19, r.alt19), (np.nan, np.nan, 0)), axis=1, result_type="expand")
    q["spliceai_raw_max"], q["spliceai_raw_max_allgenes"], q["spliceai_raw_records"] = vals[0], vals[1], vals[2].astype(int)
    m = u[KEY + K19].merge(q[KEY + ["spliceai_raw_max", "spliceai_raw_max_allgenes", "spliceai_raw_records"]], on=KEY,
                           how="left", validate="one_to_one")
    m.to_csv(f"{V2}/data/v2_spliceai_raw_d4999.csv", index=False)

    log("\nCoverage by cohort (scored by the raw run; by the precomputed masked lookup)")
    for name, d in coh.items():
        j = d[KEY].merge(m[KEY + ["spliceai_raw_max"]], on=KEY, how="left")
        log(f"  {name:<28} n={len(d):>7,}  raw {int(j.spliceai_raw_max.notna().sum()):>7,}  masked {int(d.SpliceAI.notna().sum()):>7,}")
    both = q.SpliceAI.notna() & q.spliceai_raw_max.notna()
    log(f"\nVariants with both scores: {int(both.sum()):,}; raw >= masked for {float((q.spliceai_raw_max[both] >= q.SpliceAI[both]).mean()):.2%}; "
        f"raw differs from masked by > 0.05 for {float(((q.spliceai_raw_max - q.SpliceAI)[both].abs() > 0.05).mean()):.2%}")
    log(f"wrote data/v2_spliceai_raw_d4999.csv ({len(m):,} rows; md5 {md5(f'{V2}/data/v2_spliceai_raw_d4999.csv')}); labels not used for SpliceAI settings or predictions")
    out.close()


v08 = load("v08", os.path.join(HERE, "V08_comparators.py"))
HG19 = v08.HG19

if __name__ == "__main__":
    main()
