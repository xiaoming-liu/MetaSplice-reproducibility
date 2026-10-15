"""V03 -- SpliceTransformer and Pangolin on GRCh38 for every v2.0 variant (docs/METASPLICE_V2_PLAN.md, Task 4).

Every run disables TF32 (NVIDIA_TF32_OVERRIDE=0). The first device check (2026-09-30) found that
SpliceTransformer on the GPU with TF32, PyTorch's default for convolutions on this GPU, differs
from the CPU by up to 0.101 in a tissue output (407 of 500 variants). Its tissue scores are zeroed
below a splice score of 0.09, so small precision differences near that threshold become large. With
TF32 off, GPU runs match the CPU within 7.4e-4, are identical from run to run, and are no slower.

Modes:
  device      500 variants of the union (seed 11): each tool twice on the GPU, once on the CPU, once
              on the GPU in 4 chunks run at the same time, and once on the GPU with TF32 allowed; the
              largest difference per output (Step 1).
  throughput  5,000 variants on the GPU, as 1 process and as 4 processes in parallel; variants per
              second for each tool (Step 2).
  run         the union of every variant set, deduplicated by chr:pos:ref:alt, in --chunks parallel
              processes per tool, both tools at the same time (Steps 3-4); then the coverage checks of
              Step 5: every variant has a SpliceTransformer row and a Pangolin row, and Pangolin's
              blank output (scored 0.0, as in training) is counted by set and region.
Outputs go under tools/v2 (outside git); each mode appends to results/V03_upstream_scores.txt.

SpliceTransformer runs from annotation_v2/spt_run with PYENSEMBL_CACHE_DIR set, and the md5 of the
pyensembl database V01b verified is checked before and after every run. Pangolin uses the MANE database
(annotation_v2/gencode.v50.mane.db) with pipeline/04's defaults.

Usage:  python V03_upstream_scores.py {device,throughput,run} [--chunks 4]
"""
import argparse, hashlib, importlib.util, os, subprocess, sys, time
import numpy as np
import pandas as pd

P = "/mnt/Work0/Splice_altering_variant"
V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = f"{P}/annotation_v2/spt_run"
DB = f"{RUN}/data/data_package/hg38.annotation.gtf.db"
PANGOLIN_DB = f"{P}/annotation_v2/gencode.v50.mane.db"
HG38 = f"{P}/hg38/hg38.fa"
T = f"{P}/tools/v2"
PY = sys.executable
PANGOLIN = os.path.join(os.path.dirname(PY), "pangolin")
SETS = ["training", "test_candidates", "vus", "literature", "splicevardb", "gnomad_reference", "gnomad_reference_stratified"]
spec = importlib.util.spec_from_file_location("score_v2", f"{P}/MetaSplice/score_v2.py")
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()


def union():
    parts = [pd.read_csv(f"{V2}/data/v2_variants_{s}.csv", usecols=["chr", "pos", "ref", "alt"], low_memory=False) for s in SETS]
    u = pd.concat(parts).drop_duplicates()
    order = {f"chr{c}": i for i, c in enumerate([str(k) for k in range(1, 23)] + ["X", "Y"])}
    return u.assign(o=u.chr.map(order)).sort_values(["o", "pos", "ref", "alt"]).drop(columns="o").reset_index(drop=True)


def write_inputs(d, stem):
    with open(f"{stem}.vcf", "w") as f:
        f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for c, p, r, a in d.itertuples(index=False):
            f.write(f"{c}\t{p}\t.\t{r}\t{a}\t.\tPASS\t.\n")
    d.assign(chr=d.chr.str.replace("chr", "")).rename(columns={"chr": "CHROM", "pos": "POS", "ref": "REF", "alt": "ALT"}).to_csv(
        f"{stem}.pangolin.csv", index=False)


def start(tool, stem, cpu=False, tf32=False):
    """Start one tool on stem.vcf / stem.pangolin.csv, TF32 off unless tf32; returns the process."""
    env = dict(os.environ, PYENSEMBL_CACHE_DIR=f"{RUN}/pyensembl_cache")
    env.pop("NVIDIA_TF32_OVERRIDE", None)
    if not tf32:
        env["NVIDIA_TF32_OVERRIDE"] = "0"
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    log = open(f"{stem}.{tool}.log", "w")
    if tool == "spt":
        cmd = [PY, f"{P}/SpliceTransformer/sptransformer.py", "--input", f"{stem}.vcf", "--output", f"{stem}.spt.csv",
               "--reference", "hg38", "--vcf", "True", "--raw_score", "True"]
        return subprocess.Popen(cmd, cwd=RUN, env=env, stdout=log, stderr=subprocess.STDOUT)
    cmd = [PANGOLIN, f"{stem}.pangolin.csv", HG38, PANGOLIN_DB, f"{stem}.pangolin_out", "-c", "CHROM,POS,REF,ALT"]
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def run_parallel(tool, d, stem, chunks, cpu=False, tf32=False):
    """Split d into chunks, run them at the same time, and return (seconds, stems in order)."""
    stems = []
    for k, part in enumerate(np.array_split(np.arange(len(d)), chunks)):
        s = f"{stem}.part{k}"
        write_inputs(d.iloc[part], s)
        stems.append(s)
    t0 = time.time()
    procs = [start(tool, s, cpu, tf32) for s in stems]
    for s, p in zip(stems, procs):
        if p.wait():
            raise RuntimeError(f"{tool} failed on {s}; see {s}.{tool}.log")
    return time.time() - t0, stems


def read(tool, stems):
    """{chr:pos:ref:alt: array of outputs}: SpliceTransformer score + 15 tissues; Pangolin (inc, dec)."""
    out = {}
    for s in stems:
        if tool == "spt":
            out.update(ms.load_spt(f"{s}.spt.csv"))
        else:
            out.update({k: np.array(v) for k, v in ms.load_pangolin(f"{s}.pangolin_out.csv").items()})
    return out


def max_diff(a, b):
    keys = sorted(set(a) & set(b))
    missing = len(set(a) ^ set(b))
    d = np.array([np.abs(np.asarray(a[k], float) - np.asarray(b[k], float)) for k in keys])
    return d.max(axis=0), int((d.max(axis=1) > 0).sum()), len(keys), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["device", "throughput", "run"])
    ap.add_argument("--chunks", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(T, exist_ok=True)
    out = open(f"{V2}/results/V03_upstream_scores.txt", "a")
    def log(s=""):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    db0 = md5(DB)
    u = union()
    log(f"\nV03 {args.mode} -- union of the v2.0 variant sets: {len(u):,} variants; SpliceTransformer database md5 {db0}")

    if args.mode == "device":
        d = u.sample(500, random_state=11).reset_index(drop=True)
        runs = {}
        for tool in ("spt", "pangolin"):
            for name, cpu, chunks, tf32 in [("gpu1", False, 1, False), ("gpu2", False, 1, False), ("cpu", True, 1, False),
                                            ("gpu4", False, 4, False), ("gpu_tf32", False, 1, True)]:
                sec, stems = run_parallel(tool, d, f"{T}/device.{tool}.{name}", chunks, cpu, tf32)
                runs[(tool, name)] = read(tool, stems)
                log(f"  {tool} {name}: {len(runs[(tool, name)]):,} variants in {sec:.0f} s")
            cols = ["score"] + ms.TISSUES if tool == "spt" else ["inc", "dec"]
            for a, b in [("gpu1", "gpu2"), ("gpu1", "cpu"), ("gpu1", "gpu4"), ("gpu_tf32", "cpu"), ("gpu1", "gpu_tf32")]:
                mx, n_diff, n, missing = max_diff(runs[(tool, a)], runs[(tool, b)])
                log(f"  {tool} {a} vs {b}: {n} variants, {n_diff} differ in any output, {missing} missing on one side; "
                    f"largest difference {mx.max():.3g} ({cols[int(mx.argmax())]})")
    elif args.mode == "throughput":
        d = u.sample(5000, random_state=12).reset_index(drop=True)
        for tool in ("spt", "pangolin"):
            for chunks in (1, args.chunks):
                sec, _ = run_parallel(tool, d, f"{T}/throughput.{tool}.x{chunks}", chunks)
                log(f"  {tool}, {chunks} process(es): {len(d):,} variants in {sec:.0f} s = {len(d) / sec:.1f} variants/s; "
                    f"union would take {len(u) / (len(d) / sec) / 3600:.1f} h")
    else:
        jobs = {}
        for tool in ("spt", "pangolin"):
            jobs[tool] = []
            for k, part in enumerate(np.array_split(np.arange(len(u)), args.chunks)):
                s = f"{T}/union.{tool}.part{k}"
                write_inputs(u.iloc[part], s)
                jobs[tool].append(s)
        t0 = time.time()
        procs = [(tool, s, start(tool, s)) for tool, stems in jobs.items() for s in stems]
        for tool, s, p in procs:
            if p.wait():
                raise RuntimeError(f"{tool} failed on {s}; see {s}.{tool}.log")
        log(f"  both tools, {args.chunks} processes each, at the same time: {(time.time() - t0) / 3600:.1f} h")
        keys = set(ms._key(*v) for v in u.itertuples(index=False))
        blank = {}
        for tool, stems in jobs.items():
            res = read(tool, stems)
            log(f"  {tool}: rows for {len(set(res) & keys):,} of {len(keys):,} union variants; missing {len(keys - set(res)):,}")
            if keys - set(res):
                raise RuntimeError(f"{tool} has no row for {len(keys - set(res)):,} union variants")
            pd.DataFrame({"part": stems}).to_csv(f"{T}/union.{tool}.parts.csv", index=False)
        for s in jobs["pangolin"]:
            with open(f"{s}.pangolin_out.csv") as fh:
                fh.readline()
                for line in fh:
                    x = line.rstrip("\n").split(",", 4)
                    blank[ms._key(*x[:4])] = len(x) < 5 or not x[4].strip()
        log("  Pangolin blank output (scored 0.0, as in training), by set and region:")
        for name in SETS:
            d = pd.read_csv(f"{V2}/data/v2_variants_{name}.csv", usecols=["chr", "pos", "ref", "alt", "region"], low_memory=False)
            b = pd.Series([blank[ms._key(*v)] for v in d[["chr", "pos", "ref", "alt"]].itertuples(index=False)], index=d.index)
            log(f"    {name:<28} {b.mean():6.2%} of {len(d):,}; by region " +
                ", ".join(f"{r} {b[d.region == r].mean():.2%}" for r in sorted(d.region.unique())))
    db1 = md5(DB)
    log(f"  SpliceTransformer database unchanged: {db0 == db1}")
    if db0 != db1:
        raise RuntimeError("the SpliceTransformer database changed during the run")
    out.close()


if __name__ == "__main__":
    main()
