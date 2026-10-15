"""Rebuild the comparator scores of this deposit from your own SpliceAI and CADD files.

This deposit does not redistribute SpliceAI or CADD scores. This script rebuilds them, with the build scripts' own
functions, from copies you obtain under their terms:

  - SpliceAI v1.3 precomputed masked SNV scores, GRCh37 (Illumina: spliceai_scores.masked.snv.hg19.vcf.gz), split by
    chromosome; --split-spliceai does the split;
  - CADD v1.7 GRCh38 whole_genome_SNVs.tsv.gz, with its tabix index;
  - the MetaSplice Release 2.0 data bundle (its GRCh37 FASTA and chain tables), for the gnomAD reference lookup.

  python lookup_comparators.py --split-spliceai spliceai_scores.masked.snv.hg19.vcf.gz --out SPLICEAI_DIR
  python lookup_comparators.py --spliceai SPLICEAI_DIR --cadd whole_genome_SNVs.tsv.gz --bundle metasplice_v2_data

Steps:
  1. data/v2_comparators.csv: V08's lookup (scripts/build/V08_comparators.py) for the training, test-candidate, VUS,
     literature and SpliceVarDB variants; V08's comparison with the v1 deposit is not repeated.
  2. The fresh cohort's spliceai_max and cadd_phred, as V09c computed them, restored into
     data/v2_features_fresh_20260928.csv.
  3. The pooled test set's spliceai_max and cadd_phred, as V12 took them (April variants from step 1, the others from
     step 2), restored into data/v2_features_pooled_test.csv.
  4. The four comparator columns of results/S12_mechanism.tsv, restored from step 1.
  5. data/v2_gnomad_spliceai.csv: V08b, run unchanged (it also rewrites results/V08b_gnomad_spliceai.txt).
Columns are restored at the text level into their original positions (stripped_columns.json). Every rebuilt file is
checked against originals.md5; the script exits 1 if any differs. The raw SpliceAI files (data/v2_spliceai_raw_d4999*.csv)
are rebuilt by run_spliceai_raw.py.

Needs the deposit's Python environment (requirements.txt) plus pysam, and zcat and awk.
"""
import argparse, csv, gzip, hashlib, importlib.util, io, json, multiprocessing, os, shutil, sys, tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BUILD = os.path.join(ROOT, "scripts", "build")
KEY = ["chr", "pos", "ref", "alt"]
K19 = ["chr19", "pos19", "ref19", "alt19"]
PATTERN = "spliceai_scores.masked.snv.hg19.vcf.gz.chr{}.gz"
ORDER = [str(k) for k in range(1, 23)] + ["X", "Y"]
FORK = multiprocessing.get_context("fork")         # workers find the build modules registered in sys.modules


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()


def originals():
    out = {}
    for line in open(os.path.join(HERE, "originals.md5")):
        m, rel = line.split(None, 1)
        out[rel.strip()] = m
    return out


def check(rel, want):
    got = md5(os.path.join(ROOT, rel))
    print(f"  {'ok     ' if got == want else 'DIFFERS'} {rel}", flush=True)
    return got == want


def load(name, path, source_edit=None):
    """Load a build script as a module, registered in sys.modules so worker processes can find its functions."""
    if source_edit:
        src = open(path).read()
        old, new = source_edit
        if src.count(old) != 1:
            raise RuntimeError(f"{path}: expected one '{old}'")
        m = type(sys)(name)
        m.__file__ = path
        sys.modules[name] = m
        exec(compile(src.replace(old, new), path, "exec"), m.__dict__)
        return m
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def split_spliceai(src, out):
    """Split Illumina's single SpliceAI VCF into the per-chromosome files V08 reads (header and records in file order)."""
    os.makedirs(out, exist_ok=True)
    header, handles = [], {}
    with gzip.open(src, "rt") as f:
        for line in f:
            if line[0] == "#":
                header.append(line)
                continue
            c = line.split("\t", 1)[0].replace("chr", "")
            if c not in handles:
                handles[c] = gzip.open(os.path.join(out, PATTERN.format(c)), "wt", compresslevel=4)
                handles[c].writelines(header)
            handles[c].write(line)
    for h in handles.values():
        h.close()
    print(f"wrote {len(handles)} chromosome files to {out}")


def as_text(df, cols):
    """{(chr, pos, ref, alt): [text of cols]} as pandas writes the values to CSV."""
    buf = io.StringIO()
    df[KEY + cols].to_csv(buf, index=False)
    buf.seek(0)
    t = pd.read_csv(buf, dtype=str, keep_default_na=False)
    return {tuple(k): list(v) for k, v in zip(t[KEY].itertuples(index=False, name=None), t[cols].itertuples(index=False, name=None))}


def restore(rel, positions, values, key_cols=KEY):
    """Insert the removed columns into the stripped file rel, values from values[key] (or a callable of the row)."""
    path = os.path.join(ROOT, rel)
    delim = "\t" if rel.endswith(".tsv") else ","
    with open(path, newline="") as f:
        rows = list(csv.reader(f, delimiter=delim))
    head = rows[0]
    kidx = [head.index(c) for c in key_cols]
    out = []
    for n, r in enumerate(rows):
        r = list(r)
        vals = [name for _, name in positions] if n == 0 else values(r, head) if callable(values) else \
            values.get(tuple(r[i] for i in kidx), [""] * len(positions))
        for (i, _), v in zip(positions, vals):
            r.insert(i, v)
        out.append(r)
    buf = io.StringIO(newline="")
    csv.writer(buf, delimiter=delim, lineterminator="\n").writerows(out)
    with open(path, "w", newline="") as f:
        f.write(buf.getvalue())


def step1_comparators(v08, workers, tmp):
    """V08's main without its v1 comparisons: one row per variant of the five sets."""
    a, u = v08.load_sets()
    u["hg19_ref_ok"] = v08.hg19_ref_ok(u)
    q = u[u.hg19_ref_ok].copy()
    q["c"] = q.chr19.str.replace("chr", "", regex=False)
    chroms = [c for c in v08.ORDER if (q.c == c).any()]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda c: v08.spliceai_chrom(c, q.loc[q.c == c, ["pos19", "ref19", "alt19"]].astype(str), tmp), chroms))
    sai = pd.concat([r for _, r, _ in res], ignore_index=True)
    print(f"  SpliceAI: {len(sai):,} of {len(q):,} GRCh37 alleles found; unreadable records {sum(n for _, _, n in res)}", flush=True)
    g = u.assign(c=u.chr.str.replace("chr", "", regex=False))
    chroms = [c for c in v08.ORDER if (g.c == c).any()]
    with ProcessPoolExecutor(max_workers=workers, mp_context=FORK) as ex:
        res = list(ex.map(v08.cadd_chrom, chroms, [list(g.loc[g.c == c, ["pos", "ref", "alt"]].itertuples(index=False, name=None)) for c in chroms]))
    cadd = pd.concat([pd.DataFrame(r, columns=["pos", "ref", "alt", "cadd_raw", "cadd_phred"]).assign(chr="chr" + c)
                      for c, r, _ in res], ignore_index=True)
    print(f"  CADD: {len(cadd):,} of {len(u):,} GRCh38 alleles found", flush=True)
    m = (u.merge(sai, on=K19, how="left", validate="many_to_one")
          .merge(cadd, on=KEY, how="left", validate="one_to_one"))
    assert len(m) == len(u)
    m.loc[~m.hg19_ref_ok, ["spliceai_max", "spliceai_max_allgenes", "spliceai_records"]] = np.nan
    m["o"] = m.chr.str.replace("chr", "", regex=False).map({k: i for i, k in enumerate(v08.ORDER)})
    m = m.sort_values(["o", "pos", "ref", "alt"]).drop(columns="o")
    m.to_csv(os.path.join(ROOT, "data", "v2_comparators.csv"), index=False)


def step2_fresh(v08, workers, tmp):
    """V09c's comparator lookup for the fresh cohort: {key: [spliceai_max, cadd_phred]} as text."""
    m = pd.read_csv(os.path.join(ROOT, "data", "v2_variants_fresh_20260928.csv"), low_memory=False)
    q = m[m.pos19.notna()].copy()
    q["pos19"] = q.pos19.astype(int)
    fa_ok = v08.hg19_ref_ok(q.assign(pos19=q.pos19.astype("Int64")))
    q = q[fa_ok]
    q["c"] = q.chr19.str.replace("chr", "", regex=False)
    chroms = [c for c in ORDER if (q.c == c).any()]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda c: v08.spliceai_chrom(c, q.loc[q.c == c, ["pos19", "ref19", "alt19"]].astype(str), tmp), chroms))
    sai = pd.concat([r for _, r, _ in res], ignore_index=True)
    g = m.assign(c=m.chr.str.replace("chr", "", regex=False))
    chroms = [c for c in ORDER if (g.c == c).any()]
    with ProcessPoolExecutor(max_workers=workers, mp_context=FORK) as ex:
        cres = list(ex.map(v08.cadd_chrom, chroms, [list(g.loc[g.c == c, ["pos", "ref", "alt"]].itertuples(index=False, name=None)) for c in chroms]))
    cadd = pd.concat([pd.DataFrame(r, columns=["pos", "ref", "alt", "cadd_raw", "cadd_phred"]).assign(chr="chr" + c) for c, r, _ in cres],
                     ignore_index=True)
    m["pos19"] = m.pos19.astype("Int64")
    m = m.merge(sai[K19 + ["spliceai_max"]], on=K19, how="left", validate="many_to_one").merge(
        cadd[KEY + ["cadd_phred"]], on=KEY, how="left", validate="one_to_one")
    m.loc[~m.set_index(KEY).index.isin(q.set_index(KEY).index), "spliceai_max"] = np.nan
    print(f"  fresh cohort: SpliceAI {m.spliceai_max.notna().sum():,}, CADD {m.cadd_phred.notna().sum():,} of {len(m):,}", flush=True)
    return as_text(m, ["spliceai_max", "cadd_phred"])


def shim_root(spliceai_dir, cadd, bundle, tmp):
    """A folder laid out like the original working tree, for V08b's unchanged code (its paths derive from one root)."""
    r = os.path.join(tmp, "root")
    os.makedirs(os.path.join(r, "MetaSplice"))
    os.makedirs(os.path.join(r, "hg19"))
    os.makedirs(os.path.join(r, "CADDv1.7", "hg38"))
    os.symlink(os.path.join(ROOT, "scripts", "score_v2.py"), os.path.join(r, "MetaSplice", "score_v2.py"))
    os.symlink(os.path.abspath(spliceai_dir), os.path.join(r, "spliceAI_v1.3"))
    for ext in ("", ".tbi", ".csi"):
        if ext == "" or os.path.exists(cadd + ext):
            os.symlink(os.path.abspath(cadd + ext), os.path.join(r, "CADDv1.7", "hg38", "whole_genome_SNVs.tsv.gz" + ext))
    for ext in ("", ".fai"):
        os.symlink(os.path.join(os.path.abspath(bundle), "grch37", "hg19.fa" + ext), os.path.join(r, "hg19", "hg19.fa" + ext))
    for d in ("hg38tohg19_ensembl", "hg19tohg38_ensembl"):
        os.symlink(os.path.join(os.path.abspath(bundle), "chain", d), os.path.join(r, d))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split-spliceai", metavar="VCF", help="split Illumina's single SpliceAI file into --out")
    ap.add_argument("--out", help="output folder for --split-spliceai")
    ap.add_argument("--spliceai", help=f"folder of per-chromosome SpliceAI files ({PATTERN.format('N')})")
    ap.add_argument("--cadd", help="CADD v1.7 GRCh38 whole_genome_SNVs.tsv.gz (with .tbi)")
    ap.add_argument("--bundle", help="MetaSplice Release 2.0 data bundle (grch37/hg19.fa, chain/)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.split_spliceai:
        split_spliceai(args.split_spliceai, args.out or ap.error("--out is required with --split-spliceai"))
        return 0
    if not (args.spliceai and args.cadd and args.bundle):
        ap.error("--spliceai, --cadd and --bundle are required")
    for c in ORDER:
        if not os.path.exists(os.path.join(args.spliceai, PATTERN.format(c))):
            ap.error(f"{os.path.join(args.spliceai, PATTERN.format(c))} not found (see --split-spliceai)")

    want = originals()
    tmp = tempfile.mkdtemp(prefix="metasplice_comparators_")
    root = shim_root(args.spliceai, args.cadd, args.bundle, tmp)
    edit = ('P = "/mnt/Work0/Splice_altering_variant"', f'P = "{root}"')
    v08 = load("v08", os.path.join(BUILD, "V08_comparators.py"), edit)
    positions = json.load(open(os.path.join(HERE, "stripped_columns.json")))
    ok = []
    try:
        print("1. data/v2_comparators.csv (V08)", flush=True)
        step1_comparators(v08, args.workers, tmp)
        ok.append(check("data/v2_comparators.csv", want["data/v2_comparators.csv"]))
        cmp_text = as_text(pd.read_csv(os.path.join(ROOT, "data", "v2_comparators.csv"), dtype=str, keep_default_na=False),
                           ["spliceai_max", "cadd_phred"])

        print("2. fresh cohort (V09c)", flush=True)
        fresh = step2_fresh(v08, args.workers, tmp)
        rel = "data/v2_features_fresh_20260928.csv"
        restore(rel, positions[rel], fresh)
        ok.append(check(rel, want[rel]))

        print("3. pooled test set (V12)", flush=True)
        rel = "data/v2_features_pooled_test.csv"
        def pooled(r, head):
            k = tuple(r[head.index(c)] for c in KEY)
            src = cmp_text if r[head.index("source")] == "2026-04-15" else fresh
            return src.get(k, ["", ""])
        restore(rel, positions[rel], pooled)
        ok.append(check(rel, want[rel]))

        print("4. results/S12_mechanism.tsv", flush=True)
        rel = "results/S12_mechanism.tsv"
        restore(rel, positions[rel], {k: v + v for k, v in cmp_text.items()})
        ok.append(check(rel, want[rel]))

        print("5. data/v2_gnomad_spliceai.csv (V08b, unchanged)", flush=True)
        v08b = load("v08b", os.path.join(BUILD, "V08b_gnomad_spliceai.py"), edit)
        v08b.v08 = v08                 # V08b loads V08 itself, with the original root; use the one loaded here
        v08b.main()
        ok.append(check("data/v2_gnomad_spliceai.csv", want["data/v2_gnomad_spliceai.csv"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{sum(ok)} of {len(ok)} files equal the originals" + ("" if all(ok) else "; see DIFFERS above"))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
