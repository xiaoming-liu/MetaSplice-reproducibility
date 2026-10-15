"""V08 -- Comparator scores for the v2.0 variant sets (docs/METASPLICE_V2_PLAN.md, Task 8 Step 2).

SpliceAI v1.3: the precomputed masked SNV files used here are on GRCh37, so each variant is looked up
with its GRCh37 coordinates from V02 (chr19, pos19, ref19, alt19: the GRCh37 ClinVar record with the
same ClinVar ID, or the original coordinates of a lifted set). The GRCh37 REF must equal hg19's base.
The value is v1's (S5a_spliceai_lookup.py): the max of DS_AG, DS_AL, DS_DG and DS_DL in the first
record for the allele, in file order; spliceai_max_allgenes keeps the max over all of the allele's
records (one per overlapping gene).
CADD v1.7: read on GRCh38 directly through its tabix index, by exact chromosome, position, REF and ALT.
PHRED is the score used, as in v1.

Sets: training, test candidates, VUS, literature and SpliceVarDB. The gnomAD samples have no GRCh37
coordinates and are left to Task 11. Labels are not read. A variant in more than one set gets one row,
and its GRCh37 coordinates must agree between the sets. Nothing fails silently: a missing SpliceAI
file or a failed tabix query raises an error, and unreadable SpliceAI records (v1 skipped them) are
counted in the log.

Check: the SpliceAI values must equal v1's lookup (data/S5_spliceai_train_vus.csv of the v1 deposit;
same files, same rule) wherever the GRCh37 variant is in both. CADD values are compared with v1's test
table, where v1 reached GRCh38 by its own liftover; that comparison is reported, not required.

Writes data/v2_comparators.csv and results/V08_comparators.txt.
"""
import os, subprocess, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import pandas as pd
import pysam

P = "/mnt/Work0/Splice_altering_variant"
V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1 = f"{P}/reproducibility_manuscript/data"
SPLICEAI = P + "/spliceAI_v1.3/spliceai_scores.masked.snv.hg19.vcf.gz.chr{}.gz"
CADD = f"{P}/CADDv1.7/hg38/whole_genome_SNVs.tsv.gz"
HG19 = f"{P}/hg19/hg19.fa"
SETS = ["training", "test_candidates", "vus", "literature", "splicevardb"]
KEY = ["chr", "pos", "ref", "alt"]
K19 = ["chr19", "pos19", "ref19", "alt19"]
WORKERS = int(os.environ.get("V08_WORKERS", "8"))
ORDER = [str(k) for k in range(1, 23)] + ["X", "Y"]


def load_sets():
    """All rows of the five sets (set, subset, region, GRCh38 and GRCh37 keys), and one row per variant."""
    parts = []
    for s in SETS:
        cols = KEY + K19 + ["region"] + (["subset"] if s == "test_candidates" else [])
        d = pd.read_csv(f"{V2}/data/v2_variants_{s}.csv", usecols=cols, low_memory=False)
        d["set"] = s if s != "test_candidates" else "test " + d.pop("subset")
        parts.append(d)
    a = pd.concat(parts, ignore_index=True)
    a["pos19"] = a.pos19.astype("Int64")
    has = a.loc[a.pos19.notna(), KEY + K19].drop_duplicates()
    conflict = has[has.duplicated(KEY, keep=False)]
    if len(conflict):
        raise ValueError(f"{conflict[KEY].drop_duplicates().shape[0]} variants have different GRCh37 coordinates "
                         f"in different sets, e.g. {conflict.head(4).to_dict('records')}")
    u = a[KEY].drop_duplicates().merge(has, on=KEY, how="left", validate="one_to_one")
    return a, u


def hg19_ref_ok(u):
    """True where the GRCh37 REF equals hg19's base; False where it differs or there are no coordinates."""
    fa = pysam.FastaFile(HG19)
    try:
        refs = set(fa.references)
        ok = np.zeros(len(u), bool)
        for i, (c, p, r) in enumerate(zip(u.chr19, u.pos19, u.ref19)):
            if pd.isna(p):
                continue
            if c not in refs:
                raise ValueError(f"GRCh37 contig {c} is not in {HG19}")
            ok[i] = fa.fetch(c, int(p) - 1, int(p)).upper() == r
    finally:
        fa.close()
    return ok


def spliceai_chrom(c, keys, tmp):
    """SpliceAI records for keys (pos19, ref19, alt19) on GRCh37 chromosome c: one row per allele."""
    path = SPLICEAI.format(c)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    keyf, outf = f"{tmp}/keys_{c}.txt", f"{tmp}/hits_{c}.tsv"
    keys.to_csv(keyf, sep="\t", header=False, index=False)
    awk = ("NR==FNR{k[$1\":\"$2\":\"$3]=1; next} "
           "/^#/{next} (($2\":\"$4\":\"$5) in k){print $2\"\\t\"$4\"\\t\"$5\"\\t\"$8}")
    subprocess.run(f"set -o pipefail; zcat '{path}' | awk -F'\\t' '{awk}' '{keyf}' - > '{outf}'",
                   shell=True, check=True, executable="/bin/bash")
    rows, unreadable = [], 0
    with open(outf) as f:
        for line in f:
            pos, ref, alt, info = line.rstrip("\n").split("\t")
            fl = info.split("SpliceAI=")[1].split(";")[0].split(",")[0].split("|")
            try:
                rows.append((int(pos), ref, alt, max(float(fl[2]), float(fl[3]), float(fl[4]), float(fl[5]))))
            except (ValueError, IndexError):      # counted and reported, as v1 skipped them
                unreadable += 1
    d = pd.DataFrame(rows, columns=["pos19", "ref19", "alt19", "ds"])
    g = d.groupby(["pos19", "ref19", "alt19"], sort=False)["ds"]
    out = pd.concat([g.first().rename("spliceai_max"), g.max().rename("spliceai_max_allgenes"),
                     g.size().rename("spliceai_records")], axis=1).reset_index()
    out.insert(0, "chr19", "chr" + c)
    return c, out, unreadable


def cadd_chrom(c, rows):
    """CADD (raw, PHRED) for rows (pos, ref, alt) on GRCh38 chromosome c, and how many alleles were absent
    at a position the file does cover."""
    want = {}
    for p, r, a in rows:
        want.setdefault(int(p), set()).add((r, a))
    tbx = pysam.TabixFile(CADD)
    out, other_allele = [], 0
    try:
        for p in sorted(want):
            found = {}
            for line in tbx.fetch(c, p - 1, p):
                f = line.split("\t")
                if int(f[1]) == p:
                    found[(f[2], f[3])] = (float(f[4]), float(f[5]))
            for r, a in want[p]:
                if (r, a) in found:
                    out.append((p, r, a) + found[(r, a)])
                elif found:
                    other_allele += 1
    finally:
        tbx.close()
    return c, out, other_allele


def main():
    out = open(f"{V2}/results/V08_comparators.txt", "w")
    def log(s=""):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    log("V08 -- comparator scores: SpliceAI v1.3 (GRCh37 precomputed, through the GRCh37 coordinates) and CADD v1.7 (GRCh38)")
    a, u = load_sets()
    log(f"variants: {len(u):,} distinct, from {len(a):,} set rows ({', '.join(f'{k} {v:,}' for k, v in a.set.value_counts().items())})")
    u["hg19_ref_ok"] = hg19_ref_ok(u)
    log(f"GRCh37 coordinates: present {int(u.pos19.notna().sum()):,}; REF equal to hg19 {int(u.hg19_ref_ok.sum()):,}; "
        f"REF differs {int((u.pos19.notna() & ~u.hg19_ref_ok).sum()):,}")

    t0 = time.time()
    q = u[u.hg19_ref_ok].copy()
    q["c"] = q.chr19.str.replace("chr", "", regex=False)
    tmp = tempfile.mkdtemp(prefix="v08_", dir=os.environ.get("TMPDIR", "/tmp"))
    chroms = [c for c in ORDER if (q.c == c).any()]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        res = list(ex.map(lambda c: spliceai_chrom(c, q.loc[q.c == c, ["pos19", "ref19", "alt19"]].astype(str), tmp), chroms))
    sai = pd.concat([r for _, r, _ in res], ignore_index=True)
    log(f"\nSpliceAI: {len(sai):,} of {len(q):,} GRCh37 alleles found, {time.time() - t0:.0f} s; unreadable records "
        f"{sum(n for _, _, n in res)}; first record differs from the max over all records for "
        f"{int((sai.spliceai_max != sai.spliceai_max_allgenes).sum()):,}")

    t0 = time.time()
    g = u.assign(c=u.chr.str.replace("chr", "", regex=False))
    chroms = [c for c in ORDER if (g.c == c).any()]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        res = list(ex.map(cadd_chrom, chroms, [list(g.loc[g.c == c, ["pos", "ref", "alt"]].itertuples(index=False, name=None)) for c in chroms]))
    cadd = pd.concat([pd.DataFrame(r, columns=["pos", "ref", "alt", "cadd_raw", "cadd_phred"]).assign(chr="chr" + c)
                      for c, r, _ in res], ignore_index=True)
    log(f"CADD: {len(cadd):,} of {len(u):,} GRCh38 alleles found, {time.time() - t0:.0f} s; position covered but allele absent "
        f"{sum(n for _, _, n in res):,}")

    m = (u.merge(sai, on=K19, how="left", validate="many_to_one")
          .merge(cadd, on=KEY, how="left", validate="one_to_one"))
    assert len(m) == len(u)
    m.loc[~m.hg19_ref_ok, ["spliceai_max", "spliceai_max_allgenes", "spliceai_records"]] = np.nan

    log("\nCheck: SpliceAI equals v1's lookup (same files, same rule) wherever the GRCh37 variant is in both")
    v1 = pd.read_csv(f"{V1}/S5_spliceai_train_vus.csv").rename(columns=dict(zip(KEY, K19)))
    c = m[m.spliceai_max.notna()].merge(v1, on=K19, suffixes=("", "_v1"))
    same = (c.spliceai_max == c.spliceai_max_v1) & (c.spliceai_max_allgenes == c.spliceai_max_allgenes_v1)
    log(f"  in both: {len(c):,}; identical: {int(same.sum()):,}")
    v1_only = m[m.spliceai_max.isna() & m.hg19_ref_ok].merge(v1, on=K19)
    log(f"  scored by v1's lookup but not here: {len(v1_only):,}")
    if not same.all() or len(v1_only):
        raise ValueError("the SpliceAI lookup differs from v1's on the same GRCh37 variants")

    log("\nReported: CADD against v1's test table (v1 reached GRCh38 through its own liftover)")
    tt = pd.read_csv(f"{V1}/independent_test/test_predictions_53feat_corrected.csv", usecols=KEY + ["cadd_phred"],
                     low_memory=False).rename(columns=dict(zip(KEY, K19))).dropna().drop_duplicates(K19)
    c = m[m.cadd_phred.notna()].merge(tt, on=K19, suffixes=("", "_v1"))
    log(f"  in both: {len(c):,}; identical {int((c.cadd_phred == c.cadd_phred_v1).sum()):,}; "
        f"differ {int((c.cadd_phred != c.cadd_phred_v1).sum()):,}")

    m["o"] = m.chr.str.replace("chr", "", regex=False).map({k: i for i, k in enumerate(ORDER)})
    m = m.sort_values(["o", "pos", "ref", "alt"]).drop(columns="o")
    m.to_csv(f"{V2}/data/v2_comparators.csv", index=False)

    log("\nCoverage by set and region (no labels): n; GRCh37 coordinates; hg19 REF equal; SpliceAI scored; CADD scored")
    s = a.merge(m[KEY + ["hg19_ref_ok", "spliceai_max", "cadd_phred"]], on=KEY, how="left", validate="many_to_one")
    s["has19"] = s.pos19.notna()
    for name in ["training", "test strict", "test reclassified", "vus", "literature", "splicevardb"]:
        d = s[s.set == name]
        if name == "splicevardb":
            d = d.drop_duplicates(KEY)           # assay records -> variants
        for region, r in [("all", d)] + sorted(d.groupby("region")):
            log(f"  {name:<18} {region:<24} {len(r):>8,}  {r.has19.mean():8.2%}  {r.hg19_ref_ok.mean():8.2%}  "
                f"{r.spliceai_max.notna().mean():8.2%}  {r.cadd_phred.notna().mean():8.2%}")
    log(f"\nwrote data/v2_comparators.csv ({len(m):,} rows)")
    out.close()


if __name__ == "__main__":
    main()
