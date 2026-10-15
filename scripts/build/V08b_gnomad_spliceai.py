"""V08b -- SpliceAI scores of the gnomAD reference samples (docs/METASPLICE_V2_PLAN.md, Task 11, Step 5).

Secondary to the prior, which uses MetaSplice v2.0 only. It serves the SpliceAI reference-window
configuration and population yield of S11_recalibrate. The rule was fixed in the plan before any score
was examined. Each variant is lifted from GRCh38 to GRCh37, then looked up:
  1. mapping: a unique GRCh38->GRCh37 position (score_v2.load_one_to_one on hg38tohg19_ensembl, as in
     lift_variants) that is reciprocal: the GRCh37 position has exactly one GRCh38 position
     (load_one_to_one on hg19tohg38_ensembl), and it is the variant's own position;
  2. orientation from the neighbouring positions (score_v2.lift_variants); on an inverted mapping both
     alleles are complemented;
  3. REF must equal hg19's base (score_v2.lift_variants);
  4. the exact allele is looked up with V08's SpliceAI rule (V08_comparators.spliceai_chrom: the largest
     of the four delta scores in the first record for the allele).
A variant that fails a step, or that SpliceAI does not score, stays missing; its status names the first
step it failed. Attrition is reported by sample and region.

Writes data/v2_gnomad_spliceai.csv and results/V08b_gnomad_spliceai.txt.
"""
import importlib.util, os, sys, tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P = "/mnt/Work0/Splice_altering_variant"
V2 = os.path.dirname(os.path.dirname(HERE))
FWD, REV, HG19 = f"{P}/hg38tohg19_ensembl", f"{P}/hg19tohg38_ensembl", f"{P}/hg19/hg19.fa"
KEY = ["chr", "pos", "ref", "alt"]
ORDER = [str(k) for k in range(1, 23)] + ["X", "Y"]
STATUS = ["scored", "no unique GRCh38->GRCh37 position", "mapping not reciprocal",
          "orientation not settled by neighbouring positions", "GRCh37 contig not in hg19",
          "REF differs from hg19", "not scored by SpliceAI"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ms = load("score_v2", f"{P}/MetaSplice/score_v2.py")
v08 = load("v08", os.path.join(HERE, "V08_comparators.py"))


def main():
    out = open(f"{V2}/results/V08b_gnomad_spliceai.txt", "w")
    def log(s=""):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    log("V08b -- SpliceAI v1.3 for the gnomAD reference samples, by the rule fixed in the plan (Task 11)")
    ref = pd.read_csv(f"{V2}/data/v2_gnomad_reference_scores.csv", usecols=KEY + ["region", "sample"], low_memory=False)
    u = ref[KEY].drop_duplicates().reset_index(drop=True)
    log(f"variants: {len(u):,} distinct ({len(ref):,} sample rows)")
    status = np.array(["no unique GRCh38->GRCh37 position"] * len(u), dtype=object)

    # 1. unique forward mapping (as lift_variants loads it), then reciprocity
    fwd = {}
    by_chr = defaultdict(list)
    for i, c in enumerate(u.chr):
        by_chr[c.replace("chr", "")].append(i)
    for c, idx in by_chr.items():
        mp = ms.load_one_to_one(c, FWD, {int(u.pos[i]) + d for i in idx for d in (-1, 0, 1)})
        for i in idx:
            if int(u.pos[i]) in mp:
                fwd[i] = mp[int(u.pos[i])]
    back = {}
    by19 = defaultdict(set)
    for i, (c19, q) in fwd.items():
        by19[str(c19).replace("chr", "")].add(q)
    for c19, qs in by19.items():
        mp = ms.load_one_to_one(c19, REV, qs)
        back.update({(c19, q): t for q, t in mp.items()})
    reciprocal = set()
    for i, (c19, q) in fwd.items():
        t = back.get((str(c19).replace("chr", ""), q))
        if t is not None and str(t[0]).replace("chr", "") == u.chr[i].replace("chr", "") and int(t[1]) == int(u.pos[i]):
            reciprocal.add(i)
        else:
            status[i] = "mapping not reciprocal"

    # 2-3. orientation, complement and REF check: the scorer's own lift
    v = [tuple(r) if i in reciprocal else None for i, r in enumerate(u[KEY].itertuples(index=False, name=None))]
    lifted, failed = ms.lift_variants(v, FWD, HG19)
    reasons = {"no one-to-one position in the other build": "no unique GRCh38->GRCh37 position",
               "orientation not settled by neighbouring positions": "orientation not settled by neighbouring positions",
               "REF differs between builds": "REF differs from hg19"}
    for i, why in failed.items():
        status[i] = reasons.get(why, "GRCh37 contig not in hg19" if "not in" in why else why)
    u["chr19"], u["pos19"], u["ref19"], u["alt19"], u["strand"] = None, np.nan, None, None, np.nan
    for i, (c19, q, r2, a2, strand) in lifted.items():
        u.loc[i, ["chr19", "pos19", "ref19", "alt19", "strand"]] = [c19, q, r2, a2, strand]
    ok = u.pos19.notna()
    log(f"lifted (unique, reciprocal, oriented, REF equal to hg19): {int(ok.sum()):,}; inverted mappings "
        f"(alleles complemented): {int((u.strand == -1).sum()):,}")

    # 4. V08's SpliceAI rule on the exact GRCh37 allele
    q = u[ok].copy()
    q["pos19"] = q.pos19.astype(int)
    q["c"] = q.chr19.str.replace("chr", "", regex=False)
    tmp = tempfile.mkdtemp(prefix="v08b_", dir=os.environ.get("TMPDIR", "/tmp"))
    chroms = [c for c in ORDER if (q.c == c).any()]
    with ThreadPoolExecutor(max_workers=int(os.environ.get("V08_WORKERS", "8"))) as ex:
        res = list(ex.map(lambda c: v08.spliceai_chrom(c, q.loc[q.c == c, ["pos19", "ref19", "alt19"]].astype(str), tmp), chroms))
    sai = pd.concat([r for _, r, _ in res], ignore_index=True)
    log(f"SpliceAI: {len(sai):,} of {len(q):,} lifted alleles found; unreadable records {sum(n for _, _, n in res)}")
    u["pos19"] = u.pos19.astype("Int64")
    m = u.merge(sai, on=["chr19", "pos19", "ref19", "alt19"], how="left", validate="many_to_one")
    assert len(m) == len(u)
    status[(ok & m.spliceai_max.isna()).values] = "not scored by SpliceAI"
    status[(ok & m.spliceai_max.notna()).values] = "scored"
    m["status"] = status
    m[KEY + ["chr19", "pos19", "ref19", "alt19", "strand", "status", "spliceai_max", "spliceai_max_allgenes"]].to_csv(
        f"{V2}/data/v2_gnomad_spliceai.csv", index=False)

    log("\nAttrition by sample and region (first failed step; share of the sample's variants in the region)")
    s = ref.merge(m[KEY + ["status"]], on=KEY, how="left", validate="many_to_one")
    for sample in ["uniform", "stratified"]:
        d = s[s["sample"] == sample]
        tab = pd.crosstab(d.region, d.status).reindex(columns=STATUS, fill_value=0)
        tab.loc["all"] = tab.sum()
        log(f"\n  {sample} sample")
        log("  " + (tab.T / tab.sum(axis=1)).T.map(lambda x: f"{x:.2%}").assign(n=tab.sum(axis=1).map(lambda x: f"{x:,}"))
            .to_string().replace("\n", "\n  "))
    log(f"\nwrote data/v2_gnomad_spliceai.csv ({len(m):,} rows)")
    out.close()


if __name__ == "__main__":
    main()
