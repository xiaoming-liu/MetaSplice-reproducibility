"""S15a -- The exact population and numerator of the lower bound on the intronic prior (used by S15, section 1).

The lower bound is (pathogenic variants of the population) / (the population), where the population is every rare
intronic SNV that gnomAD v4.1 observes in the calibration's scope. V02b builds that population as a benign reference
and so removes every ClinVar variant before its scope filters; its eligible count is the population less its ClinVar
variants. This script repeats V02b's classification on the same files with the same code, keeping the ClinVar
variants apart instead of removing them, so that both parts of the population pass the same filters:
  1. gnomAD v4.1 joint SNVs, chromosomes 1-22 and X, FILTER PASS, 0 < joint AF < 0.01 (the tables hold only FILTER
     PASS SNVs: MetaSplice/docker_v2/bundle/tools/build_gnomad_tables.py skips every other record, and their
     gnomAD_joint_flag is always '.', so V02b's flag test removes nothing);
  2. V02b's two positional prefilters (inside a MANE transcript span; not inside a MANE exon), which remove only
     variants the exact rule would also put out of scope;
  3. the scorer's scope() on the full GENCODE v50 GTF, region one of the five non-canonical regions;
  4. a gene with >= 1 P/LP SNV in the training release (data/v2_pathogenic_genes.csv).
Checks (exit 1 on failure): the variants in neither ClinVar release reproduce V02b's eligible count in every region
and V02b's ClinVar count after the prefilters; no variant appears twice; chromosome Y, which V02b leaves out, adds no
eligible variant (its only gene with a P/LP SNV in the training release is SRY, which has no intron).
Numerator: eligible variants with a P/LP record (Pathogenic, Likely_pathogenic, Pathogenic/Likely_pathogenic) and
>= 1 review star in the training release (2025-08-31), and no B/LB record of the same variant in that release. They
must be exactly the >= 1-star P/LP of the training calibration cohort with 0 < AF < 0.01, with the same region,
gene ID and AF (checked; exit 1 otherwise).
Writes results/S15a_prior_population.txt and data/v2_gnomad_eligible_clinvar.csv (the eligible ClinVar variants
with region, gene ID, joint AF and their 2025-08-31 classification).
"""
import importlib.util, os, sys, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
from common import OUT, Tee
from calib import ZERO_STAR, load_derivation_cohort

spec = importlib.util.spec_from_file_location("v02b", os.path.join(os.path.dirname(HERE), "build", "V02b_gnomad_reference.py"))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
PATHOGENIC = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}     # as V02
BENIGN = {"Benign", "Likely_benign", "Benign/Likely_benign"}
V02B = f"{OUT}/results/V02b_gnomad_reference.txt"
OUT_CSV = f"{OUT}/data/v2_gnomad_eligible_clinvar.csv"
G = {}      # filled in the parent before the pool forks; the workers inherit it


def setup():
    g = b.ms.parse_mane_gtf(b.FULL)          # lru_cache: parsed once here, inherited by the workers
    spans = {}
    for gid, sp in g["spans"].items():
        for c, s, e in sp:
            spans.setdefault(c, []).append((s, e))
    G["spans"] = {c: b.merged(v) for c, v in spans.items()}
    G["exons"] = {c: b.merged([(s, e) for s, e, _, _ in v]) for c, v in g["exons_by_chr"].items()}
    G["pgenes"] = set(pd.read_csv(f"{OUT}/data/v2_pathogenic_genes.csv").gene_id)
    cv = set()
    for f in b.CV38:
        c = pd.read_csv(f, sep="\t", usecols=["#chr", "pos", "ref", "alt"], dtype={"#chr": str, "ref": str, "alt": str})
        cv |= set("chr" + c["#chr"] + ":" + c.pos.astype(str) + ":" + c.ref + ":" + c.alt)
    G["clinvar"] = cv


def classify(c, d):
    """V02b's classification of one part of a chromosome: (counts by outcome, eligible rows)."""
    d = d.reset_index(drop=True)
    if d.empty:
        return {}, pd.DataFrame(columns=["chr", "pos", "ref", "alt", "region", "gene_id", "gnomad_af"])
    v = [(c, p, r, a) for p, r, a in zip(d.pos.values.tolist(), d.ref.tolist(), d.alt.tolist())]
    in_scope, genes, why = b.ms.scope(v, b.FULL)
    region = pd.Series([in_scope.get(i) for i in range(len(v))], dtype=object)
    gid = pd.Series([genes.get(i, (None, None))[1] for i in range(len(v))], dtype=object)
    cnt = {f"out of scope: {k}": int(n) for k, n in pd.Series(list(why.values()), dtype=object).value_counts().items()}
    cnt["in scope, canonical region"] = int(region.isin(set(b.ms.PRIORITY) - set(b.NONCANONICAL)).sum())
    nonc = region.isin(b.NONCANONICAL)
    cnt["non-canonical, gene without P/LP in training"] = int((nonc & ~gid.isin(G["pgenes"])).sum())
    elig = (nonc & gid.isin(G["pgenes"])).values
    assert sum(cnt.values()) + int(elig.sum()) == len(d), c          # every variant has exactly one outcome
    rows = pd.DataFrame({"chr": c, "pos": d.pos.values[elig], "ref": d.ref.values[elig], "alt": d.alt.values[elig],
                         "region": region.values[elig], "gene_id": gid.values[elig], "gnomad_af": d.gnomAD_joint_AF.values[elig]})
    return cnt, rows


def run(chrom):
    t0, c = time.time(), f"chr{chrom}"
    d = pd.read_csv(b.GNOMAD.format(chrom), sep="\t", usecols=["pos", "ref", "alt", "gnomAD_joint_flag", "gnomAD_joint_AF"],
                    dtype={"ref": str, "alt": str})
    cnt = {"gnomAD SNVs read": len(d)}
    d = d[(d.gnomAD_joint_flag == ".") & (d.gnomAD_joint_AF > 0) & (d.gnomAD_joint_AF < 0.01)]
    cnt["PASS, 0 < AF < 0.01"] = len(d)
    m = b.inside(d.pos.values, *G["spans"][c]) if c in G["spans"] else np.zeros(len(d), bool)
    d = d[m]
    if c in G["exons"]:
        d = d[~b.inside(d.pos.values, *G["exons"][c])]
    cnt["after the prefilters"] = len(d)
    in_cv = (c + ":" + d.pos.astype(str) + ":" + d.ref + ":" + d.alt).isin(G["clinvar"]).values
    out = {}
    for part, sub in (("not ClinVar", d[~in_cv]), ("ClinVar", d[in_cv])):
        k, rows = classify(c, sub)
        k["classified"] = len(sub)
        k["duplicate eligible variants"] = int(rows.duplicated(["pos", "ref", "alt"]).sum())
        k.update({f"eligible {r}": int((rows.region == r).sum()) for r in b.NONCANONICAL})
        out[part] = (k, rows if part == "ClinVar" else None)
    return chrom, cnt, out, time.time() - t0


def v02b_counts():
    import re
    txt = open(V02B).read()
    line = [l for l in txt.splitlines() if l.startswith("eligible ") and "by region" in l][0]
    by_region = {r: int(n.replace(",", "")) for r, n in re.findall(r"(Deep Intronic|Branch Point Region|PPT Region|"
                                                                 r"Acceptor Region|Donor Region) ([\d,]+)", line)}
    clinvar = int(re.search(r"in a ClinVar release ([\d,]+)", txt).group(1).replace(",", ""))
    return by_region, clinvar


def numerator(log, cv):
    """P/LP (>= 1 star) of the training release among the eligible ClinVar variants, checked against the cohort."""
    r = pd.read_csv(b.CV38[0], sep="\t", usecols=["#chr", "pos", "clinvar_id", "ref", "alt", "CLNSIG", "CLNREVSTAT"],
                    dtype={"#chr": str, "ref": str, "alt": str}, low_memory=False)
    r["key"] = "chr" + r["#chr"] + ":" + r.pos.astype(str) + ":" + r.ref + ":" + r.alt
    cv["key"] = cv.chr + ":" + cv.pos.astype(str) + ":" + cv.ref + ":" + cv.alt
    r = r[r.key.isin(set(cv.key))]
    n_rec = r.groupby("key").size()
    plp = r[r.CLNSIG.isin(PATHOGENIC)]
    blb = set(r.key[r.CLNSIG.isin(BENIGN)])
    star = plp[~plp.CLNREVSTAT.isin(ZERO_STAR)]
    agg = r.groupby("key").agg(clnsig_2025=("CLNSIG", lambda s: "|".join(sorted(set(s)))),
                               review_status_2025=("CLNREVSTAT", lambda s: "|".join(sorted(set(s)))))
    cv = cv.merge(agg, left_on="key", right_index=True, how="left")
    cv["plp_2025"] = cv.key.isin(set(plp.key))
    cv["plp_2025_one_star"] = cv.key.isin(set(star.key)) & ~cv.key.isin(blb)
    log(f"\n  eligible ClinVar variants in the 2025-08-31 release {int(cv.clnsig_2025.notna().sum()):,} (records {len(r):,}; "
        f"variants with more than one record {int((n_rec > 1).sum()):,}); P/LP {int(cv.plp_2025.sum()):,}; "
        f"P/LP with >= 1 star {int(cv.key.isin(set(star.key)).sum()):,}; of these also with a B/LB record "
        f"{int((cv.key.isin(set(star.key)) & cv.key.isin(blb)).sum()):,}")
    num = cv[cv.plp_2025_one_star]
    log(f"  numerator: eligible, P/LP with >= 1 star in the training release, no B/LB record: {len(num):,}")

    der = load_derivation_cohort(one_star_only=True)
    pos = der[(der.label == 1) & (der.gnomad_af > 0) & (der.gnomad_af < 0.01)].copy()
    pos["key"] = pos.chr.astype(str) + ":" + pos.pos.astype(str) + ":" + pos.ref + ":" + pos.alt
    m = num.merge(pos[["key", "region", "gene_id", "gnomad_af"]], on="key", how="outer", suffixes=("", "_cohort"), indicator=True)
    only_here, only_cohort = int((m._merge == "left_only").sum()), int((m._merge == "right_only").sum())
    both = m[m._merge == "both"]
    same = (both.region == both.region_cohort) & (both.gene_id == both.gene_id_cohort) & np.isclose(
        both.gnomad_af.astype(float), both.gnomad_af_cohort.astype(float), rtol=0, atol=1e-12)
    log(f"  training calibration cohort, P/LP with >= 1 star and 0 < AF < 0.01: {len(pos):,}; in both {len(both):,} "
        f"(region, gene ID and AF equal for {int(same.sum()):,}); only here {only_here}; only in the cohort {only_cohort}")
    if only_here or only_cohort or not same.all():
        print(m[(m._merge != "both")].to_string())
        raise SystemExit("the numerator differs from the training calibration cohort")
    return cv.drop(columns="key")


def main():
    log = Tee(f"{OUT}/results/S15a_prior_population.txt")
    log("S15a -- EXACT POPULATION AND NUMERATOR OF THE LOWER BOUND ON THE INTRONIC PRIOR")
    log("  population: gnomAD v4.1 joint SNVs, chr1-22 and X, FILTER PASS, 0 < AF < 0.01, in scope (scorer's scope(), full "
        "GENCODE v50 GTF) in a non-canonical region, gene with >= 1 P/LP SNV in the training release; ClinVar variants kept")
    setup()
    log(f"  pathogenic genes {len(G['pgenes']):,}; ClinVar variants (either release) {len(G['clinvar']):,}")
    chroms = b.CHROMS + ["Y"]
    with Pool(12) as p:
        res = sorted(p.map(run, chroms, chunksize=1), key=lambda x: chroms.index(x[0]))
    y = [r for r in res if r[0] == "Y"][0]
    y_elig = {part: sum(y[2][part][0][f"eligible {r}"] for r in b.NONCANONICAL) for part in ("not ClinVar", "ClinVar")}
    log(f"  chrY (left out by V02b): PASS, 0 < AF < 0.01 {y[1]['PASS, 0 < AF < 0.01']:,}; after the prefilters "
        f"{y[1]['after the prefilters']:,}; eligible not ClinVar {y_elig['not ClinVar']}, ClinVar {y_elig['ClinVar']}")
    if any(y_elig.values()):
        raise SystemExit("chromosome Y has eligible variants; the population must include it")
    res = [r for r in res if r[0] != "Y"]
    tot = {"not ClinVar": {}, "ClinVar": {}}
    top, rows = {}, []
    for chrom, cnt, out, sec in res:
        for k, n in cnt.items():
            top[k] = top.get(k, 0) + n
        for part, (k, r) in out.items():
            for kk, n in k.items():
                tot[part][kk] = tot[part].get(kk, 0) + n
        rows.append(out["ClinVar"][1])
        log(f"  chr{chrom}: eligible not ClinVar {sum(out['not ClinVar'][0][f'eligible {r}'] for r in b.NONCANONICAL):,}, "
            f"ClinVar {len(out['ClinVar'][1]):,} ({sec:.0f} s)")
    log("  " + "; ".join(f"{k} {n:,}" for k, n in top.items()))
    for part in ("not ClinVar", "ClinVar"):
        log(f"  {part}: " + "; ".join(f"{k} {n:,}" for k, n in tot[part].items()))

    v_region, v_cv = v02b_counts()
    ok_region = all(tot["not ClinVar"][f"eligible {r}"] == v_region[r] for r in b.NONCANONICAL)
    log(f"\n  check: variants in neither release reproduce V02b's eligible count in every region: {'yes' if ok_region else 'NO'}; "
        f"ClinVar variants after the prefilters {tot['ClinVar']['classified']:,} (V02b {v_cv:,}): "
        f"{'equal' if tot['ClinVar']['classified'] == v_cv else 'DIFFERENT'}")
    dup = tot["not ClinVar"]["duplicate eligible variants"] + tot["ClinVar"]["duplicate eligible variants"]
    log(f"  check: duplicate eligible variants {dup}")
    if not ok_region or tot["ClinVar"]["classified"] != v_cv or dup:
        raise SystemExit("the population does not reproduce V02b")

    cv = pd.concat(rows, ignore_index=True)
    log("\n  population by region (not ClinVar + ClinVar = total):")
    for r in b.NONCANONICAL:
        a, c = v_region[r], int((cv.region == r).sum())
        log(f"    {r:<22}{a:>13,} + {c:>8,} = {a + c:>13,}")
    a, c = sum(v_region.values()), len(cv)
    log(f"    {'all five regions':<22}{a:>13,} + {c:>8,} = {a + c:>13,}")
    cv = numerator(log, cv)
    cv.to_csv(OUT_CSV, index=False)
    log(f"\nwrote data/{os.path.basename(OUT_CSV)} ({len(cv):,} rows) and results/S15a_prior_population.txt")


if __name__ == "__main__":
    main()
