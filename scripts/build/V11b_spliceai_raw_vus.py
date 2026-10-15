"""V11b -- SpliceAI raw scores (+-4,999 nt) for the VUS and resolved-VUS cohorts (protocol Amendment 8, P3).

Reuses V11 unchanged: SpliceAI 1.3.1 (-A grch37 -D 4999 -M 0), its asserted annotation and model files, V08's GRCh37
coordinates, hg19 REF check and value rule (the first record for the allele; the all-genes maximum kept beside it), and
V11's installation check (-D 50 -M 1 on a seeded sample, >= 99% within 0.01 of V08's precomputed value) before any raw
score. Variants: S5's VUS yield cohort (61,016; paper_cohorts.load_vus_cohort, S5's logic) and S7's resolved-VUS cohort (659). The cohort loaders read labels with
the variants; labels are not used to select SpliceAI settings or to calculate its predictions.

Writes data/v2_spliceai_raw_d4999_vus.csv and results/V11b_spliceai_raw_vus.txt.
"""
import argparse, importlib.util, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from common import NONCANONICAL
from calib import one_star_cohorts, pathogenic_genes, was_vus_2025
from paper_cohorts import load_vus_cohort


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


v11 = load("v11", os.path.join(HERE, "V11_spliceai_raw.py"))
KEY, K19 = v11.KEY, v11.K19


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--check-only", action="store_true", help="installation check only; no raw score")
    args = ap.parse_args()
    out = open(f"{V2}/results/V11b_spliceai_raw_vus.txt", "w") if not args.check_only else open(os.devnull, "w")
    def log(s=""):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    log("V11b -- SpliceAI 1.3.1 raw scores, +-4,999 nt, for the VUS and resolved-VUS cohorts (Amendment 8)")
    bad = []
    for f, want in v11.SPLICEAI_MD5.items():
        got = v11.md5(f"{v11.PKG}/{f}")
        log(f"  {got}  spliceai/{f}  {'ok' if got == want else 'DIFFERS from ' + want}")
        bad += [f] if got != want else []
    if bad:
        raise SystemExit(f"SpliceAI files differ from Amendment 7: {bad}")
    fai = [(l.split("\t")[0], int(l.split("\t")[1])) for l in open(v11.HG19 + ".fai")]

    yld = load_vus_cohort(pathogenic_genes())
    _, val = one_star_cohorts("reclassified")
    res = val[was_vus_2025(val)]
    if (len(yld), (len(res), int(res.label.sum()))) != (61016, (659, 107)):
        raise SystemExit(f"cohorts differ from Amendment 8: VUS {len(yld):,}, resolved {len(res)} ({int(res.label.sum())})")
    u = pd.concat([d[KEY + K19 + ["region", "SpliceAI"]] for d in (yld, res)]).drop_duplicates(KEY).reset_index(drop=True)
    log(f"\nVUS cohort {len(yld):,} and resolved-VUS cohort {len(res)}: {len(u):,} distinct variants; "
        f"GRCh37 record {int(u.pos19.notna().sum()):,}")
    ok = v11.v08.hg19_ref_ok(u.assign(pos19=u.pos19.astype("Int64")))
    q = u[ok].copy()
    q["pos19"] = q.pos19.astype(int)
    log(f"  hg19 REF equal: {len(q):,}; precomputed masked score present: {int(q.SpliceAI.notna().sum()):,}")

    pool = q[q.SpliceAI.notna() & q.region.isin(NONCANONICAL)]
    samp = pd.concat([g.sample(min(100, len(g)), random_state=42) for _, g in pool.groupby("region")])
    log(f"\nInstallation check: -D 50 -M 1 on {len(samp):,} variants (seed 42; up to 100 per non-canonical region)")
    r = v11.run_spliceai(samp, "vus_check_d50_m1", 50, 1, min(args.chunks, 2), fai, log)
    got = samp.apply(lambda x: r.get((x.chr19, x.pos19, x.ref19, x.alt19), (np.nan,))[0], axis=1)
    got_all = samp.apply(lambda x: r.get((x.chr19, x.pos19, x.ref19, x.alt19), (np.nan, np.nan))[1], axis=1)
    diff = (got - samp.SpliceAI).abs()
    within = float((diff <= 0.01 + 1e-9).mean())
    log(f"  within 0.01 of V08's precomputed value: {within:.2%} (n={len(samp)}); unscored {int(got.isna().sum())}; "
        f"max |diff| {np.nanmax(diff.values):.3f}; identical {float((diff < 1e-9).mean()):.2%}; "
        f"variants differing by > 0.01 with more than one record: {int(((diff > 0.01) & (got_all != got)).sum())}")
    if within < 0.99:
        raise SystemExit("installation check failed (< 99% within 0.01); no raw score computed")
    log("  PASS")
    if args.check_only:
        return

    log(f"\nRaw scores: -D 4999 -M 0 on {len(q):,} variants")
    r = v11.run_spliceai(q, "vus_raw_d4999_m0", 4999, 0, args.chunks, fai, log)
    vals = q.apply(lambda x: r.get((x.chr19, x.pos19, x.ref19, x.alt19), (np.nan, np.nan, 0)), axis=1, result_type="expand")
    q["spliceai_raw_max"], q["spliceai_raw_max_allgenes"], q["spliceai_raw_records"] = vals[0], vals[1], vals[2].astype(int)
    m = u[KEY + K19].merge(q[KEY + ["spliceai_raw_max", "spliceai_raw_max_allgenes", "spliceai_raw_records"]], on=KEY,
                           how="left", validate="one_to_one")
    m.to_csv(f"{V2}/data/v2_spliceai_raw_d4999_vus.csv", index=False)
    for name, d in [("VUS cohort", yld), ("resolved-VUS cohort", res)]:
        j = d[KEY].merge(m[KEY + ["spliceai_raw_max"]], on=KEY, how="left")
        log(f"  {name:<22} n={len(d):>7,}  raw {int(j.spliceai_raw_max.notna().sum()):>7,}  masked {int(d.SpliceAI.notna().sum()):>7,}")
    both = q.SpliceAI.notna() & q.spliceai_raw_max.notna()
    log(f"\nVariants with both scores: {int(both.sum()):,}; raw >= masked for {float((q.spliceai_raw_max[both] >= q.SpliceAI[both]).mean()):.2%}; "
        f"first record below the all-genes maximum for {int((q.spliceai_raw_max_allgenes - q.spliceai_raw_max > 1e-9).sum()):,}")
    log(f"wrote data/v2_spliceai_raw_d4999_vus.csv ({len(m):,} rows; md5 {v11.md5(f'{V2}/data/v2_spliceai_raw_d4999_vus.csv')}); "
        "labels not used for SpliceAI settings or predictions")
    out.close()


if __name__ == "__main__":
    main()
