"""V11c -- SpliceAI raw scores (+-4,999 nt) for the pooled calibration-cohort variants V11 and V11b did not score
(protocol Amendment 9).

Reuses V11 unchanged: SpliceAI 1.3.1 (-A grch37 -D 4999 -M 0), its asserted annotation and model files, V08's GRCh37
coordinates, hg19 REF check and value rule (the first record for the allele; the all-genes maximum kept beside it), and
V11's installation check (-D 50 -M 1 on a seeded sample, seed 42, up to 100 variants per non-canonical region of the
pooled calibration cohort with a precomputed masked score; >= 99% within 0.01 of V08's value) before any raw score.
Variants: the pooled calibration cohort (all review statuses; V12) less those with a row in V11's or V11b's output
(Amendment 9 recorded 21). Labels are read with the cohort but not used to select SpliceAI settings or to calculate its
predictions.

Writes data/v2_spliceai_raw_d4999_topup.csv and results/V11c_spliceai_raw_topup.txt.
"""
import argparse, importlib.util, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from common import NONCANONICAL
from calib import in_cohort, pathogenic_genes


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


v11 = load("v11", os.path.join(HERE, "V11_spliceai_raw.py"))
KEY, K19 = v11.KEY, v11.K19
RAW = ["data/v2_spliceai_raw_d4999.csv", "data/v2_spliceai_raw_d4999_vus.csv"]
POOLED_MD5 = "a1847d63dd5ab2e4021b1f20a01b22c8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--check-only", action="store_true", help="installation check only; no raw score")
    args = ap.parse_args()
    out = open(f"{V2}/results/V11c_spliceai_raw_topup.txt", "w") if not args.check_only else open(os.devnull, "w")
    def log(s=""):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    log("V11c -- SpliceAI 1.3.1 raw scores, +-4,999 nt, for pooled calibration-cohort variants not yet scored (Amendment 9)")
    bad = []
    for f, want in v11.SPLICEAI_MD5.items():
        got = v11.md5(f"{v11.PKG}/{f}")
        log(f"  {got}  spliceai/{f}  {'ok' if got == want else 'DIFFERS from ' + want}")
        bad += [f] if got != want else []
    got = v11.md5(f"{V2}/data/v2_features_pooled_test.csv")
    log(f"  {got}  data/v2_features_pooled_test.csv  {'ok' if got == POOLED_MD5 else 'DIFFERS from ' + POOLED_MD5}")
    bad += ["pooled"] if got != POOLED_MD5 else []
    if bad:
        raise SystemExit(f"inputs differ from Amendments 7 and 9: {bad}")
    fai = [(l.split("\t")[0], int(l.split("\t")[1])) for l in open(v11.HG19 + ".fai")]

    p = pd.read_csv(f"{V2}/data/v2_features_pooled_test.csv", low_memory=False)
    cal = p[in_cohort(p, pathogenic_genes())].rename(columns={"spliceai_max": "SpliceAI"})
    have = pd.concat([pd.read_csv(f"{V2}/{f}", usecols=KEY) for f in RAW]).drop_duplicates()
    j = cal.merge(have.assign(_have=1), on=KEY, how="left")
    todo = j[j._have.isna()].drop(columns="_have").reset_index(drop=True)
    log(f"\nPooled calibration cohort (all statuses): {len(cal):,}; with a row in V11 or V11b: {len(cal) - len(todo):,}; "
        f"to score: {len(todo)} (GRCh37 record {int(todo.pos19.notna().sum())})")
    if len(todo) != 21:
        raise SystemExit(f"{len(todo)} variants to score; Amendment 9 recorded 21")
    ok = v11.v08.hg19_ref_ok(todo.assign(pos19=todo.pos19.astype("Int64")))
    q = todo[ok].copy()
    q["pos19"] = q.pos19.astype(int)
    log(f"  hg19 REF equal: {len(q)}; precomputed masked score present: {int(q.SpliceAI.notna().sum())}")

    pool = cal[cal.SpliceAI.notna() & cal.region.isin(NONCANONICAL) & cal.pos19.notna()].copy()
    pool["pos19"] = pool.pos19.astype(int)
    samp = pd.concat([g.sample(min(100, len(g)), random_state=42) for _, g in pool.groupby("region")])
    samp = samp[v11.v08.hg19_ref_ok(samp.assign(pos19=samp.pos19.astype("Int64")))]
    log(f"\nInstallation check: -D 50 -M 1 on {len(samp):,} variants (seed 42; up to 100 per non-canonical region)")
    r = v11.run_spliceai(samp, "topup_check_d50_m1", 50, 1, args.chunks, fai, log)
    got = samp.apply(lambda x: r.get((x.chr19, x.pos19, x.ref19, x.alt19), (np.nan,))[0], axis=1)
    diff = (got - samp.SpliceAI).abs()
    within = float((diff <= 0.01 + 1e-9).mean())
    log(f"  within 0.01 of V08's precomputed value: {within:.2%} (n={len(samp)}); unscored {int(got.isna().sum())}; "
        f"max |diff| {np.nanmax(diff.values):.3f}; identical {float((diff < 1e-9).mean()):.2%}")
    if within < 0.99:
        raise SystemExit("installation check failed (< 99% within 0.01); no raw score computed")
    log("  PASS")
    if args.check_only:
        return

    log(f"\nRaw scores: -D 4999 -M 0 on {len(q)} variants")
    r = v11.run_spliceai(q, "topup_raw_d4999_m0", 4999, 0, 1, fai, log)
    vals = q.apply(lambda x: r.get((x.chr19, x.pos19, x.ref19, x.alt19), (np.nan, np.nan, 0)), axis=1, result_type="expand")
    q["spliceai_raw_max"], q["spliceai_raw_max_allgenes"], q["spliceai_raw_records"] = vals[0], vals[1], vals[2].astype(int)
    m = todo[KEY + K19].merge(q[KEY + ["spliceai_raw_max", "spliceai_raw_max_allgenes", "spliceai_raw_records"]], on=KEY,
                              how="left", validate="one_to_one")
    m.to_csv(f"{V2}/data/v2_spliceai_raw_d4999_topup.csv", index=False)
    log(f"  scored {int(m.spliceai_raw_max.notna().sum())} of {len(m)}; masked score present for {int(todo.SpliceAI.notna().sum())}")
    log(f"wrote data/v2_spliceai_raw_d4999_topup.csv ({len(m)} rows; md5 {v11.md5(f'{V2}/data/v2_spliceai_raw_d4999_topup.csv')}); "
        "labels not used for SpliceAI settings or predictions")
    out.close()


if __name__ == "__main__":
    main()
