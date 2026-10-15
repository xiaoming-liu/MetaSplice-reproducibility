"""A6 -- SpliceVarDB external validation at the variant level (Amendment 3).

v1's A6 on v2.0. SpliceVarDB is distributed as assay records; variants are analysed once each
(common.load_splicevardb). Policy, as in v1:
  * the primary analysis is variant level, on assay-concordant variants; the 40 variants called
    splice-altering by one assay and low-frequency splice-altering by another are excluded and reported
    separately;
  * strict endpoint: splice-altering against normal (12,731 variants: 8,015 against 4,716); broad
    endpoint: splice-altering or low-frequency against normal (22,274: 17,558 against 4,716);
  * pooled and within-region auPRC for MetaSplice v2.0, SpliceAI and CADD, with paired differences;
    sensitivity and false-positive rate at scores 0.1, 0.2 and 0.5;
  * secondary: the assay records by source, with a variant-clustered bootstrap.
The Deep Intronic rows are accompanied by the deep-intronic sensitivity set (decision O5).
SpliceVarDB measures splicing, not pathogenicity.

Writes results/A6_splicevardb.txt.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from common import (OUT, REGIONS, ROOT, Tee, add_comparators, auprc, boot_ci, deep_intronic_sensitivity, in_training,
                    load_splicevardb, paired_boot_diff, score, variant_key)

COMPARATORS = [("MetaSplice v2.0", "MetaSplice"), ("SpliceAI", "spliceai_max"), ("CADD v1.7", "cadd_phred")]
THRESHOLDS = (0.1, 0.2, 0.5)
EXPECTED = {"strict": (12731, 8015, 4716), "broad": (22274, 17558, 4716)}       # Amendment 3


def cluster_boot_ci(y, s, clusters, n_boot=1000, seed=42, stat=average_precision_score):
    """Bootstrap resampling whole variants, so repeated assay records stay together (v1)."""
    y = np.asarray(y); s = np.asarray(s, dtype=float); clusters = np.asarray(clusters)
    ok = ~np.isnan(s)
    y, s, clusters = y[ok], s[ok], clusters[ok]
    if y.sum() < 3 or (y == 0).sum() < 3:
        return (np.nan, np.nan, np.nan)
    order = {c: i for i, c in enumerate(pd.unique(clusters))}
    cid = np.array([order[c] for c in clusters])
    rows_of = {}
    for i, c in enumerate(cid):
        rows_of.setdefault(c, []).append(i)
    cl_pos = sorted({c for c, r in rows_of.items() if y[r[0]] == 1})
    cl_neg = sorted({c for c, r in rows_of.items() if y[r[0]] == 0})
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(cl_pos, len(cl_pos), replace=True), rng.choice(cl_neg, len(cl_neg), replace=True)])
        idx = np.concatenate([rows_of[c] for c in pick])
        if y[idx].sum() < 3 or (y[idx] == 0).sum() < 3:
            continue
        vals.append(stat(y[idx], s[idx]))
    return (stat(y, s), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def cluster_boot_prop(values, clusters, thr, n_boot=1000, seed=42):
    """CI for the proportion at or above `thr`, resampling variants not records (v1)."""
    values = np.asarray(values, dtype=float); clusters = np.asarray(clusters)
    ok = ~np.isnan(values)
    values, clusters = values[ok], clusters[ok]
    rows_of = {}
    for i, c in enumerate(clusters):
        rows_of.setdefault(c, []).append(i)
    keys = list(rows_of)
    rng = np.random.default_rng(seed)
    props = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([rows_of[keys[j]] for j in pick])
        props.append(float((values[idx] >= thr).mean()))
    return (float((values >= thr).mean()), float(np.percentile(props, 2.5)), float(np.percentile(props, 97.5)))


def pooled_block(out, sub, y):
    out("\n  POOLED (region-confounded; interpret with care)")
    out(f"  {'Tool':<16}{'N':>8}{'cov':>7}{'auPRC':>9}{'95% CI':>21}{'AUROC':>9}")
    for name, col in COMPARATORS:
        s = sub[col].values.astype(float)
        ok = ~np.isnan(s)
        est, lo, hi = boot_ci(y[ok], s[ok])
        au = roc_auc_score(y[ok], s[ok]) if len(set(y[ok])) > 1 else np.nan
        out(f"  {name:<16}{ok.sum():>8,}{ok.mean():>7.0%}{est:>9.4f}   [{lo:.4f}, {hi:.4f}]{au:>9.4f}")
    out("\n  Paired, MetaSplice v2.0 minus comparator (variants both score):")
    for name, col in COMPARATORS[1:]:
        diff, lo, hi, n = paired_boot_diff(y, sub["MetaSplice"].values, sub[col].values)
        out(f"    vs {name:<12} N={n:>7,}  diff {diff:+.4f}  [{lo:+.4f}, {hi:+.4f}]")


def region_masks(sub):
    for reg in REGIONS:
        m = (sub.region == reg).values
        if m.any():
            yield reg, m
            if reg == "Deep Intronic":
                yield "Deep Intronic, sensitivity set", deep_intronic_sensitivity(sub).values


def within_region(out, sub, y):
    out("\n  WITHIN REGION (the interpretable comparison)")
    hdr = f"  {'Region':<32}{'N':>7}{'pos':>6}" + "".join(f"{n[:11]:>13}" for n, _ in COMPARATORS)
    out(hdr); out("  " + "-" * (len(hdr) - 2))
    for reg, m in region_masks(sub):
        yy = y[m]
        if min(yy.sum(), (yy == 0).sum()) < 6:
            out(f"  {reg:<32}{m.sum():>7,}{int(yy.sum()):>6,}   (SUPPRESSED: only {int(min(yy.sum(), (yy == 0).sum()))} in the minority class)")
            continue
        out(f"  {reg:<32}{m.sum():>7,}{int(yy.sum()):>6,}" + "".join(f"{auprc(yy, sub[col].values[m]):>13.4f}" for _, col in COMPARATORS))
    out("\n  Within-region AUROC (prevalence-independent):")
    out(f"  {'Region':<32}{'N':>7}{'':>6}" + "".join(f"{n[:11]:>13}" for n, _ in COMPARATORS))
    for reg, m in region_masks(sub):
        yy = y[m]
        if min(yy.sum(), (yy == 0).sum()) < 6:
            continue
        cells = ""
        for _, col in COMPARATORS:
            s = sub[col].values[m].astype(float)
            ok = ~np.isnan(s)
            cells += f"{roc_auc_score(yy[ok], s[ok]):>13.4f}" if len(set(yy[ok])) > 1 else f"{'n/a':>13}"
        out(f"  {reg:<32}{m.sum():>7,}{'':>6}{cells}")
    out("\n  Paired per region, MetaSplice v2.0 minus comparator (auPRC):")
    for reg, m in region_masks(sub):
        yy = y[m]
        if min(yy.sum(), (yy == 0).sum()) < 6:
            continue
        for name, col in COMPARATORS[1:]:
            diff, lo, hi, n = paired_boot_diff(yy, sub["MetaSplice"].values[m], sub[col].values[m])
            out(f"    {reg:<32} vs {name:<10} N={n:>6,}  diff {diff:+.4f}  [{lo:+.4f}, {hi:+.4f}]")


def main():
    out = Tee(f"{OUT}/results/A6_splicevardb.txt")
    out("=" * 88)
    out("A6 -- SpliceVarDB EXTERNAL VALIDATION, VARIANT LEVEL (MetaSplice v2.0; Amendment 3)")
    out("=" * 88)
    out("Target: experimentally measured splice disruption, not ClinVar pathogenicity. MetaSplice was trained on")
    out("pathogenicity; SpliceAI on splicing. Read the per-region numbers: classes are distributed very unevenly")
    out("across regions.")
    rec = pd.read_csv(f"{ROOT}/data/v2_features_splicevardb.csv", low_memory=False)
    if in_training(rec).any():
        raise ValueError("SpliceVarDB records include training variants")
    rec["k"] = variant_key(rec)
    uni = add_comparators(load_splicevardb())
    uni["MetaSplice"] = score(uni)
    uni["k"] = variant_key(uni)
    prim = uni[uni.assay_concordant].copy()
    out(f"\nAssay records: {len(rec):,}; unique SNVs: {len(uni):,} ({int((uni.n_assay_records > 1).sum()):,} built from "
        f"more than one record); assay-discordant (excluded): {int((~uni.assay_concordant).sum())}; primary set: {len(prim):,}")
    out("\nClass composition (primary, variant level):")
    out(prim.svdb_classes.value_counts().to_string())
    out("\nRegion x class:")
    out(pd.crosstab(prim.region, prim.svdb_classes).to_string())

    for key, label, col in [("strict", "STRICT (splice-altering against normal; low-frequency excluded)", "label_strict"),
                            ("broad", "BROAD (splice-altering or low-frequency against normal)", "label_broad")]:
        sub = prim[prim[col] >= 0].reset_index(drop=True)
        y = sub[col].astype(int).values
        got = (len(sub), int(y.sum()), int((y == 0).sum()))
        if got != EXPECTED[key]:
            raise SystemExit(f"{key} endpoint {got} differs from Amendment 3's {EXPECTED[key]}")
        out("\n" + "=" * 88)
        out(label)
        out("=" * 88)
        out(f"N={len(sub):,}  positive={int(y.sum()):,}  negative={int((y == 0).sum()):,}  prevalence={y.mean():.1%} (equals Amendment 3)")
        pooled_block(out, sub, y)
        within_region(out, sub, y)

    sa, nm, lf = (prim[prim.svdb_classes == c] for c in ["Splice-altering", "Normal", "Low-frequency splice-altering"])
    out("\n" + "=" * 88)
    out("OPERATING POINTS (MetaSplice v2.0 score)")
    out("=" * 88)
    out(f"   {'threshold':>9}{'sens (SA)':>13}{'sens (low-freq)':>18}{'FPR (normal)':>15}")
    for t in THRESHOLDS:
        out(f"   {t:>9}{(sa.MetaSplice >= t).mean():>12.1%}{(lf.MetaSplice >= t).mean():>17.1%}{(nm.MetaSplice >= t).mean():>14.1%}")
    for name, sel in [("Deep Intronic", lambda d: d.region == "Deep Intronic"),
                      ("Deep Intronic, sensitivity set", deep_intronic_sensitivity)]:
        dsa, dnm = sa[sel(sa).values], nm[sel(nm).values]
        out(f"\n  {name} only (n={len(dsa)} splice-altering, {len(dnm)} normal):")
        for t in THRESHOLDS:
            out(f"   {t:>9}{(dsa.MetaSplice >= t).mean():>12.1%}{'':>17}{(dnm.MetaSplice >= t).mean():>14.1%}")

    out("\n" + "=" * 88)
    out("SECONDARY: BY ASSAY SOURCE (assay records, variant-clustered bootstrap)")
    out("=" * 88)
    ms = uni.set_index("k").MetaSplice
    r = rec[rec.svdb_class == "Splice-altering"].copy()
    r["ms"] = r.k.map(ms)
    out(f"  {'assay source':<14}{'records':>9}{'variants':>10}{'sens@0.1':>11}{'95% CI':>20}")
    for src, sub in r.groupby("dataset"):
        if len(sub) < 25:
            continue
        est, lo, hi = cluster_boot_prop(sub.ms.values, sub.k.values, 0.1)
        out(f"  {src:<14}{len(sub):>9,}{sub.k.nunique():>10,}{est:>10.1%}   [{lo:5.1%}, {hi:5.1%}]")
    sr = rec[rec.svdb_class.isin(["Splice-altering", "Normal"])].copy()
    sr["ms"] = sr.k.map(ms)
    est, lo, hi = cluster_boot_ci((sr.svdb_class == "Splice-altering").astype(int).values, sr.ms.values, sr.k.values)
    out(f"\n  Record-level strict auPRC, variant-clustered CI: {est:.4f} [{lo:.4f}, {hi:.4f}] "
        f"(N={len(sr):,} records, {sr.k.nunique():,} variants); reported for completeness, the primary analysis is variant level.")

    dis = uni[~uni.assay_concordant]
    out("\n" + "=" * 88)
    out("ASSAY-DISCORDANT VARIANTS (excluded from the primary analysis)")
    out("=" * 88)
    out(f"{len(dis)} SNVs were called splice-altering by one assay and low-frequency splice-altering by another.")
    out(f"  MetaSplice v2.0 >= 0.1: {(dis.MetaSplice >= 0.1).mean():.1%} ({int((dis.MetaSplice >= 0.1).sum())}/{len(dis)})")
    out(f"  dataset pairs: {dis.datasets.value_counts().to_dict()}")
    out(f"  by region: {dis.region.value_counts().to_dict()}")
    out.close()


if __name__ == "__main__":
    main()
