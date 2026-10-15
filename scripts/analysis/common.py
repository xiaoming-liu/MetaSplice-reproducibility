"""Shared definitions, loaders and checks for the MetaSplice v2.0 analyses (docs/METASPLICE_V2_PLAN.md).

Task 5 added check_missingness_parity, the standing check that would have caught v1's hg38 bridge
defect (71,362 training rows with no conservation, GPN-MSA or allele frequency). Task 8 adds the
loaders ported from v1 (load_train, load_test, load_literature, in_training), score(), the comparator
columns and the statistics the analyses share. Everything reads the v2.0 tables in data/ and models/;
nothing here computes features.
"""
import functools
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("METASPLICE_V2_ROOT", os.path.dirname(os.path.dirname(_HERE)))
OUT = ROOT
SCORER = os.environ.get("METASPLICE_V2_SCORER", "/mnt/Work0/Splice_altering_variant/MetaSplice/score_v2.py")

_spec = importlib.util.spec_from_file_location("score_v2", SCORER)
scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorer)
FEATURES = scorer.FEATURES                     # the 38 features, in model order

GENE_LEVEL = ["gnomAD_pLI", "gnomAD_LOEUF", "ClinGen_Haploinsufficiency_Score"]
# Gene-level features whose missingness is a property of the gene in dbNSFP (no LOEUF for any chrX or
# chrY gene; ClinGen HI curated for few genes), so it follows each set's gene mix. Their missing rates
# are reported, not gated; check_gene_level_values checks their values gene by gene against the
# source instead. pLI stays gated: its presence is the rate at which genes are found at all.
REPORTED_NOT_GATED = ["gnomAD_LOEUF", "ClinGen_Haploinsufficiency_Score"]
SETS = ["training", "test_candidates", "vus", "literature", "splicevardb", "gnomad_reference",
        "gnomad_reference_stratified"]
NONCANONICAL = ["Donor Region", "Acceptor Region", "PPT Region", "Branch Point Region", "Deep Intronic"]
CANONICAL = ["Canonical Donor Site", "Canonical Acceptor Site"]
REGIONS = CANONICAL + NONCANONICAL
NONCANONICAL_REGIONS = NONCANONICAL                 # v1's name, kept for the ported scripts
KEY = ["chr", "pos", "ref", "alt"]
MODEL_DIR = f"{ROOT}/models"
# The comparators of the v2.0 protocol, and their score columns (data/v2_comparators.csv, built by V08)
COMPARATORS = [("SpliceAI", "spliceai_max"), ("CADD", "cadd_phred")]
# The ClinVar release the training labels came from, both builds (raw inputs pinned in results/V00_inputs.txt);
# override with METASPLICE_V2_CLINVAR_2025_GRCH38 / _GRCH37 where the package is moved.
CLINVAR_2025 = {
    "GRCh38": os.environ.get("METASPLICE_V2_CLINVAR_2025_GRCH38",
                             "/mnt/Work0/Splice_altering_variant/clinvar/hg38/clinvar_20250831.vcf.gz.summary.gz"),
    "GRCh37": os.environ.get("METASPLICE_V2_CLINVAR_2025_GRCH37",
                             "/mnt/Work0/Splice_altering_variant/reproducibility/raw_inputs/clinvar_20250831.vcf.gz.summary.gz")}
TEST_SUBSETS = ("strict", "reclassified")


def missingness_table(tables, features=FEATURES):
    """Missing rate of every feature, by table and region: DataFrame (table, region, feature, n, missing)."""
    rows = []
    for name, t in tables.items():
        for region, sub in t.groupby("region"):
            for f in features:
                rows.append((name, region, f, len(sub), float(sub[f].isna().mean())))
    return pd.DataFrame(rows, columns=["table", "region", "feature", "n", "missing"])


def check_missingness_parity(tables, features=None, reference="training", tol=0.05, explained=None, log=print,
                             min_n=100):
    """Raise unless each feature's missing rate, within each region, is within `tol` (absolute) of the
    reference table's rate, for every table. A difference is accepted only when `explained` names it,
    {(table, region, feature): reason}; the reasons are logged, so they reach the results log. Groups
    smaller than min_n are reported in the table but not checked: their rates are too noisy.

    v1's training table lacked conservation, GPN-MSA and allele frequency for 18.7% of rows, almost
    all benign, while every test table had them: a feature missing at a different rate in training
    than where the model is applied puts the model out of distribution without an error."""
    explained = explained or {}
    features = [f for f in FEATURES if f not in REPORTED_NOT_GATED] if features is None else features
    m = missingness_table(tables, features)
    ref = m[m.table == reference].set_index(["region", "feature"]).missing
    m["reference"] = [ref.get((r, f), float("nan")) for r, f in zip(m.region, m.feature)]
    m["difference"] = m.missing - m.reference
    bad = m[(m.table != reference) & (m.difference.abs() > tol) & (m.n >= min_n)]
    problems = []
    for r in bad.itertuples():
        k = (r.table, r.region, r.feature)
        if k in explained:
            log(f"  accepted: {r.table} / {r.region} / {r.feature}: missing {r.missing:.1%} vs {reference} "
                f"{r.reference:.1%}: {explained[k]}")
        else:
            problems.append(f"{r.table} / {r.region} / {r.feature}: missing {r.missing:.1%} (n={r.n:,}) vs "
                            f"{reference} {r.reference:.1%}")
    if problems:
        raise ValueError(f"missingness differs from {reference} by more than {tol:.0%}:\n  " + "\n  ".join(problems))
    return m


def dbnsfp_gene_values(path):
    """The three gene-level values per Ensembl gene ID (version dropped), read straight from the dbNSFP
    gene file: the first value of each ';'-separated field, '.' as NaN; an ID listed in more than one
    row takes the last row's values (the scorer's stated rule, implemented here independently)."""
    d = pd.read_csv(path, sep="\t", dtype=str, usecols=["Ensembl_gene"] + GENE_LEVEL)
    for c in GENE_LEVEL:
        d[c] = pd.to_numeric(d[c].str.split(";").str[0].where(lambda x: ~x.isin([".", ""])), errors="coerce")
    d = d.assign(eid=d.Ensembl_gene.str.split(";")).explode("eid")
    d = d[d.eid.notna() & ~d.eid.isin([".", ""])]
    return d.drop_duplicates("eid", keep="last").set_index("eid")[GENE_LEVEL]


def check_gene_level_values(tables, dbnsfp_path, log=print):
    """Raise unless every represented gene has one set of gene-level values across all tables, and those
    values equal the dbNSFP gene file's for its Ensembl ID (all three missing when the ID is absent).
    Agreement across tables alone is weak when the tables come from one scoring run; the source check
    is the independent one."""
    rows = pd.concat([t[["gene_id"] + GENE_LEVEL] for t in tables.values()], ignore_index=True)
    distinct = rows.astype(str).drop_duplicates()
    multi = distinct.gene_id.value_counts()
    inconsistent = multi[multi > 1]
    genes = rows.drop_duplicates("gene_id").reset_index(drop=True)
    src = dbnsfp_gene_values(dbnsfp_path)
    key = genes.gene_id.str.split(".").str[0]
    want = src.reindex(key.values).values.astype(float)
    got = genes[GENE_LEVEL].values.astype(float)
    same = np.isclose(got, want, rtol=0, atol=1e-12, equal_nan=True).all(axis=1)
    found = key.isin(src.index).values
    log(f"  genes represented {len(genes):,}; found in the dbNSFP gene file {int(found.sum()):,}; "
        f"with more than one set of values across tables {len(inconsistent)}; "
        f"values equal to the source {int(same.sum()):,} of {len(genes):,}")
    if len(inconsistent) or not same.all():
        bad = genes.gene_id[~same].head(5).tolist()
        raise ValueError(f"gene-level values: {len(inconsistent)} genes inconsistent across tables; "
                         f"{int((~same).sum())} differ from the source, e.g. {bad}")


# ----------------------------------------------------------------------------- variants and the training data

def _flag(df, col):
    """A boolean column as written by V02; refuses any other dtype (astype(bool) would read 'False' as True)."""
    if df[col].dtype != bool:
        raise TypeError(f"{col} has dtype {df[col].dtype}, expected bool")
    return df[col]


def variant_key(df):
    """chr:pos:ref:alt on GRCh38, with the 'chr' prefix the v2.0 tables use."""
    c = df["chr"].astype(str)
    c = c.where(c.str.startswith("chr"), "chr" + c)
    return c + ":" + df["pos"].astype(int).astype(str) + ":" + df["ref"].astype(str) + ":" + df["alt"].astype(str)


@functools.lru_cache(maxsize=1)
def training_variants():
    """(variant keys, ClinVar IDs) of every v2.0 training variant."""
    tr = pd.read_csv(f"{ROOT}/data/v2_variants_training.csv", usecols=KEY + ["clinvar_id"], low_memory=False)
    return frozenset(variant_key(tr)), frozenset(tr["clinvar_id"])


@functools.lru_cache(maxsize=1)
def training_release():
    """Every record of the 2025-08-31 release the training labels came from: GRCh38 variant keys, and the
    ClinVar IDs and CLNSIG of both builds (V02 takes the release's IDs from both)."""
    r = pd.read_csv(CLINVAR_2025["GRCh38"], sep="\t", usecols=["#chr", "pos", "clinvar_id", "ref", "alt", "CLNSIG"],
                    dtype={"#chr": str, "ref": str, "alt": str}, low_memory=False).rename(columns={"#chr": "chr"})
    r37 = pd.read_csv(CLINVAR_2025["GRCh37"], sep="\t", usecols=["clinvar_id", "CLNSIG"], low_memory=False)
    clnsig = pd.concat([r[["clinvar_id", "CLNSIG"]], r37]).drop_duplicates("clinvar_id").set_index("clinvar_id").CLNSIG
    return frozenset(variant_key(r)), frozenset(clnsig.index), clnsig


def in_training(df):
    """True for rows whose variant, or ClinVar ID where the table has one, is in the v2.0 training data."""
    keys, ids = training_variants()
    hit = variant_key(df).isin(keys)
    if "clinvar_id" in df.columns:
        hit |= df["clinvar_id"].isin(ids)
    return hit


def deep_intronic_sensitivity(df):
    """Deep Intronic rows at positions neither canonical nor exonic in any GENCODE v50 transcript: the wider
    sensitivity set of decision O5, reported beside every deep-intronic result with its own denominator."""
    return (df["region"] == "Deep Intronic") & ~_flag(df, "canonical_any_gencode") & ~_flag(df, "exonic_any_gencode")


def load_train():
    """The v2.0 training variants (V05) with their out-of-fold scores (V06): columns oof and fold added."""
    tr = pd.read_csv(f"{ROOT}/data/v2_features_training.csv", low_memory=False)
    oof = pd.read_csv(f"{ROOT}/data/v2_oof.csv", low_memory=False)
    m = tr.merge(oof, on=KEY, how="left", validate="one_to_one", suffixes=("", "_oof"))
    if len(m) != len(tr) or m.oof.isna().any():
        raise ValueError(f"{int(m.oof.isna().sum())} training variants have no out-of-fold score")
    for c in ["clinvar_id", "gene_id", "region"]:
        if (m[c].astype(str) != m[f"{c}_oof"].astype(str)).any():
            raise ValueError(f"{c} differs between v2_features_training.csv and v2_oof.csv")
    if (m.label.astype(float) != m.label_oof.astype(float)).any():
        raise ValueError("label differs between v2_features_training.csv and v2_oof.csv")
    m = m.drop(columns=["clinvar_id_oof", "gene_id_oof", "region_oof", "label_oof"])
    m["label"] = m["label"].astype(int)
    m["is_canonical"] = m["region"].isin(CANONICAL)
    return m


def load_test(subset="strict"):
    """Test candidates of the 2026-04-15 release (V02), defined against the training data by v1's rule
    of 2026-09-29.

    subset="strict" (the test set): absent from the training data, by variant and by ClinVar ID, and from
      the 2025-08-31 release the training labels came from, by GRCh38 variant or by ClinVar ID (either build).
    subset="reclassified": absent from the training data but in the 2025-08-31 release with a classification
      training does not use. A subgroup for S7, not part of the test set.

    The rule is recomputed at every load and must reproduce V02's subset column on every row; no returned
    variant is in the training data. Adds is_canonical; label as int.
    """
    if subset not in TEST_SUBSETS:
        raise ValueError(f"subset must be one of {TEST_SUBSETS}, not {subset!r}")
    t = pd.read_csv(f"{ROOT}/data/v2_features_test_candidates.csv", low_memory=False)
    rel_keys, rel_ids, _ = training_release()
    trained = in_training(t)
    in_release = variant_key(t).isin(rel_keys) | t["clinvar_id"].isin(rel_ids)
    rule = np.where(trained, "training", np.where(in_release, "reclassified", "strict"))
    if (rule != t["subset"].values).any():
        raise ValueError(f"{int((rule != t['subset'].values).sum())} test candidates: the recomputed subset differs from V02's")
    keep = rule == subset
    print(f"  [test candidates: {len(t):,}; in the training data {int(trained.sum())}; in the 2025-08-31 release "
          f"{int((~trained & in_release).sum()):,}; {subset}: n={int(keep.sum()):,}, P/LP={int(t['label'][keep].sum()):,}]")
    t = t.loc[keep].copy()
    assert not in_training(t).any()
    t["label"] = t["label"].astype(int)
    t["is_canonical"] = t["region"].isin(CANONICAL)
    return t.reset_index(drop=True)


def load_literature():
    """Literature-confirmed splice-altering variants (all P/LP; v1's mapped set lifted to GRCh38, in scope; V02),
    less any variant in the training data, as in v1."""
    lit = pd.read_csv(f"{ROOT}/data/v2_features_literature.csv", low_memory=False)
    trained = in_training(lit)
    print(f"  [literature set: {len(lit):,} variants; excluded {int(trained.sum())} in the training data; "
          f"n={int((~trained).sum()):,}]")
    lit = lit.loc[~trained].copy()
    lit["label"] = lit["label"].astype(int)
    return lit.reset_index(drop=True)


def load_splicevardb():
    """SpliceVarDB at the variant level, by v1's rule (A6): one row per SNV (features are per variant), with
    svdb_classes (every class its assay records report), n_assay_records and datasets. assay_concordant is
    False where the records disagree; those variants are excluded from the primary analysis.
    label_strict: 1 splice-altering, 0 normal, -1 low-frequency or discordant (excluded);
    label_broad: 1 splice-altering or low-frequency, 0 normal, -1 discordant (excluded)."""
    d = pd.read_csv(f"{ROOT}/data/v2_features_splicevardb.csv", low_memory=False)
    if in_training(d).any():
        raise ValueError(f"{int(in_training(d).sum())} SpliceVarDB records are variants in the training data")
    k = variant_key(d)
    g = d.groupby(k, sort=False)
    u = d.loc[~k.duplicated()].copy()
    u.index = variant_key(u)
    u["svdb_classes"] = g["svdb_class"].agg(lambda s: "|".join(sorted(set(s))))
    u["n_assay_records"] = g.size()
    u["datasets"] = g["dataset"].agg(lambda s: "|".join(sorted(set(s.astype(str)))))
    u["assay_concordant"] = ~u["svdb_classes"].str.contains("|", regex=False)
    cls = u["svdb_classes"].where(u["assay_concordant"])
    u["label_strict"] = cls.map({"Splice-altering": 1, "Normal": 0}).fillna(-1).astype(int)
    u["label_broad"] = cls.map({"Splice-altering": 1, "Low-frequency splice-altering": 1, "Normal": 0}).fillna(-1).astype(int)
    return u.drop(columns=["svdb_class", "dataset"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- scores

@functools.lru_cache(maxsize=1)
def _model():
    import joblib
    names = list(joblib.load(f"{MODEL_DIR}/metasplice_v2_features.joblib"))
    if names != FEATURES:
        raise ValueError("the stored feature list differs from the scorer's FEATURES")
    return names, joblib.load(f"{MODEL_DIR}/metasplice_v2_full.joblib")


def score(df):
    """MetaSplice v2.0 scores from the full model, with the feature order stored beside the model."""
    names, model = _model()
    missing = [f for f in names if f not in df.columns]
    if missing:
        raise KeyError(f"feature columns absent from the input table: {missing}")
    return model.predict_proba(df[names].astype(np.float32).values)[:, 1]


@functools.lru_cache(maxsize=1)
def _comparators():
    return pd.read_csv(f"{ROOT}/data/v2_comparators.csv", usecols=KEY + [c for _, c in COMPARATORS], low_memory=False)


def add_comparators(df):
    """df with the comparator scores (spliceai_max, cadd_phred) joined by variant. Every variant must be in
    data/v2_comparators.csv (V08: training, test candidates, VUS, literature, SpliceVarDB); a missing value
    there means the comparator does not score the variant."""
    m = df.drop(columns=[c for _, c in COMPARATORS if c in df.columns]).merge(
        _comparators().assign(_looked_up=True), on=KEY, how="left", validate="many_to_one")
    if len(m) != len(df) or m["_looked_up"].isna().any():
        raise ValueError(f"{int(m['_looked_up'].isna().sum())} variants are not in data/v2_comparators.csv")
    return m.drop(columns="_looked_up")


# ----------------------------------------------------------------------------- statistics (from v1)

def auprc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    if ok.sum() == 0 or y[ok].sum() < 3 or (y[ok] == 0).sum() < 3:
        return np.nan
    return average_precision_score(y[ok], s[ok])


def boot_ci(y, s, n_boot=1000, seed=42, stat=average_precision_score):
    """Stratified bootstrap CI for a single score."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if y.sum() < 3 or (y == 0).sum() < 3:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        vals.append(stat(y[idx], s[idx]))
    return (stat(y, s), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def paired_boot_diff(y, s1, s2, n_boot=1000, seed=42, stat=average_precision_score):
    """Paired stratified bootstrap for stat(s1) - stat(s2) on variants both tools score.

    Returns (diff, lo, hi, n_paired). The pairing is what makes this a test rather
    than two overlapping marginal intervals.
    """
    y = np.asarray(y); s1 = np.asarray(s1, dtype=float); s2 = np.asarray(s2, dtype=float)
    ok = ~np.isnan(s1) & ~np.isnan(s2)
    y, s1, s2 = y[ok], s1[ok], s2[ok]
    if y.sum() < 3 or (y == 0).sum() < 3:
        return (np.nan, np.nan, np.nan, int(ok.sum()))
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        diffs.append(stat(y[idx], s1[idx]) - stat(y[idx], s2[idx]))
    return (stat(y, s1) - stat(y, s2),
            float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), int(len(y)))


class Tee:
    """Write to stdout and a results file simultaneously."""
    def __init__(self, path):
        self.f = open(path, "w")

    def __call__(self, *args):
        line = " ".join(str(a) for a in args)
        print(line)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def load_fresh():
    """The fresh temporal cohort of ClinVar 2026-09-28 (V09b, V09c): variants first classified P/LP or B/LB after
    2026-04-15, with features and comparator scores. Rechecked at every load: no variant is in the training data, the
    strict test set or the reclassified subgroup, by variant or by ClinVar ID. Adds is_canonical; label as int."""
    f = pd.read_csv(f"{ROOT}/data/v2_features_fresh_20260928.csv", low_memory=False)
    te = pd.read_csv(f"{ROOT}/data/v2_variants_test_candidates.csv", usecols=KEY + ["clinvar_id"], low_memory=False)
    seen = in_training(f) | variant_key(f).isin(set(variant_key(te))) | f["clinvar_id"].isin(set(te["clinvar_id"]))
    if seen.any():
        raise ValueError(f"{int(seen.sum())} fresh variants are in the training data or the 2026-04-15 test candidates")
    f["label"] = f["label"].astype(int)
    f["is_canonical"] = f["region"].isin(CANONICAL)
    return f
