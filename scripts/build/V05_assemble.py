"""V05 -- Per-set feature tables and the missingness-parity check (docs/METASPLICE_V2_PLAN.md, Task 5).

Splits data/v2_features_union.tsv (V04, the v2.0 scorer) into one table per variant set, joined to the
set's V02 columns (labels, ClinVar IDs, subsets, the sensitivity flags, GRCh37 coordinates). The
scorer's region and gene must equal V02's for every variant. Then two checks (refined with the user,
2026-10-01):
  - check_gene_level_values: every represented gene has one set of pLI, LOEUF and ClinGen HI values
    across all tables, equal to the dbNSFP 5.4 gene file's for its Ensembl ID (read independently);
  - check_missingness_parity: every feature's missing rate, within each region of each set, within 5
    percentage points of training's. LOEUF and ClinGen HI are reported, not gated: their gaps are
    properties of the genes (no LOEUF for any chrX or chrY gene; HI curated for few genes). pLI stays
    gated as the rate at which genes are found. The one other difference, phyloP470way in the
    literature set's Donor Region, is accepted only after its positions are confirmed absent from the
    source wigFix file.
These are the checks that would have caught v1's hg38 bridge defect and its gene-level lookup defect.

Writes data/v2_features_{set}.csv, results/V05_missingness.tsv and results/V05_assemble.txt.
"""
import gzip, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from common import FEATURES, SETS, check_gene_level_values, check_missingness_parity, missingness_table

V2 = os.path.dirname(os.path.dirname(HERE))
KEY = ["chr", "pos", "ref", "alt"]
DBNSFP = "/mnt/Work0/Splice_altering_variant/reproducibility/raw_inputs/dbNSFP5.4_gene.gz"
PHYLOP = "/mnt/Work0/Splice_altering_variant/reproducibility/raw_inputs/phyloP_GRCh38"


def wigfix_values(path, positions):
    """{position: value or None} read straight from a wigFix file (fixedStep blocks, step 1)."""
    want, found, cur = set(positions), {}, 0
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("fixedStep"):
                cur = int(line.split("start=")[1].split()[0])
                if cur > max(want):
                    break
                continue
            if cur in want:
                found[cur] = float(line)
            cur += 1
    return {p: found.get(p) for p in positions}


def phylop470_exception(tables, log):
    """The literature Donor Region difference in phyloP470way, accepted only if every missing position
    is absent from the source wigFix file; any position with a source value raises."""
    lit = tables["literature"]
    miss = lit[(lit.region == "Donor Region") & lit.phyloP470way_mammalian.isna()]
    notes = []
    for c, sub in miss.groupby("chr"):
        vals = wigfix_values(f"{PHYLOP}/{c}.phyloP470way.wigFix.gz", sub.pos.tolist())
        for p, v in vals.items():
            if v is not None:
                raise ValueError(f"literature {c}:{p}: phyloP470way is {v} in the source but missing in the table")
            notes.append(f"{c}:{p}")
    reason = (f"{len(notes)} of {int((lit.region == 'Donor Region').sum())} variants, at positions the 470-way "
              f"wigFix does not cover (checked against the source: {', '.join(notes)})")
    log(f"  phyloP470way, literature Donor Region: {reason}")
    return {("literature", "Donor Region", "phyloP470way_mammalian"): reason}


def main():
    out = open(f"{V2}/results/V05_assemble.txt", "w")
    def log(s=""):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    log("V05 -- per-set feature tables from the scorer's union table, and the missingness-parity check")
    f = pd.read_csv(f"{V2}/data/v2_features_union.tsv", sep="\t", low_memory=False)
    assert not f.duplicated(KEY).any()
    tables = {}
    for name in SETS:
        v = pd.read_csv(f"{V2}/data/v2_variants_{name}.csv", low_memory=False)
        m = v.merge(f.drop(columns=["status"]), on=KEY, how="left", suffixes=("", "_scorer"), validate="many_to_one")
        if m[FEATURES[-1]].isna().any():
            raise ValueError(f"{name}: {int(m[FEATURES[-1]].isna().sum())} variants have no scorer row")
        for c in ["region", "gene", "gene_id"]:
            diff = int((m[c].astype(str) != m[f"{c}_scorer"].astype(str)).sum())
            if diff:
                raise ValueError(f"{name}: {c} differs between V02 and the scorer for {diff} variants")
        if "gnomad_af_scorer" in m:        # the gnomAD samples carry V02b's allele frequency as well
            # both must be present first: max() skips NaN, so a missing value would compare as equal
            na = int(m.gnomad_af.isna().sum()) + int(m.gnomad_af_scorer.isna().sum())
            if na:
                raise ValueError(f"{name}: gnomAD AF missing in {na} V02b or scorer values")
            d = (m.gnomad_af - m.gnomad_af_scorer).abs().max()
            if d > 0:
                raise ValueError(f"{name}: gnomAD AF differs between V02b and the scorer (max {d})")
            log(f"  {name}: gnomAD AF present in V02b and the scorer for all {len(m):,} rows, and equal")
            m = m.drop(columns=["gnomad_af_scorer"])
        m = m.drop(columns=["region_scorer", "gene_scorer", "gene_id_scorer"])
        m.to_csv(f"{V2}/data/v2_features_{name}.csv", index=False)
        tables[name] = m
        log(f"  {name:<28} {len(m):>9,} rows; region, gene and gene ID equal V02's")
    miss = missingness_table(tables)
    miss.to_csv(f"{V2}/results/V05_missingness.tsv", sep="\t", index=False)
    log("\nMissing rate by feature (all regions), per set:")
    w = pd.DataFrame({name: t[FEATURES].isna().mean() for name, t in tables.items()})
    w = w[(w > 0).any(axis=1)]
    log(w.map(lambda x: f"{x:.2%}").to_string())
    log("\nGene-level values against the dbNSFP 5.4 gene file, and across tables:")
    check_gene_level_values(tables, DBNSFP, log=log)
    log("  PASS")
    log("\nParity with training within each region (tolerance 5 points; groups of >= 100 variants; "
        "LOEUF and ClinGen HI reported above, not gated):")
    check_missingness_parity(tables, explained=phylop470_exception(tables, log), log=log)
    log("  PASS: every gated feature, region and set within tolerance, or the one verified exception")
    out.close()


if __name__ == "__main__":
    main()
