"""V02c -- The production scorer and V02 agree on scope (docs/METASPLICE_V2_PLAN.md, Task 3).

V02 classifies with the scorer's scope() function. This runs the scorer's command line instead
(score_v2.py --features-only --include-out-of-scope, no feature sources), which adds input parsing
and the REF check, on complete inputs, and requires the same in-scope variants with the same region,
gene and gene ID:
  1. every P/LP or B/LB SNV of the 2025-08-31 GRCh38 release          -> the V02 training set;
  2. every P/LP or B/LB SNV of the 2026-04-15 GRCh38 release, less the
     training variants (by variant or ClinVar ID)                      -> the V02 test candidates;
  3. the literature set given as GRCh37 input (--genome hg19), less
     training variants                                                 -> the V02 literature set.
It then records, for every set, the deep-intronic counts of the primary scope and of the wider
sensitivity set (decision O5: neither canonical nor exonic in any GENCODE v50 transcript).

Writes results/V02c_scope_agreement.txt; exits 1 on any disagreement.
"""
import importlib.util, os, subprocess, sys
import pandas as pd

P = "/mnt/Work0/Splice_altering_variant"
V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W = f"{P}/tools/v2_scope_agreement"
spec = importlib.util.spec_from_file_location("v02", os.path.join(os.path.dirname(os.path.abspath(__file__)), "V02_variant_sets.py"))
v02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v02)
failures = []


def check(log, name, ok, detail=""):
    log(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures.append(name)


def write_vcf(path, d, cols=("chr", "pos", "ref", "alt"), id_col=None):
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        ids = d[id_col] if id_col else ["."] * len(d)
        for (c, p, r, a), i in zip(d[list(cols)].itertuples(index=False), ids):
            f.write(f"{c}\t{p}\t{i}\t{r}\t{a}\t.\tPASS\t.\n")


def scorer(vcf, name, *extra):
    """The command-line scorer, features only, every row reported with its scope."""
    out = f"{W}/{name}.tsv"
    r = subprocess.run([sys.executable, f"{P}/MetaSplice/score_v2.py", "--input", vcf, "--output", out,
                        "--features-out", f"{W}/{name}.features.tsv", "--features-only", "--include-out-of-scope",
                        "--fasta", v02.HG38, "--gtf", v02.FULL, *extra], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    os.remove(f"{W}/{name}.features.tsv")
    return pd.read_csv(out, sep="\t", dtype={"region": str, "gene": str, "gene_id": str}, keep_default_na=False)


def compare(log, label, got, want, key_got, key_want):
    """In-scope variants and their region, gene and gene ID: scorer command line against V02."""
    g = got.assign(k=v02.key(got, key_got)).set_index("k")[["region", "gene", "gene_id"]]
    w = want.assign(k=v02.key(want, key_want)).set_index("k")[["region", "gene", "gene_id"]].astype(str)
    same_set = set(g.index) == set(w.index)
    check(log, f"{label}: the same in-scope variants", same_set,
          f"scorer {len(g):,}, V02 {len(w):,}; only scorer {len(set(g.index) - set(w.index)):,}, only V02 {len(set(w.index) - set(g.index)):,}")
    both = g.index.intersection(w.index)
    for c in ["region", "gene", "gene_id"]:
        diff = int((g.loc[both, c] != w.loc[both, c]).sum())
        check(log, f"{label}: the same {c}", diff == 0, f"{diff:,} differ")


def di_counts(log, name, d, label_col=None):
    di = d.region == "Deep Intronic"
    sens = di & ~d.canonical_any_gencode.astype(bool) & ~d.exonic_any_gencode.astype(bool)
    lab = (lambda m: f" (P/LP {int(d.loc[m, label_col].sum()):,})") if label_col else (lambda m: "")
    log(f"  {name:<32} n={len(d):>9,}{lab(d.index == d.index)}; Deep Intronic, primary scope {int(di.sum()):>8,}{lab(di)}; "
        f"wider sensitivity set {int(sens.sum()):>8,}{lab(sens)}")


def main():
    os.makedirs(W, exist_ok=True)
    out = open(f"{V2}/results/V02c_scope_agreement.txt", "w")
    def log(s=""):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    log("V02c -- the command-line scorer and V02 agree on scope")
    data = f"{V2}/data"
    tr = pd.read_csv(f"{data}/v2_variants_training.csv", low_memory=False)
    te = pd.read_csv(f"{data}/v2_variants_test_candidates.csv", low_memory=False)
    lit = pd.read_csv(f"{data}/v2_variants_literature.csv", low_memory=False)

    log("\n1. ClinVar 2025-08-31, every P/LP or B/LB SNV")
    c25 = v02.read_clinvar(v02.CV38["2025"], v02.PATHOGENIC | v02.BENIGN)
    write_vcf(f"{W}/c25.vcf", c25, id_col="clinvar_id")
    s25 = scorer(f"{W}/c25.vcf", "c25")
    log(f"  scorer: {len(s25):,} rows; in scope {int(s25.in_scope.sum()):,}; out of scope "
        f"{s25.out_of_scope_reason[~s25.in_scope].value_counts().to_dict()}")
    compare(log, "2025 release vs training", s25[s25.in_scope], tr, ["chr", "pos", "ref", "alt"], ["chr", "pos", "ref", "alt"])

    log("\n2. ClinVar 2026-04-15, every P/LP or B/LB SNV, less training variants")
    c26 = v02.read_clinvar(v02.CV38["2026"], v02.PATHOGENIC | v02.BENIGN)
    write_vcf(f"{W}/c26.vcf", c26, id_col="clinvar_id")
    s26 = scorer(f"{W}/c26.vcf", "c26")
    s26 = s26[s26.in_scope].copy()
    s26["clinvar_id"] = v02.key(s26).map(dict(zip(v02.key(c26), c26.clinvar_id)))
    in_tr = v02.key(s26).isin(set(v02.key(tr))) | s26.clinvar_id.isin(set(tr.clinvar_id))
    log(f"  scorer: in scope {len(s26):,}; in the training data {int(in_tr.sum()):,}")
    compare(log, "2026 release vs test candidates", s26[~in_tr], te, ["chr", "pos", "ref", "alt"], ["chr", "pos", "ref", "alt"])

    log("\n3. Literature, GRCh37 input")
    src = pd.read_csv(v02.LIT)
    write_vcf(f"{W}/lit19.vcf", src)
    sl = scorer(f"{W}/lit19.vcf", "lit19", "--genome", "hg19", "--liftover-dir", v02.LIFT19)
    sl = sl[sl.in_scope].copy()
    for c in ["chr38", "pos38", "ref38", "alt38"]:
        sl[c] = sl[c].astype(str)
    in_tr = v02.key(sl, ["chr38", "pos38", "ref38", "alt38"]).isin(set(v02.key(tr)))
    log(f"  scorer: in scope {len(sl):,}; in the training data {int(in_tr.sum())}")
    compare(log, "literature (hg19 input) vs V02 literature", sl[~in_tr], lit, ["chr38", "pos38", "ref38", "alt38"],
            ["chr", "pos", "ref", "alt"])

    log("\n4. Deep-intronic counts: primary scope and the wider sensitivity set (O5)")
    di_counts(log, "training", tr, "label")
    for sub in ["strict", "reclassified"]:
        di_counts(log, f"test set, {sub}", te[te.subset == sub], "label")
    di_counts(log, "VUS", pd.read_csv(f"{data}/v2_variants_vus.csv", low_memory=False))
    di_counts(log, "literature", lit, "label")
    sv = pd.read_csv(f"{data}/v2_variants_splicevardb.csv", low_memory=False)
    di_counts(log, "SpliceVarDB records", sv)
    for f, name in [("v2_variants_gnomad_reference.csv", "gnomAD reference, uniform"),
                    ("v2_variants_gnomad_reference_stratified.csv", "gnomAD reference, stratified")]:
        if os.path.exists(f"{data}/{f}"):
            di_counts(log, name, pd.read_csv(f"{data}/{f}", low_memory=False))
        else:
            log(f"  {name}: not yet written (V02b)")
    log(f"\nV02c: {'ALL CHECKS PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    out.close()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
