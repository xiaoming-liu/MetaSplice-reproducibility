"""V09a -- ClinVar VCF to the tab-separated summary V02 reads (docs/METASPLICE_V2_PLAN.md, Task 11 Step 8).

The summaries used so far (clinvar_20250831, clinvar_20260415) were made by a step not kept in the repository. This
writes the same layout from a ClinVar VCF: #chr, pos, clinvar_id (the VCF ID), ref, alt, then INFO fields CLNSIG,
CLNDN, CLNREVSTAT, CLNHGVS, CLNVI and CLNDISDB, with "." for an absent field.

Validation (--check): the 2026-04-15 GRCh38 VCF (NCBI archive) must reproduce the existing 2026-04-15 summary
exactly in every column V02 reads (#chr, pos, clinvar_id, ref, alt, CLNSIG, CLNREVSTAT), record for record.

Usage: V09a_clinvar_summary.py <clinvar.vcf.gz> <out.summary.gz> [--check <existing.summary.gz>]
"""
import gzip, sys
import pandas as pd

FIELDS = ["CLNSIG", "CLNDN", "CLNREVSTAT", "CLNHGVS", "CLNVI", "CLNDISDB"]
USED = ["#chr", "pos", "clinvar_id", "ref", "alt", "CLNSIG", "CLNREVSTAT"]


def write_summary(vcf, out):
    n = 0
    with gzip.open(vcf, "rt") as f, gzip.open(out, "wt") as o:
        o.write("\t".join(["#chr", "pos", "clinvar_id", "ref", "alt"] + FIELDS) + "\n")
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            info = dict(kv.split("=", 1) if "=" in kv else (kv, "") for kv in c[7].split(";"))
            o.write("\t".join([c[0], c[1], c[2], c[3], c[4]] + [info.get(k, ".") for k in FIELDS]) + "\n")
            n += 1
    return n


def main():
    vcf, out = sys.argv[1], sys.argv[2]
    n = write_summary(vcf, out)
    print(f"wrote {out}: {n:,} records")
    if "--check" in sys.argv:
        ref = sys.argv[sys.argv.index("--check") + 1]
        a = pd.read_csv(out, sep="\t", usecols=USED, dtype=str, keep_default_na=False)
        b = pd.read_csv(ref, sep="\t", usecols=USED, dtype=str, keep_default_na=False)
        same = len(a) == len(b) and (a.values == b.values).all()
        print(f"check against {ref}: {len(a):,} vs {len(b):,} records; columns {USED} identical: {same}")
        if not same:
            diff = (a.values != b.values).any(axis=1) if len(a) == len(b) else None
            raise SystemExit(f"the summary differs from the existing one ({'row count' if diff is None else int(diff.sum())} rows)")


if __name__ == "__main__":
    main()
