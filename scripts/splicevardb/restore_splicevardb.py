"""Restore the SpliceVarDB classifications of this deposit from the source studies' published files.

The SpliceVarDB set (data/v2_variants_splicevardb.csv, data/v2_features_splicevardb.csv) was rebuilt for this study
from SpliceVarDB's source studies, with SpliceVarDB's published criteria (Sullivan et al., Am J Hum Genet 2024; Table 1),
and not downloaded from SpliceVarDB. The deposit keeps each record's variant, source study (dataset) and regions, but
not its classification (svdb_class). This script recomputes the classification from the studies' own files:

  --savnet    Supplemental_Table_S3.xlsx of Shiraishi et al., Genome Res 2018;28:1111-25 (SAVNet; supplementary data)
  --misplice  1-s2.0-S2211124718303978-mmc2.xlsx of Jayasinghe et al., Cell Rep 2018;23:270-81 (MiSplice; supplementary file mmc2)
  --mfass     processed_data/snv/snv_data_clean.txt of Chong et al., Mol Cell 2019;73:183-94, from the authors'
              repository https://github.com/KosuriLab/MFASS

  python restore_splicevardb.py --savnet Supplemental_Table_S3.xlsx --misplice 1-s2.0-S2211124718303978-mmc2.xlsx \
      --mfass snv_data_clean.txt

Rules (classify_all_datasets.py, next to this script, unchanged from the reconstruction):
  - SAVNet: each instance classified by supporting read count and Bayes factor (instances with Bayes factor
    "Rescued" skipped); a variant whose instances disagree is excluded.
  - MiSplice: every call on the "Passed Manual Review" sheet is splice-altering; alleles from the Site column.
  - MFASS: mutant constructs classified by delta PSI (v2_dpsi, else v1_dpsi), keyed by chr, snp_position, ref_allele
    and alt_allele.
  - The 14 records from two studies without a machine-readable table, Casadei et al. 2019 (PNAS) and
    Fraile-Bethencourt et al. 2019 (J Pathol), were extracted by hand. Their classifications are in
    literature_classifications.csv.
Each record is matched by its GRCh37 variant (chr19, pos19, ref19, alt19) and dataset. The class column is restored at
its original position, and both files are checked against scripts/comparators/originals.md5. Exits 1 if either
differs or a record has no classification.
"""
import argparse, csv, hashlib, importlib.util, io, json, os, sys, warnings
from collections import defaultdict
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FILES = ["data/v2_variants_splicevardb.csv", "data/v2_features_splicevardb.csv"]
SOURCE_MD5 = {"savnet": "9200d7df99c3b593104516d98cc06c88", "misplice": "cfc8c488c0614fd52c09fce52c18def7",
              "mfass": "1476eb9b8309c5c743ec81219461b9c9"}       # the copies the reconstruction used

spec = importlib.util.spec_from_file_location("classify_all_datasets", os.path.join(HERE, "classify_all_datasets.py"))
cad = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cad)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()


def savnet(path):
    x = pd.read_excel(path, header=2)                  # two title rows precede the header
    inst = defaultdict(set)
    for k, reads, bf in zip(x["Mutation key"], x["Supporting read count"], x["Bayes factor"]):
        if pd.isna(k) or str(bf) == "Rescued":
            continue
        inst[str(k).strip('"')].add(cad.classify_savnet_variant(int(reads), float(bf)))
    out = {}
    for k, c in inst.items():
        if len(c) == 1:                                # conflicting instances: excluded
            ch, p, r, a = k.split(",")
            out[(ch, int(p), r, a, "SAVNet")] = c.pop()
    return out


def misplice(path):
    y = pd.read_excel(path, sheet_name="Passed Manual Review")
    out = {}
    for s in y["Site"].astype(str):                    # SampleID_chr_position_ref_alt
        ch, pos, r, a = s.split("_")[-4:]
        out[("chr" + ch.removeprefix("chr"), int(pos), r, a, "MiSplice")] = "Splice-altering"
    return out


def mfass(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["category"] != "mutant" or row["snp_position"] in ("NA", ""):
                continue
            d = None
            for col in ("v2_dpsi", "v1_dpsi"):
                if row[col] != "NA":
                    try:
                        d = float(row[col])
                        break
                    except ValueError:
                        pass
            if d is not None:
                out[(row["chr"], int(row["snp_position"]), row["ref_allele"], row["alt_allele"], "MFASS")] = \
                    cad.classify_mfass_variant(d)
    return out


def literature():
    t = pd.read_csv(os.path.join(HERE, "literature_classifications.csv"), dtype=str)
    return {(c, int(p), r, a, d): k for c, p, r, a, d, k in
            zip(t.chr19, t.pos19, t.ref19, t.alt19, t.dataset, t.svdb_class)}


def restore(rel, cls, position):
    path = os.path.join(ROOT, rel)
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    head = rows[0]
    idx = [head.index(c) for c in ("chr19", "pos19", "ref19", "alt19", "dataset")]
    out, missing = [], 0
    for n, r in enumerate(rows):
        r = list(r)
        if n == 0:
            v = "svdb_class"
        else:
            c, p, ref, alt, d = (r[i] for i in idx)
            v = cls.get((c, int(float(p)), ref, alt, d))
            if v is None:
                missing += 1
                v = ""
        r.insert(position, v)
        out.append(r)
    buf = io.StringIO(newline="")
    csv.writer(buf, lineterminator="\n").writerows(out)
    with open(path, "w", newline="") as f:
        f.write(buf.getvalue())
    return missing


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--savnet", required=True)
    ap.add_argument("--misplice", required=True)
    ap.add_argument("--mfass", required=True)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
    for name in ("savnet", "misplice", "mfass"):
        got = md5(getattr(args, name))
        if got != SOURCE_MD5[name]:
            print(f"  note: {name} file md5 {got} differs from the reconstruction's copy ({SOURCE_MD5[name]}); "
                  "the output check below decides", flush=True)
    cls = {**savnet(args.savnet), **misplice(args.misplice), **mfass(args.mfass), **literature()}
    positions = json.load(open(os.path.join(ROOT, "scripts", "comparators", "stripped_columns.json")))
    want = {l.split(None, 1)[1].strip(): l.split()[0] for l in open(os.path.join(ROOT, "scripts", "comparators", "originals.md5"))}
    ok = True
    for rel in FILES:
        (pos, _), = positions[rel]
        missing = restore(rel, cls, pos)
        same = md5(os.path.join(ROOT, rel)) == want[rel]
        ok &= same and not missing
        print(f"  {'ok     ' if same else 'DIFFERS'} {rel}" + (f" ({missing} records without a classification)" if missing else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
