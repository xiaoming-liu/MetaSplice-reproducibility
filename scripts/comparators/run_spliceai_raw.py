"""Rebuild the raw SpliceAI scores of this deposit (data/v2_spliceai_raw_d4999*.csv) with your own SpliceAI 1.3.1.

ClinGen calibrated SpliceAI on raw scores over +-4,999 nt (Walker et al. 2023), which the precomputed files do not hold.
The build scripts V11, V11b and V11c computed them with SpliceAI 1.3.1 (-A grch37 -D 4999 -M 0). This script runs those
three scripts unchanged, pointed at your SpliceAI installation and the data bundle's GRCh37 FASTA:

  python run_spliceai_raw.py --spliceai-env /path/to/env --bundle metasplice_v2_data [--chunks 4] [--check-only]

  --spliceai-env  the prefix of a Python environment with spliceai==1.3.1 and TensorFlow (bin/spliceai, bin/python);
                  each script first checks SpliceAI's version and the md5 of its GRCh37 annotation and model files
  --check-only    the installation check only: SpliceAI in its precomputed configuration (-D 50 -M 1) on a seeded
                  sample must reproduce the precomputed scores (>= 99% within 0.01)

Run lookup_comparators.py first: the cohorts these scripts score are defined with the precomputed SpliceAI scores, and
V11c reads the restored pooled test set. Each output is checked against originals.md5. V11 took 94 minutes for 37,249
variants in 4 processes on our machine; V11b scores about 62,000 variants. The scripts rewrite their logs in results/.
"""
import argparse, hashlib, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BUILD = os.path.join(ROOT, "scripts", "build")
os.environ.setdefault("METASPLICE_V2_SCORER", os.path.join(ROOT, "scripts", "score_v2.py"))
os.environ.setdefault("METASPLICE_V2_CLINVAR_2025_GRCH38", os.path.join(ROOT, "data", "clinvar", "clinvar_20250831.GRCh38.summary.gz"))
os.environ.setdefault("METASPLICE_V2_CLINVAR_2025_GRCH37", os.path.join(ROOT, "data", "clinvar", "clinvar_20250831.GRCh37.summary.gz"))
sys.path.insert(0, HERE)
from lookup_comparators import load, md5, originals     # noqa: E402

STEPS = [("V11", "V11_spliceai_raw.py", "data/v2_spliceai_raw_d4999.csv"),
         ("V11b", "V11b_spliceai_raw_vus.py", "data/v2_spliceai_raw_d4999_vus.csv"),
         ("V11c", "V11c_spliceai_raw_topup.py", "data/v2_spliceai_raw_d4999_topup.csv")]


def point(v11, v08, env, pkg, hg19, work):
    v11.ENV, v11.PKG, v11.HG19, v11.TMP, v11.v08 = env, pkg, hg19, work, v08


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spliceai-env", required=True)
    ap.add_argument("--bundle", required=True, help="MetaSplice Release 2.0 data bundle (grch37/hg19.fa)")
    ap.add_argument("--work", default=None, help="folder for SpliceAI's input and output VCFs (default: a temporary one)")
    ap.add_argument("--chunks", type=int, default=4, help="SpliceAI processes")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--only", default="V11,V11b,V11c")
    args = ap.parse_args()
    env = os.path.abspath(args.spliceai_env)
    pkg = subprocess.run([f"{env}/bin/python", "-c", "import os, spliceai; print(os.path.dirname(spliceai.__file__))"],
                         capture_output=True, text=True, check=True, env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")).stdout.strip()
    hg19 = os.path.join(os.path.abspath(args.bundle), "grch37", "hg19.fa")
    work = args.work or tempfile.mkdtemp(prefix="metasplice_spliceai_raw_")
    root = os.path.join(work, "root")              # V08's paths derive from one root; only its GRCh37 FASTA is used here
    os.makedirs(os.path.join(root, "hg19"), exist_ok=True)
    for ext in ("", ".fai"):
        link = os.path.join(root, "hg19", "hg19.fa" + ext)
        if not os.path.lexists(link):
            os.symlink(hg19 + ext, link)
    v08 = load("v08", os.path.join(BUILD, "V08_comparators.py"),
               ('P = "/mnt/Work0/Splice_altering_variant"', f'P = "{root}"'))
    want, ok = originals(), []
    for name, script, out in STEPS:
        if name not in args.only.split(","):
            continue
        print(f"== {name}", flush=True)
        mod = load(name.lower(), os.path.join(BUILD, script))
        point(mod if name == "V11" else mod.v11, v08, env, pkg, hg19, work)
        if name == "V11":
            mod.v08 = v08
        sys.argv = [script, "--chunks", str(args.chunks)] + (["--check-only"] if args.check_only else [])
        mod.main()
        if not args.check_only:
            got = md5(os.path.join(ROOT, out))
            ok.append(got == want[out])
            print(f"  {'ok     ' if ok[-1] else 'DIFFERS'} {out}", flush=True)
    if not args.check_only:
        print(f"{sum(ok)} of {len(ok)} files equal the originals")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
