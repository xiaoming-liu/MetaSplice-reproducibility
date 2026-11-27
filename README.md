# MetaSplice Release 2.0: reproducibility package

Data, models and code behind the manuscript *MetaSplice: pathogenicity prediction and ACMG/AMP evidence for intronic
variants outside the canonical splice sites*. `MANIFEST.md` maps each result in the manuscript to the command that
produces it and the file that holds it. MetaSplice itself (scorer, Docker image, data bundle) is at
https://github.com/xiaoming-liu/MetaSplice.

## Changes in release 2.0.2

- **Calibration at a lower bound on the prior.** The PP3/BP4 thresholds are now derived at a lower bound on the prior
  probability of pathogenicity for rare non-canonical intronic variants, in place of the missense prior 0.0441:
  1,099 P/LP variants (at least one review star, training release) among the 81,116,352 rare intronic SNVs that
  gnomAD v4.1 observes in the calibration's regions and genes, 1.35 × 10⁻⁵. `scripts/analysis/S15a_prior_population.py`
  builds that population and checks the numerator against it (new data file `data/v2_gnomad_eligible_clinvar.csv`);
  `scripts/analysis/S15_lower_bound_prior.py` derives the thresholds and evaluates the resulting rule on the test set.
- **Only what the manuscript uses.** The package now holds the analyses, build steps, data and results behind the
  manuscript's results. The study's other analyses, including the calibration at the missense prior (S5, S6, X3, Q2),
  are not included; release v2.0.1 keeps them.
- **Figures.** `scripts/figures/` draws the manuscript's Figures 1–4 and Figure S1, and `figures/` holds them.
- **Self-contained scripts.** Definitions that the reported analyses took from unreported ones are now in the analyses
  themselves, and script descriptions and log labels describe the analyses rather than the study's planning documents.
  The changed analyses were rerun: every number in their results is unchanged.
- **Fix.** In a fresh copy, `run_spliceai_raw.py --check-only` stopped at V11c, because V11c listed the variants left
  to score from V11's and V11b's output before its installation check. V11c now skips that list in a check-only run,
  and `run_spliceai_raw.py` stops with an instruction when `lookup_comparators.py` has not been run first. A full run
  of V11c is unchanged.
- The model, its scores and the data tables of release 2.0.1 are unchanged.

## What is here

| Path | Contents |
|---|---|
| `data/` | Downloaded by `get_data.py` from this repository's release (665 MB unpacked). Variant sets with their 38 features, labels and regions: training (ClinVar 2025-08-31), test candidates and the temporal test set (ClinVar 2026-09-28), VUS, literature, SpliceVarDB, gnomAD reference samples. Also out-of-fold scores, the gnomAD reference scores, the ClinVar variants of the prior's population (`v2_gnomad_eligible_clinvar.csv`), and the two ClinVar 2025-08-31 summaries the cohort loaders read (`data/clinvar/`). |
| `models/` | The Release 2.0 model (`metasplice_v2_full.joblib`), its five cross-validation fold models, the feature order and the fold map |
| `scripts/analysis/` | The analyses behind the manuscript's results |
| `scripts/build/` | The build steps that made `data/` and `models/` (V00–V12, R01), and `gene_level/`, which builds the gene-level values from their sources. They need the raw inputs: ClinVar, the annotation data, gnomAD, and GPU runs of SpliceTransformer and Pangolin. See below. |
| `scripts/comparators/` | Rebuild the SpliceAI and CADD scores, which are not redistributed (below) |
| `scripts/splicevardb/` | Restore the SpliceVarDB classifications from the source studies' files (below) |
| `scripts/figures/` | Figures 1–4 and Figure S1 of the manuscript |
| `scripts/score_v2.py` | The MetaSplice Release 2.0 scorer, whose definitions the analyses import |
| `results/` | The output of every analysis and build step |
| `figures/` | The manuscript's figures, and the mixture-fit figure written by `S11_prior.py` |
| `tests/` | Unit tests of the calibration code |
| `get_data.py` | Download `data/` from the release, check the archive's SHA-256 and every file's md5 |
| `rerun.sh`, `env.sh` | Rerun everything; set the paths for running a script by hand |
| `MD5SUMS` | md5 of every file |

## Quick start

    pip install -r requirements.txt        # Python 3.12; the versions the analyses ran with
    python get_data.py                     # downloads and checks data/ (278 MB download)
    ./rerun.sh                             # writes results/ and figures/; logs in results/logs/

Without SpliceAI and CADD scores and SpliceVarDB classes, `rerun.sh` runs the analyses that use MetaSplice scores only
(the literature set and the gnomAD reference scores), draws the figures from the stored results, and runs the unit
tests. It lists every other analysis as skipped, with the command that enables it.

## SpliceAI and CADD scores

Most analyses compare MetaSplice with SpliceAI and CADD. SpliceAI's precomputed scores are free for academic and
not-for-profit use, and CADD's for non-commercial applications; neither grants redistribution. This package therefore
holds the variants but not their SpliceAI or CADD scores:
- left out: `data/v2_comparators.csv` and the three `data/v2_spliceai_raw_d4999*.csv` files;
- removed: the `spliceai_max` and `cadd_phred` columns of `data/v2_features_pooled_test.csv` and
  `data/v2_features_fresh_20260928.csv`.

To rebuild them exactly, obtain the sources under their own terms and run the build scripts' own code:

1. **SpliceAI v1.3 precomputed scores** (Illumina; `spliceai_scores.masked.snv.hg19.vcf.gz`) and **CADD v1.7 GRCh38**
   (`whole_genome_SNVs.tsv.gz` with its index), plus the **MetaSplice Release 2.0 data bundle** (its GRCh37 FASTA):

       python scripts/comparators/lookup_comparators.py --split-spliceai spliceai_scores.masked.snv.hg19.vcf.gz --out spliceai_by_chr
       python scripts/comparators/lookup_comparators.py --spliceai spliceai_by_chr --cadd whole_genome_SNVs.tsv.gz --bundle metasplice_v2_data

2. **Raw SpliceAI scores** (±4,999 nt, the configuration ClinGen calibrated) need SpliceAI 1.3.1 installed in its own
   environment:

       python scripts/comparators/run_spliceai_raw.py --spliceai-env /path/to/spliceai-env --bundle metasplice_v2_data

Each script checks every file it rebuilds against the md5 of the original (`scripts/comparators/originals.md5`). Then
run `./rerun.sh` again.

## SpliceVarDB classifications

The SpliceVarDB set was rebuilt for this study from SpliceVarDB's source studies with SpliceVarDB's published criteria,
not downloaded from SpliceVarDB. Its 22,908 records come from MFASS (12,692), SAVNet (9,304), MiSplice (898), Casadei et
al. 2019 (12) and Fraile-Bethencourt et al. 2019 (2). The tables keep each record's variant, source study and regions,
but not its class (`svdb_class`), because the source studies' terms do not grant redistribution. To restore the classes
exactly, download three files and run:

    python scripts/splicevardb/restore_splicevardb.py --savnet Supplemental_Table_S3.xlsx \
        --misplice 1-s2.0-S2211124718303978-mmc2.xlsx --mfass snv_data_clean.txt

- `Supplemental_Table_S3.xlsx`: supplementary data of Shiraishi et al., Genome Res 2018 (SAVNet);
- `1-s2.0-S2211124718303978-mmc2.xlsx`: supplementary file mmc2 of Jayasinghe et al., Cell Rep 2018 (MiSplice);
- `snv_data_clean.txt`: `processed_data/snv/` of the MFASS repository, https://github.com/KosuriLab/MFASS
  (Chong et al., Mol Cell 2019).

The 14 records from Casadei et al. and Fraile-Bethencourt et al. were extracted by hand from PDF tables. Their classes
are in `scripts/splicevardb/literature_classifications.csv`. The script applies the reconstruction's own rules
(`scripts/splicevardb/classify_all_datasets.py`) and checks both tables against their original md5.

## Building the data from raw inputs

`scripts/build/` documents how `data/` and `models/` were made. The scripts read the raw inputs at the paths of the
study's working tree, recorded with their md5 in `results/V00_inputs.txt`. The annotation data, with a script that
fetches and builds them from their original sources, form the MetaSplice Release 2.0 data bundle, mirrored on USF Box
(https://usf.box.com/s/44v8lpk6rxi8d6vdvw1032cfr5olmzdq). The MetaSplice Docker image runs SpliceTransformer and
Pangolin and computes the 38 features from a VCF exactly as the build did. In its acceptance test, 2,200 ClinVar
variants scored from scratch on the GPU were each bit-identical to the stored full-precision score. The image is
`ghcr.io/xiaoming-liu/metasplice:2.0.1` (manifest digest
`sha256:c32fc8d49da016efd39bc0ac3738ee46d873476e775c0a01a6c714621914f105`, image ID
`sha256:2dee3c2ef835843b08770a6f6b93305410b94f4b75b3c0f80c19f5819610a4bd`). `data/v2_features_union.tsv`, an
intermediate table of the feature build, is not included; the per-set feature tables come from it.

**The prior's population.** `scripts/analysis/S15a_prior_population.py` repeats the gnomAD reference build
(`scripts/build/V02b_gnomad_reference.py`) on the data bundle's gnomAD v4.1 tables with the ClinVar variants kept, and
writes `data/v2_gnomad_eligible_clinvar.csv` and `results/S15a_prior_population.txt`. It checks that the variants in
neither ClinVar release reproduce V02b's count in every region, and that the numerator equals the training
calibration cohort's P/LP variants with an allele frequency below 0.01.

**Gene-level values.** `scripts/build/R01_gene_level_extract.py` took the three gene-level values (gnomAD pLI and
LOEUF, ClinGen haploinsufficiency score) from the dbNSFP 5.4 gene file. `scripts/build/gene_level/build_gene_level.py`
builds the same file from the original sources, gnomAD, ClinGen and HGNC (ClinGen and HGNC under CC0; gnomAD as open
data, with CC0 named for its primary data), and its output is byte-identical (md5 `088ba4d5101c01a8697fbbf503affd06`).
It needs the GENCODE v50 GTF from the data bundle and downloads the gnomAD files;
`scripts/build/gene_level/README_gene_level.md` describes it.

**Retraining.** `scripts/build/V06_train.py` retrains the model from `data/v2_features_training.csv`. Note that it
writes into `models/`, replacing the package's files, so run it in a copy of the package. In an independent rerun:
- the five fold models, the feature list, the fold map and the out-of-fold scores were byte-identical;
- the retrained full model gave identical predictions on all 413,199 training and 46,112 test variants;
- its file still differed, in one stored training statistic (a tree's sum of Hessians) in the last digit, which
  multithreaded training can change, so its md5 differed.

The analyses lock the md5 of the package's `models/metasplice_v2_full.joblib` and stop with another file. Run them
with the package's model, and compare a retrained model by its predictions, not its md5. A different checksum here is
not a change in any result.

**Build logs.** The logs of the build steps were written when the data were built and were not regenerated for this
release. Six lines in them still carry the study's internal labels: the headers of `V11_spliceai_raw.txt`,
`V11b_spliceai_raw_vus.txt` and `V11c_spliceai_raw_topup.txt` (protocol amendment numbers), the V11 cohort line ("X3's
four cohorts", the four calibration cohorts now defined in `V11_spliceai_raw.py` itself), the V01b section header of
`V01_annotation.txt`, and the deep-intronic row of `V06_train.txt` ("(O5)"). No value is affected.

## Verification

Checked on 2026-11-26 in clean copies of this package:

- **Without SpliceAI or CADD scores.** In a copy without `data/`, `get_data.py` unpacked the release archive, and all
  23 data files matched `MD5SUMS`. `rerun.sh` then ran in under a minute: the literature set, the gnomAD reference
  scores, Figures 1–4 and S1, and the 15 unit tests (all passed). It listed the other six analyses as skipped. Every
  file still matched `MD5SUMS` afterwards. In our environment the figures were byte-identical to the manuscript's;
  elsewhere fonts and rendering can differ.
- **Comparator scores rebuilt.** `lookup_comparators.py` ran on our copies of the SpliceAI and CADD files in 15
  minutes, and the three files it rebuilds match the originals' md5s. In a fresh copy, `run_spliceai_raw.py
  --check-only` then passed the installation checks of V11, V11b and V11c (500 variants each). With V11's and V11b's
  output copied from the originals, a full run of V11c rebuilt its file with the original md5. V11 and V11b were not
  rerun in full for this release. Their scripts changed only in descriptions and messages, and in V11's cohort
  definition, which now sits in V11 itself and gives the same 37,251 variants. In release 2.0.1's check,
  `run_spliceai_raw.py` rebuilt all three raw-score files in 4.5 hours, each matching its original md5.
- **SpliceVarDB classes restored.** `restore_splicevardb.py`, given the three source files, restored both SpliceVarDB
  tables exactly (md5) in 3 seconds.
- **Every analysis.** With these inputs in place, `rerun.sh` ran every analysis in 29 minutes. The results and figures
  it wrote are byte-identical to the stored ones, except for provenance lines in four files: the paths of the two
  ClinVar summaries (same md5s) in the logs of P1, P2 and S15, and the creation time in `S11_prior_frozen.json`. No
  value differs.
- **The prior's population.** `S15a_prior_population.py`, run on the raw gnomAD and ClinVar inputs in 9 minutes with
  12 processes, rebuilt `data/v2_gnomad_eligible_clinvar.csv` byte-identically. Its log differs from the stored one
  only in the per-chromosome run times.

## Licences

The MetaSplice code, model and derived data are under the MetaSplice licence (`LICENSE`). It grants permission "to
use, copy, modify, and distribute MetaSplice and its associated model files for non-commercial research, educational,
and clinical (non-revenue-generating) purposes, subject to the following conditions", and states: "For commercial use
— including use within commercial products, fee-for-service offerings, and internal use by for-profit entities —
please contact xiaomingliu@usf.edu." The data include third-party content under its own terms (`THIRD_PARTY_NOTICES`).
